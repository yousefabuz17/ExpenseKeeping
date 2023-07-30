import asyncio
import functools
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime as dt
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, OrderedDict

import aiohttp
import aiohttp_cache
import black
import magic
import matplotlib.pyplot as plt
import pandas as pd
import psycopg
import requests_cache
import seaborn as sns
from babel.numbers import get_currency_symbol
from matplotlib import dates as mpl_dates
from mindee import Client, documents
from pydantic import BaseModel
from wordcloud import WordCloud

requests_cache.install_cache(expire_after=7200)

@dataclass
class Args:
    arg1: Optional[List[str]] = None
    arg2: Optional[str] = None
    arg3: Optional[str] = None

@dataclass
class FileInfo(BaseModel):
    name: Optional[str] = None
    lang: Optional[str] = None
    type_: Optional[str] = None
    path: Optional[str] = None
    category: Optional[str] = None
    subcat: Optional[str] = None
    contents: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    amount: Optional[str] = None
    vendor: Optional[str] = None
    currency: Optional[str] = None
    symbol: Optional[str] = None
    note: Optional[str] = None

class ConfigInfo(NamedTuple):
    host: Optional[str] = None
    dbname: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    apis: OrderedDict = None
    
    @dataclass
    class DirInfo:
        dirs: Dict = None
    
    @lru_cache(maxsize=None)
    @staticmethod
    def get_config():
        config_parser = ConfigParser(allow_no_value=True)
        config_parser.read('config.ini')
        config = ConfigInfo(*[val for _, val in config_parser.items('Database')],
                            apis=OrderedDict(config_parser.items('API')))
        return config
    
    @lru_cache(maxsize=None)
    @staticmethod
    def get_dir():
        config_parser = ConfigParser(allow_no_value=True)
        config_parser.read('config.ini')
        directories = ConfigInfo.DirInfo(dirs=dict(config_parser.items('Directory')))
        return directories

@dataclass
class Currency(BaseModel):
    rate: float
    sym: str
    
    @aiohttp_cache.cache(expires=7200)
    async def get_rates():
        config = ConfigInfo.get_config()
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(config.apis['currency_url'],
                                    params={'app_id':config.apis['currency_api'],'base':'USD'}) as response:
                if response.status == 200:
                    response = await response.json()
                    data = response['rates']
                    currency_data = OrderedDict({key: Currency(rate=value, sym=get_currency_symbol(key, locale="en_US")[-3:]) for key, value in data.items()})
                    return currency_data
    
    @staticmethod
    async def exchange_rate(total, from_curr, to_curr='USD'):
        currency_data = await Currency.get_rates()
        if from_curr==to_curr:
            return f'${total.strip("$")}'
        rate = Decimal(currency_data[to_curr].rate) / Decimal(currency_data[from_curr].rate)
        return f'{currency_data[to_curr].sym}{rate * Decimal(total.strip("$")):.2f}'

class ExpenseDB:
    def __init__(self, data):
        self.data = data
        self.sql_script = Args(*open(Path(__file__).parent.absolute() / 'expense_db.sql').read().split('\n\n'))
        self.connection = None
        self.cursor = None
    
    def _sql_connect(self):
        if self.connection is None or self.cursor is None:
            self.config = ConfigInfo.get_config()
            try:
                self.connection = psycopg.connect(
                    host=self.config.host,
                    dbname=self.config.dbname,
                    user=self.config.user,
                    password=self.config.password
                )
                self.cursor = self.connection.cursor()

            except (psycopg.ConnectionFailureError, psycopg.ConfigFileError, psycopg.ConnectionDoesNotExistError) as e:
                self.connection = None
                self.cursor = None
                return f'Error failed while trying to connect to database {e}'

    def update_db(self):
        self._sql_connect()
        data = self.data.to_records(index=False).tolist()
        query1 = self.sql_script.arg1
        query2 = self.sql_script.arg2
        self.cursor.execute(query1)
        self.connection.commit()
        for _, item in enumerate(data):
            data_ = FileInfo(date=item[0],
                            time=item[1],
                            lang=item[2],
                            currency=item[3],
                            symbol= item[4],
                            category=item[5],
                            subcat=item[6],
                            amount=item[7],
                            vendor=item[8],
                            note=item[10]
                            )
            self.cursor.execute(query2, (data_.date,
                                            data_.time,
                                            data_.lang,
                                            data_.currency,
                                            data_.symbol,
                                            data_.category,
                                            data_.subcat,
                                            data_.amount,
                                            data_.vendor,
                                            data_.note))
        
        self.connection.commit()
        self._close_db()

    def _close_db(self):
        if self.connection:
            try:
                self.connection.rollback()
                print(f"\nDatabase Updated Successfully.")
            except psycopg.Error as e:
                print(f"An error occurred during transaction rollback: {e}")
            self.cursor.close()
            self.connection.close()
            print(f"Database Server Closed.")


class Wrapper:
    def __init__(self):
        pass
    
    @staticmethod
    def _receipt_writer():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                files = await func(*args)
                receipt_data = {}
                for _, file in enumerate(files):
                    if file.type_ == 'pdf':
                        receipt_data[file.name] = {
                                    'Date': str(file.date),
                                    'Time': str(file.time),
                                    'Language': str(file.lang),
                                    'Currency': file.currency,
                                    'Category': str(file.category),
                                    'Sub-Category': str(file.subcat),
                                    'Amount': str(file.amount),
                                    'Vendor': str(file.vendor),
                                    'Contents': str(file.contents)
                                }
                    else:
                        pass
                    
                with open(Path(__file__).parent.absolute() / 'receipt_info.json', 'w') as receipt_info:
                        json.dump(receipt_data, receipt_info, indent=2)
                        
                return files
            return wrapper
        return decorator
    
    @staticmethod
    def _receipt_parser():
        def decorator(func):
            async def wrapper(*args):
                files = await func(*args)
                new_files = []
                file_type_mapping = {
                        'json': pd.read_json,
                        'csv': pd.read_csv,
                        'xls': pd.read_excel
                    }
                mindee_client = Client(api_key=ConfigInfo.get_config().apis['mindee_api'])
                for _, file in enumerate(files):
                    path_ = Path(file)
                    file = file.as_posix()
                    # file_name = (file_name := re.findall(r'(\w+)\.[pdf|json|csv|png|jpeg|jpg|txt]', file)) and file_name[0]
                    # type_ = (type_ := re.findall(r'\.(pdf|json|csv|png|jpeg|jpg|txt)$', file.lower())) and type_[0]
                    type_ = magic.from_file(path_, mime=True).split('/')[-1]
                    file = FileInfo(type_=type_, path=path_)
                    if file.type_ in ('jpeg', 'png', 'jpg', 'pdf'):
                        input_doc = mindee_client.doc_from_path(file.path)
                        api_response = input_doc.parse(documents.TypeReceiptV5)
                        file.name = api_response.document.filename
                        file.category = api_response.document.category
                        file.subcat = api_response.document.subcategory
                        file.date = api_response.document.date
                        file.amount = api_response.document.total_amount
                        file.lang = api_response.document.locale
                        file.vendor = api_response.document.supplier_name
                        file.contents = api_response.document
                        file.time = api_response.document.time
                        file.currency = str(api_response.document.locale).split(';')[-2].lstrip()
                        new_files.append(file)
                    
                    if file.type_ in file_type_mapping:
                        file.contents = file_type_mapping[file.type_](file.path)
                        new_files.append(file)
                    
                return new_files
            return wrapper
        return decorator
    
    @staticmethod
    def _modify_json():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                json_file = await func(*args)
                for _, item in json_file.items():
                    try:
                        modified_date = dt.strptime(item['Date'], '%Y-%m-%d').strftime('%m-%d-%Y')
                        item['Date'] = modified_date
                        modified_time = dt.strptime(item.get('Time', ''), '%I:%M').strftime('%I:%M %p')
                        item['Time'] = modified_time
                    except ValueError:
                        modified_currency = await Currency.exchange_rate(total=item['Amount'],
                                                                        from_curr=item['Currency'])
                        item['Amount'] = modified_currency
                        language = item['Language'].split(';')[-3].lstrip()
                        item['Language'] = language if language.isupper() else item['Language'].split(';')[-2].lstrip()
                    new_json = OrderedDict(sorted(json_file.items(), key=lambda i: float(i[1]['Amount'].strip('$'))))
                    with open(Path(__file__).parent.absolute() / 'modified_receipts.json', 'w') as new_file:
                        json.dump(new_json, new_file, indent=2)
                return new_json
            return wrapper
        return decorator
    
    @staticmethod
    def _json_to_pd():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                json_file = await func(*args)
                columns = set()
                rows = []
                for _, item in enumerate(json_file.items()):
                    for contents in item[1].items():
                        columns.add(contents[0])
                    rows.append(item[1])
                df = pd.DataFrame(rows)
                return df
            return wrapper
        return decorator
    
    @staticmethod
    def _clean_pd():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                df = await func(*args)
                pd.set_option('display.max_columns', None)
                df["Time"] = df["Time"].replace("", float("nan")).str.strip()
                df['Time'].fillna('00:00', inplace=True)
                df['Time'] = df['Time'].apply(lambda time: time if time=='00:00' else dt.strptime(time[:5], '%H:%M').strftime('%I:%M %p'))
                df['Vendor'] = df['Vendor'].replace('', float('nan')).str.strip()
                df['Vendor'].fillna('UNKNOWN', inplace=True)
                df['Sub-Category'] = df['Sub-Category'].replace('', float('nan')).str.strip()
                df['Sub-Category'].fillna('unknown', inplace=True)
                # df['Contents'] = df['Contents'].apply(lambda i: base64.b64encode(i.encode()))
                language = str(df['Language']).split(';')[-3].lstrip()
                df['Language'] = language if language.isupper() else df['Language'].split(';')[-2].lstrip()
                df['Amount'] = df['Amount'].apply(lambda i: float(i.strip('$')))
                df.index = range(1, len(df)+1)
                return df
            return wrapper
        return decorator
    
    @staticmethod
    def _merge_all():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                csv_files = await func(*args)
                csvs = csv_files.arg1
                df = csv_files.arg2
                df['Category'] = df['Category'].apply(lambda i: i.title())
                csvs_files = []
                for _, csv in enumerate(csvs):
                    read_csv = pd.read_csv(csv)
                    new_csv = read_csv.loc[:, ['Date', 'Category', 'Note', 'Amount', 'Currency']]
                    csvs_files.append(new_csv)
                    new_df = pd.concat([df, new_csv])
                return new_df
            return wrapper
        return decorator
    
    @staticmethod
    def _clean_merged():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                df = await func(*args)
                df['Vendor'].fillna('UNKNOWN', inplace=True)
                df['Language'].fillna('EN', inplace=True)
                df['Sub-Category'].fillna('unknown', inplace=True)
                df['Time'] = df['Time'].apply(lambda i: dt.strptime(i, '%H:%M').strftime('%I:%M %p') if i=='00:00' else i)
                df['Note'].fillna('N/A', inplace=True)
                df['Contents'].fillna('N/A', inplace=True)
                # df['Contents'] = df['Contents'].apply(lambda i: base64.b64encode(i.encode('utf-8')) if i!='N/A' else 'N/A')
                date_time = [Args(*i.split()) for i in df['Date'].iloc[29:].values.tolist()]
                df.reset_index(drop=True, inplace=True)
                df.loc[29:, 'Date'] = [dt.strptime(i.arg1, '%m/%d/%Y').strftime('%m-%d-%Y') for i in date_time]
                df.loc[29:, 'Time'] = [dt.strptime(i.arg2, '%H:%M').strftime('%I:%M %p') for i in date_time]
                df.loc[29:, 'Date'] = [dt.strptime(i.arg1, '%m/%d/%Y').strftime('%m-%d-%Y') for i in date_time]
                df.loc[29:, 'Time'] = [dt.strptime(i.arg2, '%H:%M').strftime('%I:%M %p') for i in date_time]
                currency_sym = df['Currency'].apply(get_currency_symbol)
                insert_index = df.columns.get_loc('Currency') + 1
                df.insert(insert_index, 'Symbol', currency_sym)
                return df
            return wrapper
        return decorator


class Plotter:
    def __init__(self):
        pass
    
    #TODO:
        #^//: Time Series Plot- Plot the amount spent over time to see spending trends and patterns. Use a line plot or an area plot for this.
        #^//: Bar-Chart (Category, Sub-Category)- total spending for each category or sub-category. Helps identify which categories account for the most spending.
        #^: Stacked Bar Chart: Visualize the total spending for each category, segmented by language or currency. Helps compare spending across different regions or currencies.
        #^: Grouped Bar Chart: Group expenses by language or currency and use a bar chart to compare spending across these groups.
        #^//: Pie Chart: Use a pie chart to visualize the percentage of spending for each category. Quick overview of the distribution of expenses.
        #^: Histogram: Create a histogram to understand the distribution of expense amounts. Helps identify the most common spending range.
        #^: Box Plot: Use a box plot to visualize the spread of expense amounts for each category. Shows the median, quartiles, and outliers.
        #^: Line Plot: Use a line plot to visualize the spread of expense amounts for each language or currency.
        #^//: Scatter Plot: Plot the expenses against time or amount to look for any patterns or correlations.
        #^//: Word Cloud: Create a word cloud using the notes column to visualize the most frequently mentioned words or phrases in your expenses.
    
    # @staticmethod
    # def _plot_all()
    
    @staticmethod
    def _scattplot_time_series():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                dataframe = await func(*args)
                sns.set_style('darkgrid')
                df = dataframe.groupby('Date')['Amount'].sum().reset_index()
                df['Date'] = pd.to_datetime(df['Date'])
                
                _, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(df['Date'], df['Amount'], c=pd.DatetimeIndex(df['Date']).year,
                                    cmap='rainbow', marker='o', edgecolors='blue', alpha=1)
                
                date_format = mpl_dates.DateFormatter('%b, %d %Y')
                ax.xaxis.set_major_formatter(date_format)
                plt.xlabel('Date')
                plt.ylabel('Total Amount Spent per Day')
                plt.title('Expense Time Series by Year', pad=10)
                plt.xticks(rotation=20, ha='right')
                
                legend = plt.legend(*scatter.legend_elements(), title='Year')
                plt.gca().add_artist(legend)
                
                plt.tight_layout()
                plt.show()
                return df
            return wrapper
        return decorator
    
    @staticmethod
    def _word_cloud_cats():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                df = await func(*args)
                notes = ' '.join([i.title() for i in pd.concat([df['Category'], df['Sub-Category']]) if i != 'unknown'])
                wordcloud = WordCloud(width=800, height=400,
                                    background_color='white', collocations=False).generate(notes)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.show()
                return df
            return wrapper
        return decorator
    
    @staticmethod
    def _word_cloud_notes():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                df = await func(*args)
                notes = ' '.join([i for i in pd.concat([df['Note'], df['Vendor']]) if i != 'N/A' and i != 'UNKNOWN'])
                wordcloud = WordCloud(width=800, height=400,
                                    background_color='white', collocations=False,
                                    min_word_length=3).generate(notes)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.show()
                return df
            return wrapper
        return decorator
    
    @staticmethod
    def _pie_chart_cat():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                df = await func(*args)
                sns.set_style('white')
                category_totals = df.groupby('Category')['Amount'].sum()
                category_totals = category_totals[category_totals > 200]
                categories = category_totals.index.tolist()
                amounts = category_totals.values.tolist()
                _, ax = plt.subplots(figsize=(12, 6))
                ax.pie(amounts, labels=categories, shadow=True, rotatelabels=True,
                                labeldistance=1.1, autopct='%1.2f%%')
                plt.title('Pie Chart per Category', pad=10)
                plt.xticks(rotation=20, ha='right')
                plt.show()
                return df
            return wrapper
        return decorator
    
    @staticmethod
    def _barplot_cats():
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args):
                df = await func(*args)
                sns.set_style('darkgrid')
                category_totals = df.groupby('Category')['Amount'].sum()
                categories = category_totals.index.tolist()
                amounts = category_totals.values.tolist()
                _, ax = plt.subplots(figsize=(12, 6))
                ax.bar(categories, amounts, color=plt.colormaps['tab10'](amounts))
                plt.xlabel('Categories')
                plt.ylabel('Total Amount', labelpad=20)
                plt.title('Total Amount per Category', pad=10)
                plt.xticks(rotation=20, ha='right')
                plt.tight_layout()
                plt.show()
                return df
            return wrapper
        return decorator
    


class ExpenseKeeping:
    _instance = False
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.dirs = Path(__file__).parent.absolute() / ConfigInfo.get_dir().dirs['dir']
            self.files = None
            self.df = None

    @aiohttp_cache.cache(expires=7200)
    @Wrapper._receipt_writer()
    @Wrapper._receipt_parser()
    async def parse_receipts(self):
        self.files = [Path(root) / i for root, _, files in os.walk(self.dirs) for i in files if files]
        return self.files

    @Wrapper._modify_json()
    async def receipt_json(self):
        receipt_file = json.load(open(Path(__file__).parent.absolute() / 'receipt_info.json', encoding='utf-8'))
        return receipt_file
    
    @aiohttp_cache.cache(expires=7200)
    @Wrapper._clean_pd()
    @Wrapper._json_to_pd()
    async def get_pd(self):
        modified_json = json.load(open(Path(__file__).parent.absolute() / 'modified_receipts.json', encoding='utf-8'))
        return modified_json
    
    @Wrapper._clean_merged()
    @Wrapper._merge_all()
    async def parse_csvs(self, df):
        self.csv_files = [Path(root) / i for root, _, files in os.walk(self.dirs) for i in files if files and magic.from_file(Path(Path(root) / i), mime=True).split('/')[-1] == 'csv']
        zipped = Args(arg1=self.csv_files, arg2=df)
        return zipped
    
    # async def activate_parser(self):
    #     await asyncio.gather(
    #                 self.parse_receipts(),
    #                 self.receipt_json()
    #             )
    
@aiohttp_cache.cache(expires=7200)
async def main():
    #!> Make a method for asyncio/ThreadPool
    #!> If json file already exists and contains data, skip OCR process
    #!> Merge all plots into one figure by default
    #!> Make Error Handling class, all classes will inherit this
    #!> Make a final analysis output based on expense findings
    expense = ExpenseKeeping()
    
    # await asyncio.gather(
    #                 expense.parse_receipts(),
    #                 expense.receipt_json()
    #             )
    # await asyncio.gather(
    #     expense.get_pd(),
    #     expense.parse_csvs()
    # )
    df = await expense.get_pd()
    expense_df = await expense.parse_csvs(df)
    # expense_db = ExpenseDB(expense_df)
    # expense_db.update_db()
    # print(expense_df)
    
    
    #!?> Move this to ExpenseKeeping class
    # @Plotter._scattplot_time_series()
    # @Plotter._word_cloud_cats()
    # @Plotter._word_cloud_notes()
    @Plotter._pie_chart_cat()
    # @Plotter._barplot_cats()
    async def graph_expense(data):
        return data
    
    print(await graph_expense(expense_df))


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())