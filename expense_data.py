import asyncio
import functools
import json
import os
import re
import subprocess
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime as dt
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from pprint import pprint
from typing import NamedTuple
import base64

import aiohttp
import aiohttp_cache
import asyncpg
import black
import magic
import pandas as pd
import requests_cache
from babel.numbers import get_currency_symbol
from mindee import Client, documents

requests_cache.install_cache(expire_after=7200)


@dataclass
class Args:
    arg1: list|str = None
    arg2: str = None

@dataclass
class FileInfo:
    name: str = None
    lang: str = None
    type_: str = None
    path: str = None
    category: str = None
    subcat: str = None
    contents: str = None
    date: str = None
    time: str = None
    amount: str = None
    vendor: str = None
    currency: str = None
    
class ConfigInfo(NamedTuple):
    host: str = None
    dbname: str = None
    user: str = None
    password: str = None
    apis: OrderedDict = None
    
    @dataclass
    class DirInfo:
        dirs: dict = None
    
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
class Currency:
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
    config: ConfigInfo
    connection = None
    cursor = None
    
    @classmethod
    async def _sql_connect(cls):
        if cls.connection is None or cls.cursor is None:
            await cls.db_connect()
        return cls.connection, cls.cursor
    
    @classmethod
    async def db_connect(cls):
        cls.config = ConfigInfo.get_config()
        try:
            cls.connection = await asyncpg.connect(
                host=cls.config.host,
                dbname=cls.config.dbname,
                user=cls.config.user,
                password=cls.config.password)
            
            cls.cursor = cls.connection.cursor()

        except (asyncpg.ConnectionFailure, asyncpg.ConfigFileError, asyncpg.ConnectionException) as e:
            cls.connection = None
            cls.cursor = None
            return f'Error failed while trying to connect to database {e}'
    
    @classmethod
    def update_db(cls):
        pass


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
                df['Amount'] = df['Amount'].apply(lambda i: '{}{}'.format('$' if '$' not in i else '', i))
                df.index = range(1, len(df)+1)
                return df
            return wrapper
        return decorator
    
    @staticmethod
    def _merge_all():
        def decorator(func):
            async def wrapper(*args):
                csv_files = await func(*args)
                csvs = csv_files.arg1
                df = csv_files.arg2
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
            async def wrapper(*args):
                df = await func(*args)
                df['Vendor'].fillna('UNKNOWN', inplace=True)
                df['Language'].fillna('N/A', inplace=True)
                df['Sub-Category'].fillna('unknown', inplace=True)
                df['Note'].fillna('N/A', inplace=True)
                df['Contents'].fillna('N/A', inplace=True)
                df['Contents'] = df['Contents'].apply(lambda i: base64.b64encode(i.encode('utf-8')) if i!='N/A' else 'N/A')
                df['Amount'] = df['Amount'].apply(lambda i: '{}{}'.format('$' if '$' not in str(i) else '', i))
                date_time = [Args(*i.split()) for i in df['Date'].iloc[29:].values.tolist()]
                df['Date'].iloc[29:] = [dt.strptime(i.arg1, '%m/%d/%Y').strftime('%m-%d-%Y') for i in date_time]
                df['Time'].iloc[29:] = [dt.strptime(i.arg2, '%H:%M').strftime('%I:%M %p') for i in date_time]
                return df
            return wrapper
        return decorator



class TextExtractor:
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
    
@aiohttp_cache.cache(expires=7200)
async def main():
    text_extract = TextExtractor()
    
    # parsed_receipts = await asyncio.gather(
    #                 text_extract.parse_receipts(),
    #                 text_extract.receipt_json()
    #             )
    df = await text_extract.get_pd()
    # print(df)
    print(await text_extract.parse_csvs(df))
    
#!> Merge other csv files

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

#!> ReceiptProcessor class
    #?> Input various files for each type of expense bill for ML
    #?> Checks receipt type, if PDF checks expense category else return CSV/JSON
    #?> Returns the detected receipt type
    #! Steps:
        #?> Gather and Prepare Data:
            #? Collect labeled data for different types of expense bills. This data should include various file formats (e.g., PDF, PNG/JPEG, CSV) and corresponding labels indicating the type of expense bill (e.g., hotel bill, restaurant bill, grocery bill). Preprocess the data to extract relevant features and convert text-based files into a structured format suitable for ML.
        #?> Feature Extraction:
            #? For each file type, extract relevant features that can help distinguish different types of expense bills. For PDF or image files, use Optical Character Recognition (OCR) to extract text. For CSV files, identify key columns like 'category,' 'date,' 'total/amount,' 'currency,' and 'vendor.' The extracted features will serve as input data for your ML model.
        #?> Data Labeling and Splitting:
            #? Label the data appropriately with the correct expense bill types. Split the data into training and testing sets for model training and evaluation.
        #?> Model Selection:
            #? Choose a suitable ML model that can handle multi-class classification (since you have multiple types of expense bills). Common choices include Random Forest, Support Vector Machines (SVM), or Neural Networks.
        #?> Model Training:
            #? Train the ML model using the labeled training data and the extracted features.

#!> TextExtractor Class
    #?> Input file(s)
    #?> Detect file format with ReceiptProcessor class (Each format will be parsed differently)
    #! If PDF or PNG/JPEG:
        #?> Page Detection
            #? Page Segmentation: Receipts with multiple correlating pages before extracting info
            #? Page Detection: Receipts with multiple non-correlating pages
            #? ML to detect receipt type
        #?> OCR Text Extraction
            #? Detect language of text
            #? Region of Interest (ROI) Detection (Date, Vendor, Total Amount)
            #? Date Recognition and parsing to recognize date formats from different regions/locales
                #? Once language detected and data collected, find any word translated to 'total'-'cash'-'credit'
                #? Depending on currency, convert to USD
    #! If CSV:
        #?> Column detection
            #? Get CSV delimiter
            #? Find columns that matches 'category'-'date'-'total/amount'-'currency'-'vendor'
            #? Date Recognizition: If dates vary to ensure consistency

#!> DataStorage class
    #?> Responsible for interacting with the database (Postgres, SQLite) to store and retrieve expense data.
    #?> Provides methods to save the processed expense data to the database.
    #! Returns:
        #? table_id |     date   |   receipt_type  | total_amount | vendor
        #*-----------------------------------------------------------------*#
        #?     1    | 07/01/2023 |    hotel_bill   |   $9,127     | Hilton

#!> ExpenseKeeper class
    #?> Main class that ties everything together and acts as a facade for the expense keeping functionality.
    #?> Utilizes the other classes to process different types of receipt files and store the data in the database.
    #! ExpenseFormatter class (Nested class)
        #? Handles formatting the expense data in different output formats (CSV, JSON, etc.).
        #? Provides methods to export the data to various formats.

#!> ErrorHandler class
    #? Contains methods to handle errors and exceptions throughout the system gracefully.
    #? Provides user-friendly error messages for various scenarios.

#! Considerations:
    #? Data Storage (Postgres, SQLite)
    #? Output Format (CSV, JSON)
    #? Error Handling