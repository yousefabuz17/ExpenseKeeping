import asyncio
import functools
import os
import re
import subprocess
from collections import OrderedDict
from configparser import ConfigParser
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from pprint import pprint
from typing import NamedTuple

import aiohttp
import asyncpg
import boto3
import pandas as pd
import PyPDF2
import pytesseract
import requests_cache
from babel.numbers import get_currency_symbol
from PIL import Image

requests_cache.install_cache(expire_after=7200)

class ConfigInfo(NamedTuple):
    host: str = None
    dbname: str = None
    user: str = None
    password: str = None
    apis: OrderedDict = None
    
    @dataclass
    class DirInfo:
        dirs: dict = None
    
    @staticmethod
    @lru_cache(maxsize=None)
    def get_config():
        config_parser = ConfigParser(allow_no_value=True)
        config_parser.read('config.ini')
        config = ConfigInfo(*[val for _, val in config_parser.items('Database')],
                            apis=OrderedDict(config_parser.items('API')))
        return config
    
    @staticmethod
    @lru_cache(maxsize=None)
    def get_dir():
        config_parser = ConfigParser(allow_no_value=True)
        config_parser.read('config.ini')
        directories = ConfigInfo.DirInfo(dirs=dict(config_parser.items('Directory')))
        return directories

@dataclass
class Currency:
    rate: float
    sym: str
    
    async def get_rates():
        config = ConfigInfo.get_config()
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(config.apis['currency_url'], params={'app_id':config.apis['currency_api'],
                                                                        'base':'USD'}) as response:
                response = await response.json()
                data = response['rates']
                currency_data = OrderedDict({key: Currency(rate=value, sym=get_currency_symbol(key, locale="en_US")[-3:]) for key, value in data.items()})
                return currency_data
    
    @staticmethod
    async def exchange_rate(total, from_curr, to_curr='USD'):
        currency_data = await Currency.get_rates()
        rate = Decimal(currency_data[to_curr].rate) / Decimal(currency_data[from_curr].rate)
        return f'{currency_data[to_curr].sym} {rate * total:.2f}'

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


class ReceiptProcessor:
    def __init__(self):
        pass

@dataclass
class FileInfo:
    lang: str = None
    type_: str = None
    path: str = None
    category: str = None
    contents: str = None
    
    def _file_modifier(print_results=True):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args):
                new_files = []
                files = func(*args)
                for _, i in enumerate(files):
                    path_ = Path(i)
                    i = i.as_posix()
                    lang_ = (lang := re.findall(r'/(\w{2})/', i)) and lang[0]
                    if lang_:
                        type_ = (type_ := re.findall(r'\.(pdf|json|csv|png|jpeg|txt)$', i.lower())) and type_[0]
                        categ_ = (categ := re.findall(r'(\w+\s?\w+?)/', i[i.index('receipts'):])) and \
                                    (categ[-1] if len(categ) == 4 else None)
                        
                        new_files.append(FileInfo(lang=lang_,
                                                    type_=type_,
                                                    path=path_,
                                                    category=categ_))
                #!> Use ML to determine what the None valued category should become
                return new_files
            return wrapper
        return decorator
    
    def _file_reader(print_resuts=True):
        def decorator(func):
            def wrapper(*args):
                all_files = func(*args)
                for file in all_files:
                    if file.type_ in ('pdf', 'jpeg', 'png'):
                        if file.type_.lower() == 'pdf':
                            pdf_file = open(file.path, 'rb')
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            if len(pdf_reader.pages) > 1:
                                for idx in range(len(pdf_reader.pages)):
                                    page_contents = pdf_reader.pages[idx].extract_text()
                                    file.contents = page_contents
                            else:
                                page_contents = pdf_reader.pages[0].extract_text()
                                file.contents = page_contents
                return all_files
            return wrapper
        return decorator
    
    def _translate_contents(print_results=True):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args):
                translate_client = boto3.client('translate', region_name='us-east-1')
                files = func(*args)
                try:
                    for file in files:
                        response = translate_client.translate_text(
                            Text=file.contents,
                            SourceLanguageCode='auto',
                            TargetLanguageCode='en'
                        )
                        translated_text = response['TranslatedText']
                        file.contents = translated_text
                    return files

                except translate_client.exceptions.UnsupportedLanguagePairException as e:
                    # print(f"Unsupported language pair: {e}")
                    pass

                except Exception as e:
                    # print(f"Error occurred during translation: {e}")
                    pass
                
                return files
            return wrapper
        return decorator

class TextExtractor:
    def __init__(self):
        self.dirs = Path(__file__).parent.absolute() / ConfigInfo.get_dir().dirs['dir']
        self.files = None
    
    @FileInfo._translate_contents()
    @FileInfo._file_reader()
    @FileInfo._file_modifier()
    def get_files(self):
        self.files = [Path(root) / i for root, _, files in os.walk(self.dirs) for i in files if files]
        return self.files


class Wrapper:
    pass

async def main():
    a = TextExtractor().get_files()
    pprint(a)

if __name__ == '__main__':
    asyncio.run(main())

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