import re
from collections import OrderedDict
from configparser import ConfigParser
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pprint import pprint
from typing import NamedTuple

import requests
from babel.numbers import get_currency_symbol
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import LanguageTranslatorV3


class Currency(NamedTuple):
    rate: float
    sym: str

    @lru_cache(maxsize=None)
    def get_rates():
        response = requests.get(config.apis['currency_url'], params={'app_id':config.apis['currency_api'],
                                                                    'base':'USD'}).json()
        data = response['rates']
        currency_data = OrderedDict({key: Currency(rate=value, sym=get_currency_symbol(key, locale="en_US")[-3:]) for key, value in data.items()})
        return currency_data
    
    def exchange_rate(total, from_curr, to_curr='USD'):
        currency_data = Currency.get_rates()
        rate = currency_data[to_curr].rate / currency_data[from_curr].rate
        return rate * total


class ConfigInfo(NamedTuple):
    host: str
    dbname: str
    user: str
    password: str
    apis: dict
    
    @lru_cache(maxsize=None)
    def get_config():
        global config
        
        config_parser = ConfigParser(allow_no_value=True)
        config_parser.read('config.ini')
        config = ConfigInfo(*[val for _, val in config_parser.items('Database')], apis=OrderedDict(config_parser.items('API')))
        return config


pprint(Currency.get_rates())
print(Currency.exchange_rate(300, 'YER'))

def main():
    config = ConfigParser()

if __name__ == '__main__':
    main()



#!> ReceiptProcessor class
    #?> Input various files for each type of expense bill for ML
    #?> Checks receipt type, if PDF checks expense category else return CSV/JSON
    #?> Returns the detected receipt type

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

#!> ExpenseKeeper class
    #?> Main class that ties everything together and acts as a facade for the expense keeping functionality.
    #?> Utilizes the other classes to process different types of receipt files and store the data in the database.

#!> ExpenseFormatter class
    #? Handles formatting the expense data in different output formats (CSV, JSON, etc.).
    #? Provides methods to export the data to various formats.

#!> ErrorHandler class
    #? Contains methods to handle errors and exceptions throughout the system gracefully.
    #? Provides user-friendly error messages for various scenarios.

#! Considerations:
    #? Data Storage (Postgres, SQLite)
    #? Output Format (CSV, JSON)
    #? Error Handling
#! Returns:
    #? table_id |     date   |   receipt_type  | total_amount | vendor
    #*-----------------------------------------------------------------*#
    #?     1    | 07/01/2023 |    hotel_bill   |   $9,127     | Hilton