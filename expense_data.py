import requests
from pprint import pprint
from babel.numbers import get_currency_symbol
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

response = requests.get('https://openexchangerates.org/api/latest.json', params={'app_id':'fd41866037bd43f0a97c648d4b727dc1'}).json()
data = response['rates']
new_data = {key: f'{value} {get_currency_symbol(key, locale="en_US")}' for key, value in data.items()}
pprint(new_data)







def main():
    pass

if __name__ == '__main__':
    main()