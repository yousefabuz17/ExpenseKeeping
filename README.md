# ExpenseKeeping

``ExpenseKeeping`` is a Python project designed to help you manage your expenses by extracting and storing data from various types of files, including ``PDFs, CSVs, TXTs, and JSONs``. It uses ``OCR (Optical Character Recognition)`` to extract relevant information from receipt files, and if needed, it can also translate text to facilitate accurate analysis. The extracted data is then saved into a database, allowing you to track your expenses over time. Additionally, the project provides a visual graph to visualize your spending patterns.

## Table of Contents
- [Features](#features)
- [Configuration](#configuration)
- [Text Translation and Currency Conversion](#text-translation-and-currency-conversion)
- [Progress](#progress)

## Features

- **``Multiple File Formats:``** ExpenseKeeping accepts a variety of file formats, including PDF, CSV, TXT, and JSON, containing receipt information. This flexibility allows you to easily import receipts from various sources.

- **``OCR for Text Extraction:``** By utilizing advanced OCR technology, ExpenseKeeping can extract essential details such as the date, vendor, and total amount spent from receipt PDFs. This automation eliminates the need for manual data entry, saving you time and effort.

- **``Language Translation:``** In case your receipt text is in a different language than the default one, ExpenseKeeping supports language translation. The project can automatically translate text, enabling accurate analysis and reporting regardless of the receipt's original language.

- **``Efficient Data Storage:``** ExpenseKeeping securely stores the extracted data into a database, providing a centralized location for managing and organizing your expenses. This ensures easy expense tracking and facilitates data-driven decision-making.

- **``Visual Graphs for Insights:``** To provide a comprehensive overview of your spending patterns, ExpenseKeeping generates visual graphs. These graphs offer valuable insights into your financial habits and help you identify areas where you can optimize your expenses.

## Configuration

- ExpenseKeeping can be easily configured according to your preferences using the `config.ini` file. You can adjust the default language for translation, fine-tune database settings, and customize other options to best suit your individual needs.
    - **Place all your receipts in the `dir` directory specified in the configuration file.**
```ini
[Database]
host = <host>
dbname = <database name>
user = <user>
password = <password>

[API]
currency_url = https://openexchangerates.org/api/latest.json
currency_api = <openexchange api>
mindee_api = <mindee api>

[Directory]
dir = ./receipts
```

## Text Translation and Currency Conversion

- ExpenseKeeping implements text translation through integration with a reliable and user-friendly language translation API. This allows the project to automatically handle receipts in various languages, making it accessible and efficient for multilingual users.

- For currency conversion, the project relies on a trusted currency converter API that provides up-to-date exchange rates. ExpenseKeeping leverages this API to convert the total amount spent in each receipt into your desired currency, ensuring accurate financial analysis and reporting.

## Progress
- [x] Configuration File
- [x] Text Translation & Currency Conversion
- [x] Text Extraction from various filetypes using OCR
- [ ] Data Storage in a Database
- [x] Support for CSV, TXT, and JSON Files
- [ ] Expense Analytics for Insights
- [ ] Visual Graphs for Expense Tracking
---
