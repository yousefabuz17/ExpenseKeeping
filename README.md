# ExpenseKeeping

**``ExpenseKeeping``** is a Python project designed to help you manage your expenses by extracting and storing data from various types of files, including **``PDFs, CSVs, TXTs, and JSONs``**. It uses **``OCR (Optical Character Recognition)``** to extract relevant information from receipt files, and if needed, it can also translate text to facilitate accurate analysis. The extracted data is then saved into a database, allowing you to track your expenses over time. Additionally, the project provides a visual graph to visualize your spending patterns.

**Important Notes**:

- The receipts used in this project were found on [Kaggle](https://www.kaggle.com/):
    - [``My Expenses Data``](https://www.kaggle.com/datasets/tharunprabu/my-expenses-data)
    - [``My Receipts (PDF Scans)``](https://www.kaggle.com/datasets/jenswalter/receipts)
- The project uses the [Mindee API](https://mindee.com/) for OCR and the [Open Exchange Rates API](https://openexchangerates.org/) for currency conversion.
- [PostgreSQL](https://www.postgresql.org/) database is used.
- Due to the ```limitations``` of the free plan, the Mindee API can only process 250 receipts per month. Only **```~30 receipts```** were used in this project to avoid exceeding the limit.
However, the project can be easily extended to support more receipts by upgrading to a paid plan.

## Table of Contents
- [Features](#features)
- [Technical Details](#technical-details)
- [Configuration](#configuration)
- [Progress](#progress)
- [Output](#output)

## Features

- **``Support for Multiple File Formats:``** ExpenseKeeping accepts a variety of file formats, including PDF, CSV, TXT, and JSON, containing receipt information. This flexibility allows you to easily import receipts from various sources.

- **``OCR for Text Extraction:``** By utilizing advanced OCR technology, ExpenseKeeping can extract essential details such as the date, vendor, and total amount spent from receipt PDFs. This automation eliminates the need for manual data entry, saving you time and effort.

- **``Language Translation:``** In case your receipt text is in a different language than the default one, ExpenseKeeping supports language translation. The project can automatically translate text, enabling accurate analysis and reporting regardless of the receipt's original language.

- **``Efficient Data Storage:``** ExpenseKeeping securely stores the extracted data into a database, providing a centralized location for managing and organizing your expenses. This ensures easy expense tracking and facilitates data-driven decision-making.

- **``Visualization with Matplotlib and Seaborn:``** ExpenseKeeping utilizes the powerful visualization libraries Matplotlib and Seaborn to generate insightful graphs and charts. These visualizations help users gain a better understanding of their spending patterns and financial habits.

- **``Visual Graphs for Insights:``** To provide a comprehensive overview of your spending patterns, ExpenseKeeping generates visual graphs. These graphs offer valuable insights into your financial habits and help you identify areas where you can optimize your expenses.

- **``Currency Conversion:``** ExpenseKeeping supports currency conversion, allowing you to view your expenses in your desired currency. This feature ensures accurate financial analysis and reporting, regardless of the currency used in the receipt. Also includes retrieval of the latest exchange rates, currency symbols, and currency names.

- **``Expense Analytics for Insights:``** ExpenseKeeping not only helps you track and store expenses but also provides powerful analytics tools to gain insights into your spending patterns. You can generate various types of visualizations, such as bar charts, line plots, and pie charts, to better understand your expenses and identify trends.

## Technical Details
- **``Object-Oriented Programming:``** ExpenseKeeping is built using object-oriented programming principles, which allows for the creation of modular, reusable code. This approach also enables the project to be easily extended and scaled in the future.

- **``Asynchronous Operations:``** ExpenseKeeping leverages asynchronous programming to efficiently handle tasks that may cause delays, such as I/O operations (reading files, making API calls, etc.). Asynchronous programming ensures that the project can perform other tasks while waiting for time-consuming operations to complete, resulting in improved overall performance and responsiveness.

- **``Concurrent Threading:``** The project takes advantage of concurrent threading using the ThreadPoolExecutor and asyncio modules. This enables ExpenseKeeping to perform multiple tasks concurrently, further enhancing the speed of processing and data retrieval. Additionally, the use of asyncio allows for efficient handling of multiple asynchronous tasks.

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

## Progress
- [x] Configuration File
- [x] Text Translation & Currency Conversion
- [x] Text Extraction from various filetypes using OCR
- [x] Data Storage in a Database
- [x] Support for CSV, TXT, and JSON Files
- [ ] Expense Analytics for Insights
- [x] Visual Graphs for Expense Tracking
---

## Output

