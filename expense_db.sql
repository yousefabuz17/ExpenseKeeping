CREATE TABLE IF NOT EXISTS Expense (
    expense_id SERIAL PRIMARY KEY,
    date VARCHAR(255),
    time VARCHAR(255),
    language VARCHAR(255),
    currency VARCHAR(255),
    symbol VARCHAR(255),
    category VARCHAR(255),
    subcategory VARCHAR(255),
    amount FLOAT,
    vendor VARCHAR(255),
    note VARCHAR(255)
);

WITH new_values (date, time, language, currency, symbol, category, subcategory, amount, vendor, note) AS (
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
),
upsert AS (
    UPDATE Expense AS a
    SET date = nv.date,
        time = nv.time,
        language = nv.language,
        currency = nv.currency,
        category = nv.category,
        subcategory = nv.subcategory,
        amount = nv.amount,
        vendor = nv.vendor,
        note = nv.note
    FROM new_values AS nv
    WHERE a.date = nv.date
    RETURNING a.*
)
INSERT INTO Expense (date, time, language, currency, symbol, category, subcategory, amount, vendor, note)
SELECT date, time, language, currency, symbol, category, subcategory, amount, vendor, note
FROM new_values
WHERE NOT EXISTS (
    SELECT 1
    FROM upsert AS u
    WHERE new_values.date = u.date
);