# create_failed_logins_table.py
import sqlite3
from pathlib import Path

db_path = Path("hybrid_summarizer/data/database.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS failed_logins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address TEXT,
    reason TEXT
);
''')

conn.commit()
conn.close()
print("Table 'failed_logins' created (if not already present).")
