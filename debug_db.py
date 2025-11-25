import sqlite3
from pathlib import Path

db_path = Path("data/database.db")
print(f"Database exists: {db_path.exists()}")

if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if users table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
    result = cursor.fetchone()
    print(f"Users table exists: {result is not None}")
    
    # Count users
    if result:
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        print(f"Number of users: {count}")
    
    conn.close()
