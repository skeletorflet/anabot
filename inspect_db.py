
import sqlite3
import os

DB_NAME = "bot_data.db"

def inspect_db():
    if not os.path.exists(DB_NAME):
        print(f"Database {DB_NAME} not found.")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print("--- Table Schema ---")
    cursor.execute("PRAGMA table_info(generations)")
    columns = cursor.fetchall()
    for col in columns:
        print(col)
        
    print("\n--- Last 5 Generations ---")
    cursor.execute("SELECT id, prompt, seed FROM generations ORDER BY rowid DESC LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
        
    conn.close()

if __name__ == "__main__":
    inspect_db()
