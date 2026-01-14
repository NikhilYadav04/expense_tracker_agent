import sqlite3

conn = sqlite3.connect("N:/Dev/Langgraph-Project/Expense-Whatsapp/data/test.db")
conn.close()

print("SQLite OK")
