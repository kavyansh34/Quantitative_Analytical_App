import sqlite3

def inspect_database(db_path="quant_data.db"):
    """
    Connects to the SQLite database, prints table names, and displays
    the first 10 rows of the 'ohlc_bars' table.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            print("Successfully connected to the database.")
            cursor = conn.cursor()

            # Get a list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            if not tables:
                print("The database is empty (no tables found).")
                return

            print("\nAvailable tables:")
            for table in tables:
                print(f"- {table[0]}")

            # Inspect the 'ohlc_bars' table if it exists
            if ('ohlc_bars',) in tables:
                print("\nContents of 'ohlc_bars' table (first 10 rows):")
                cursor.execute("SELECT * FROM ohlc_bars LIMIT 10")
                rows = cursor.fetchall()
                
                if not rows:
                    print("The 'ohlc_bars' table is empty.")
                else:
                    # Get column names
                    col_names = [description[0] for description in cursor.description]
                    print(" | ".join(col_names))
                    print("-" * (len(" | ".join(col_names)))) # separator
                    for row in rows:
                        print(" | ".join(map(str, row)))

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_database()
