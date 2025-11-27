# db_manager.py

import aiosqlite
from typing import Dict, List

class DBManager:
    """Handles asynchronous SQLite connections and operations."""

    def __init__(self, db_path: str = "quant_data.db"):
        self.db_path = db_path

    async def initialize(self):
        """Creates the necessary tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            # Table to store resampled OHLC bars for all timeframes
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ohlc_bars (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timeframe, ts)
                )
            """)
            await db.commit()
        print(f"Database initialized at {self.db_path}")

    async def insert_bar(self, bar: Dict):
        """Inserts a single resampled OHLC bar into the database."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO ohlc_bars 
                (symbol, timeframe, ts, open, high, low, close, volume) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bar['symbol'],
                bar['timeframe'],
                bar['ts'],
                bar['open'],
                bar['high'],
                bar['low'],
                bar['close'],
                bar['volume']
            ))
            await db.commit()
            
    # Placeholder for fetching data, will be used by the Analytics Engine later
    async def fetch_bars(self, symbol: str, timeframe: str, limit: int = 100):
        """Fetches the latest OHLC bars."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row # Allows accessing columns by name
            cursor = await db.execute("""
                SELECT * FROM ohlc_bars 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY ts DESC
                LIMIT ?
            """, (symbol, timeframe, limit))
            return [dict(row) for row in await cursor.fetchall()]