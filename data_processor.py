# data_processor.py

import asyncio
from datetime import datetime
from typing import Dict
from db_manager import DBManager
import pandas as pd # Used for the resampling logic helper

# Define the timeframes we need to support (in seconds)
TIMEFRAMES = {
    "1S": 1, 
    "1M": 60, 
    "5M": 300
}

class DataProcessor:
    """
    Consumes raw ticks, generates OHLC bars for multiple timeframes, 
    and persists them to the database.
    """

    def __init__(self, data_queue: asyncio.Queue, db_manager: DBManager):
        self.data_queue = data_queue
        self.db_manager = db_manager
        # Stores the currently building OHLC bars: 
        # {symbol: {timeframe: {'ts': timestamp_sec, 'open': ..., 'volume': ...}}}
        self.open_bars = {} 

    def _get_bar_timestamp(self, ts: datetime, interval_sec: int) -> int:
        """Calculates the opening timestamp (in seconds) for a bar interval."""
        total_sec = int(ts.timestamp())
        # Snap the timestamp to the start of the bar interval
        return total_sec - (total_sec % interval_sec)

    def _process_tick(self, tick: Dict):
        """Processes a single tick against all required timeframes."""
        symbol = tick['symbol']
        price = tick['price']
        size = tick['size']
        ts = datetime.fromisoformat(tick['ts'])

        if symbol not in self.open_bars:
            self.open_bars[symbol] = {}

        for tf_str, tf_sec in TIMEFRAMES.items():
            # Calculate the expected opening timestamp for the current bar
            current_bar_ts = self._get_bar_timestamp(ts, tf_sec)
            
            # --- 1. Check if we need to close the previous bar ---
            if tf_str in self.open_bars[symbol]:
                prev_bar = self.open_bars[symbol][tf_str]
                
                # If the tick's bar is newer than the one we are building, close the old one
                if current_bar_ts > prev_bar['ts']:
                    # The previous bar is complete, save it to the DB
                    asyncio.create_task(self.db_manager.insert_bar({
                        'symbol': symbol,
                        'timeframe': tf_str,
                        'ts': prev_bar['ts'], # Bar's starting timestamp
                        'open': prev_bar['open'],
                        'high': prev_bar['high'],
                        'low': prev_bar['low'],
                        'close': prev_bar['close'],
                        'volume': prev_bar['volume'],
                    }))
                    
                    # Start a new bar
                    self.open_bars[symbol][tf_str] = {
                        'ts': current_bar_ts,
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': size
                    }
                else:
                    # Continue building the current bar
                    prev_bar['high'] = max(prev_bar['high'], price)
                    prev_bar['low'] = min(prev_bar['low'], price)
                    prev_bar['close'] = price
                    prev_bar['volume'] += size
            
            # --- 2. Initialize the first bar ---
            else:
                self.open_bars[symbol][tf_str] = {
                    'ts': current_bar_ts,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': size
                }


    async def run(self):
        """Main processing loop: pulls ticks from the queue and processes them."""
        print("Starting Data Processor (Resampling)...")
        while True:
            # Wait for the next tick from the ingestion client
            tick = await self.data_queue.get() 
            
            # This logic is non-blocking and processes the tick immediately
            self._process_tick(tick)
            
            self.data_queue.task_done()
            
            # In a real-time system, a brief pause here can prevent the processor 
            # from consuming too much CPU during high tick rates, though not strictly required now.
            # await asyncio.sleep(0)