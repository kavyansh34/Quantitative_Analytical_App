# ingestion_client.py
import asyncio
import json
import websockets
from datetime import datetime
from typing import List, Dict

# Configuration
BINANCE_WSS_URL = "wss://fstream.binance.com/ws/{sym}@trade"

class QuantIngestionClient:
    """Handles connection, streaming, and normalization of Binance tick data."""

    def __init__(self, symbols: List[str], data_queue: asyncio.Queue):
        """
        :param symbols: List of trading pairs to subscribe to (e.g., ["btcusdt", "ethusdt"]).
        :param data_queue: An asyncio.Queue to push normalized ticks to.
        """
        self.symbols = symbols
        self.data_queue = data_queue
        self.connections = []
        print(f"Subscribing to symbols: {self.symbols}")

    def _normalize_tick(self, message: Dict) -> Dict:
        """Converts raw Binance trade message into a standardized format and validates it."""
        if message.get('e') == 'trade':
            
            # --- VALIDATION BLOCK (The FIX) ---
            price = float(message.get('p', 0))
            size = float(message.get('q', 0))
            
            # Reject ticks where price is zero or negative, or size is zero.
            if price <= 0.0 or size <= 0.0:
                # You can log this for monitoring purposes
                # print(f"Rejected tick for {message['s']}: Price={price}, Size={size}")
                return None
            # --- END VALIDATION BLOCK ---

            ts_ms = message.get('T') or message.get('E')
            
            return {
                "symbol": message['s'].upper(),
                "ts": datetime.fromtimestamp(ts_ms / 1000).isoformat(),
                "price": price,
                "size": size,
            }
        return None

    async def _connect_and_stream(self, symbol: str):
        """Connects to a single symbol's WebSocket and streams data."""
        url = BINANCE_WSS_URL.format(sym=symbol.lower())
        
        # Continuously try to reconnect
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    print(f"WS connected: {symbol.upper()}")
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            tick = self._normalize_tick(data)
                            if tick:
                                # Push the normalized tick to the central queue
                                await self.data_queue.put(tick)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON for {symbol}")
                        except Exception as e:
                            print(f"Processing error for {symbol}: {e}")

            except websockets.exceptions.ConnectionClosedOK:
                print(f"WS closed OK: {symbol}")
            except Exception as e:
                print(f"Connection error for {symbol}: {e}. Retrying in 5s...")
                await asyncio.sleep(5)

    async def start(self):
        """Starts all concurrent connections."""
        tasks = [self._connect_and_stream(sym) for sym in self.symbols]
        self.connections = await asyncio.gather(*tasks)

    def stop(self):
        """Cancels all streaming tasks (less critical for this prototype)."""
        for task in self.connections:
            if task:
                task.cancel()
        print("Ingestion client stopped.")
