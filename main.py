# app.py (Further Updated)

import asyncio
from datetime import datetime
from Ingestion_client import QuantIngestionClient
from db_manager import DBManager
from data_processor import DataProcessor
import numpy as np
from analytics import AnalyticsEngine # Import the new module

# We will analyze this pair for cointegration
PAIR_X, PAIR_Y = "BTCUSDT", "ETHUSDT" 
TIMEFRAME = "1M" # Use 1-minute bars for analysis
WINDOW = 60 # Rolling window for Z-score and Correlation

# ... (Previous imports and SYMBOLS definition remain the same) ...

async def analytics_runner(analytics_engine: AnalyticsEngine):
    """Periodically triggers the main analytic computations."""
    print("Starting Analytics Runner...")
    while True:
        await asyncio.sleep(60) # Run every minute, synced with 1M bar creation
        print(f"\n--- Running Analytics at {datetime.now().strftime('%H:%M:%S')} ---")
        
        try:
            # 1. Compute Spread and Hedge Ratio
            df_spread, hedge_ratio = await analytics_engine.get_spread_data(PAIR_X, PAIR_Y, TIMEFRAME)
            
            if df_spread.empty:
                print("Analytics: Insufficient data for Spread calculation.")
                continue

            print(f"Hedge Ratio ({PAIR_Y} vs {PAIR_X}): {hedge_ratio:.4f}")
            
            # 2. Compute Z-Score
            df_z_score = await analytics_engine.get_z_score(df_spread.copy(), WINDOW) # Use a copy
            latest_z = df_z_score['Z_Score'].iloc[-1] if not df_z_score.empty else np.nan
            print(f"Latest Spread: {df_spread['Spread'].iloc[-1]:.4f} | Latest Z-Score: {latest_z:.2f}")

            # 3. Run ADF Test
            adf_results = await analytics_engine.run_adf_test(df_spread.copy())
            print(f"ADF Test: {adf_results['Test_Status']} (P-Value: {adf_results['P_Value']:.4f})")
            
            # 4. Compute Rolling Correlation
            df_corr = await analytics_engine.get_rolling_correlation(PAIR_X, PAIR_Y, TIMEFRAME, WINDOW)
            latest_corr = df_corr['Correlation'].iloc[-1] if not df_corr.empty else np.nan
            print(f"Latest {WINDOW}-Bar Rolling Correlation: {latest_corr:.4f}")
            
        except Exception as e:
            print(f"Analytics Runner Error: {e}")


async def main():
    # 1. Initialize Database Manager
    db_manager = DBManager()
    await db_manager.initialize()

    # 2. Initialize the shared queue and Ingestion Client
    tick_queue = asyncio.Queue()
    ingestion_client = QuantIngestionClient([PAIR_X, PAIR_Y], tick_queue) # Focus on the pair
    ingestion_task = asyncio.create_task(ingestion_client.start())

    # 3. Start the Data Processor
    data_processor = DataProcessor(tick_queue, db_manager)
    processor_task = asyncio.create_task(data_processor.run())
    
    # 4. Initialize and Start the Analytics Engine
    analytics_engine = AnalyticsEngine(db_manager)
    analytics_task = asyncio.create_task(analytics_runner(analytics_engine)) # New task

    print("--- System running. Press Ctrl+C to stop. ---")
    
    # ... (Monitor task can be removed or simplified as analytics_runner provides output) ...
    # Simple monitor to keep things tidy
    async def monitor():
        while True:
            await asyncio.sleep(5)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ingesting and Processing...", end='\r')

    monitor_task = asyncio.create_task(monitor())

    # Keep the main loop running
    try:
        await asyncio.gather(ingestion_task, processor_task, analytics_task, monitor_task)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        ingestion_client.stop()
        processor_task.cancel()
        analytics_task.cancel()
        monitor_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
             raise