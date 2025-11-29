# analytics_engine.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
# Ensure pykalman is installed: pip install pykalman
from pykalman import KalmanFilter 
from db_manager import DBManager # Assumed to exist
from typing import Dict, List, Tuple

class AnalyticsEngine:
    """
    Computes quantitative metrics including OLS, Kalman Filter, Z-score, 
    Correlation, and ADF test, accessing data asynchronously via DBManager.
    """

    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager

    def _prepare_data(self, bars: List[Dict]) -> pd.DataFrame:
        """Converts database bar data (list of dicts) into a time-indexed DataFrame."""
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        df['dt'] = pd.to_datetime(df['ts'], unit='s')
        df = df.set_index('dt').sort_index()
        return df

    # --- OLS REGRESSION (STATIC HEDGE) ---
    async def get_spread_data(self, symbol_x: str, symbol_y: str, timeframe: str) -> Tuple[pd.DataFrame, float]:
        """
        Computes the OLS hedge ratio and the resulting spread time series.
        :returns: (DataFrame with Spread, Hedge Ratio)
        """
        bars_x = await self.db_manager.fetch_bars(symbol_x, timeframe, limit=500)
        bars_y = await self.db_manager.fetch_bars(symbol_y, timeframe, limit=500)

        df_x = self._prepare_data(bars_x)
        df_y = self._prepare_data(bars_y)

        if df_x.empty or df_y.empty:
            return pd.DataFrame(), np.nan

        df = pd.merge(
            df_x[['close']].rename(columns={'close': 'X'}),
            df_y[['close']].rename(columns={'close': 'Y'}),
            left_index=True, right_index=True, how='inner'
        )

        if df.empty:
            return pd.DataFrame(), np.nan

        # OLS Regression: Y = beta * X + alpha
        Y = df['Y']
        X = sm.add_constant(df['X'])
        model = sm.OLS(Y, X)
        results = model.fit()

        hedge_ratio = results.params['X']
        df['Spread'] = df['Y'] - (hedge_ratio * df['X'])

        return df, hedge_ratio
    
    # --- KALMAN FILTER (DYNAMIC HEDGE) ---
    async def get_kalman_spread_data(self, symbol_x: str, symbol_y: str, timeframe: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Computes a dynamically hedged spread and the time-varying hedge ratio using a Kalman Filter.
        """
        bars_x = await self.db_manager.fetch_bars(symbol_x, timeframe, limit=500)
        bars_y = await self.db_manager.fetch_bars(symbol_y, timeframe, limit=500)

        df_x = self._prepare_data(bars_x)
        df_y = self._prepare_data(bars_y)

        if df_x.empty or df_y.empty:
            return pd.DataFrame(), pd.Series()

        df = pd.merge(
            df_x[['close']].rename(columns={'close': 'X'}),
            df_y[['close']].rename(columns={'close': 'Y'}),
            left_index=True, right_index=True, how='inner'
        )
        if len(df) < 10:
            return pd.DataFrame(), pd.Series()
        
        # --- SHAPE FIX: ALL PARAMETERS MUST BE EXPLICIT NUMPY ARRAYS ---
        
        # Observation matrices must be (n_samples, n_dim_obs, n_dim_state). Here, (n, 1, 1)
        observation_matrices = df['X'].values.reshape(-1, 1, 1) 
        
        kf = KalmanFilter(
            # State mean and covariance must be 1D and 2D arrays, respectively
            initial_state_mean=np.array([0]),
            initial_state_covariance=np.array([[1]]),
            
            # Transition matrices must be 2D array (1, 1)
            transition_matrices=np.array([[1]]),
            
            # Use the corrected 3D array for observations
            observation_matrices=observation_matrices,
            
            # Covariances must be 2D arrays (1, 1)
            observation_covariance=np.array([[0.1]]), 
            transition_covariance=np.array([[0.001]]) 
        )

        # Run the filter
        state_means, _ = kf.filter(df['Y'].values)
        
        # The state means are the time-varying hedge ratio (beta_t)
        df['Kalman_Beta'] = state_means.flatten()
        
        # Calculate the Kalman Spread
        df['Kalman_Spread'] = df['Y'] - (df['Kalman_Beta'] * df['X'])

        return df[['Kalman_Spread', 'Kalman_Beta']], df['Kalman_Beta']
    
    # --- Z-SCORE CALCULATION ---
    async def get_z_score(self, df_spread: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Calculates the rolling Z-score of the spread."""
        if df_spread.empty:
            return pd.DataFrame()
        
        # Calculate rolling mean and standard deviation
        df_spread['Rolling_Mean'] = df_spread['Spread'].rolling(window=window).mean()
        df_spread['Rolling_Std'] = df_spread['Spread'].rolling(window=window).std()
        
        # Calculate Z-score
        df_spread['Z_Score'] = (df_spread['Spread'] - df_spread['Rolling_Mean']) / df_spread['Rolling_Std']
        
        return df_spread.dropna()

    # --- ROLLING CORRELATION ---
    async def get_rolling_correlation(self, symbol_x: str, symbol_y: str, timeframe: str, window: int = 60) -> pd.DataFrame:
        """Calculates the rolling correlation between two assets' close prices."""
        
        bars_x = await self.db_manager.fetch_bars(symbol_x, timeframe, limit=500)
        bars_y = await self.db_manager.fetch_bars(symbol_y, timeframe, limit=500)

        df_x = self._prepare_data(bars_x)
        df_y = self._prepare_data(bars_y)

        if df_x.empty or df_y.empty:
            return pd.DataFrame()

        df = pd.merge(
            df_x[['close']].rename(columns={'close': 'X'}),
            df_y[['close']].rename(columns={'close': 'Y'}),
            left_index=True, right_index=True, how='inner'
        )
        
        if df.empty:
            return pd.DataFrame()

        df['Correlation'] = df['X'].rolling(window=window).corr(df['Y'])
        return df[['Correlation']].dropna()

    # --- ADF TEST ---
    async def run_adf_test(self, df_spread: pd.DataFrame) -> Dict:
        """Performs the Augmented Dickey-Fuller (ADF) test on the spread time series."""
        if df_spread.empty or len(df_spread) < 20:
            return {"Test_Status": "Insufficient Data", "P_Value": np.nan, "T_Statistic": np.nan}

        series = df_spread['Spread'].dropna()
        if series.empty:
            return {"Test_Status": "Data Error", "P_Value": np.nan, "T_Statistic": np.nan}

        try:
            adf_result = adfuller(series)
            p_value = adf_result[1]
            status = "Stationary (Mean-Reverting)" if p_value <= 0.05 else "Non-Stationary (Trending)"

            return {
                "Test_Status": status,
                "T_Statistic": adf_result[0],
                "P_Value": p_value,
                "Critical_Values": adf_result[4]
            }
        except Exception as e:
            return {"Test_Status": f"ADF Error: {e}", "P_Value": np.nan, "T_Statistic": np.nan}
    
    # --- LIQUIDITY STATS (ADVANCED EXTENSION) ---
    async def get_liquidity_stats(self, timeframe: str) -> Dict[str, float]:
        """Calculates the latest volume (liquidity proxy) for all symbols."""
        latest_bars = await self.db_manager.fetch_latest_ohlcv_for_all(timeframe, limit=1)
        
        liquidity_stats = {}
        for symbol, data in latest_bars.items():
            liquidity_stats[symbol] = data.get('volume', 0.0)
        return liquidity_stats

    # --- CROSS-CORRELATION MATRIX (ADVANCED EXTENSION) ---
    async def get_cross_correlation_matrix(self, symbols: List[str], timeframe: str, window: int = 60) -> pd.DataFrame:
        """Computes the correlation matrix for a list of symbols."""
        all_data = {}
        
        for symbol in symbols:
            bars = await self.db_manager.fetch_bars(symbol, timeframe, limit=window * 2)
            df = self._prepare_data(bars)
            if not df.empty:
                all_data[symbol] = df['close']
        
        if not all_data:
            return pd.DataFrame()

        combined_df = pd.DataFrame(all_data).dropna()
        
        if len(combined_df) < window:
             return pd.DataFrame()
             
        corr_matrix = combined_df.tail(window).corr()
        return corr_matrix
