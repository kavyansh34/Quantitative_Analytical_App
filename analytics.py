# analytics_engine.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from db_manager import DBManager
from typing import Dict, List, Tuple

class AnalyticsEngine:
    """Computes quantitative metrics (OLS, Z-score, Correlation, ADF test)."""

    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager

    def _prepare_data(self, bars: List[Dict]) -> pd.DataFrame:
        """Converts database bar data (list of dicts) into a time-indexed DataFrame."""
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        # Convert timestamp (seconds) to datetime and set as index
        df['dt'] = pd.to_datetime(df['ts'], unit='s')
        df = df.set_index('dt').sort_index()
        return df

    async def get_spread_data(self, symbol_x: str, symbol_y: str, timeframe: str) -> Tuple[pd.DataFrame, float]:
        """
        Computes the OLS hedge ratio and the resulting spread time series.
        :returns: (DataFrame with Spread, Hedge Ratio)
        """
        # Fetch the two time series
        bars_x = await self.db_manager.fetch_bars(symbol_x, timeframe, limit=500)
        bars_y = await self.db_manager.fetch_bars(symbol_y, timeframe, limit=500)

        df_x = self._prepare_data(bars_x)
        df_y = self._prepare_data(bars_y)

        if df_x.empty or df_y.empty:
            return pd.DataFrame(), np.nan

        # Merge them on the common timestamp index
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

        # The Hedge Ratio (beta) is the coefficient of X
        hedge_ratio = results.params['X']

        # Calculate the Spread: Spread = Y - (Hedge Ratio * X)
        df['Spread'] = df['Y'] - (hedge_ratio * df['X'])

        return df, hedge_ratio
    
    async def get_z_score(self, df_spread: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Calculates the rolling Z-score of the spread.
        :param window: The lookback window (number of bars) for mean and std dev calculation.
        """
        if df_spread.empty:
            return pd.DataFrame()
        
        # Calculate rolling mean and standard deviation
        df_spread['Rolling_Mean'] = df_spread['Spread'].rolling(window=window).mean()
        df_spread['Rolling_Std'] = df_spread['Spread'].rolling(window=window).std()
        
        # Calculate Z-score
        df_spread['Z_Score'] = (df_spread['Spread'] - df_spread['Rolling_Mean']) / df_spread['Rolling_Std']
        
        return df_spread.dropna()

    async def get_rolling_correlation(self, symbol_x: str, symbol_y: str, timeframe: str, window: int = 60) -> pd.DataFrame:
        """Calculates the rolling correlation between two assets' close prices."""
        
        # Data preparation (identical to get_spread_data setup)
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

        # Calculate rolling correlation
        df['Correlation'] = df['X'].rolling(window=window).corr(df['Y'])
        
        return df[['Correlation']].dropna()

    async def run_adf_test(self, df_spread: pd.DataFrame) -> Dict:
        """
        Performs the Augmented Dickey-Fuller (ADF) test on the spread time series.
        Tests for stationarity, a key requirement for mean-reversion strategies.
        """
        if df_spread.empty or len(df_spread) < 20: # ADF needs sufficient data
            return {"Test_Status": "Insufficient Data", "P_Value": np.nan, "T_Statistic": np.nan}

        # Use the spread column for the test
        series = df_spread['Spread'].dropna()
        if series.empty:
            return {"Test_Status": "Data Error", "P_Value": np.nan, "T_Statistic": np.nan}

        try:
            # Perform ADF Test
            adf_result = adfuller(series)
            
            p_value = adf_result[1]
            t_statistic = adf_result[0]

            # Determine stationarity based on the p-value
            if p_value <= 0.05:
                status = "Stationary (Mean-Reverting)"
            else:
                status = "Non-Stationary (Trending)"

            return {
                "Test_Status": status,
                "T_Statistic": t_statistic,
                "P_Value": p_value,
                "Critical_Values": adf_result[4] # 1%, 5%, 10% critical values
            }
        except Exception as e:
            return {"Test_Status": f"ADF Error: {e}", "P_Value": np.nan, "T_Statistic": np.nan}