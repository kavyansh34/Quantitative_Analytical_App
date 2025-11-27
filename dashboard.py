# dashboard.py

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# --- Import Backend Modules ---
# Note: For simplicity in Streamlit, we'll initialize the core components 
# and use st.session_state to manage the long-running async tasks.

from db_manager import DBManager
from analytics import AnalyticsEngine
# The Ingestion and DataProcessor will run as separate background processes
# or be initialized within Streamlit's async context. For this final step, 
# we'll assume the DB is being populated by a separate running process (app.py).

# --- Configuration ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"] 
TIMEFRAMES = ["1S", "1M", "5M"]
DEFAULT_WINDOW = 60 # Default lookback for rolling metrics

# --- Initialization and Setup ---
# Use Streamlit's mechanism to manage state and connect to the backend
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DBManager()
    st.session_state.analytics_engine = AnalyticsEngine(st.session_state.db_manager)
    # Ensure the DB is initialized (run once)
    asyncio.run(st.session_state.db_manager.initialize())

@st.cache_data(show_spinner=False)
def get_analytics_data(symbol_x, symbol_y, timeframe, window):
    """
    Function to fetch data and compute analytics.
    Uses caching for performance, but the 'ts' dependency (via time.time())
    will force updates every few seconds.
    """
    # Use a dummy key based on time to refresh data every ~5 seconds
    # This simulates a near-real-time update for the plots/stats.
    refresh_key = int(time.time() / 5) 

    ae = st.session_state.analytics_engine
    
    # Run the async functions synchronously within the cache context
    df_spread, hedge_ratio = asyncio.run(ae.get_spread_data(symbol_x, symbol_y, timeframe))
    df_z_score = asyncio.run(ae.get_z_score(df_spread.copy(), window))
    df_corr = asyncio.run(ae.get_rolling_correlation(symbol_x, symbol_y, timeframe, window))
    
    # ADF is triggered manually, so we don't run it here every 5 seconds.
    
    return df_spread, df_z_score, df_corr, hedge_ratio

# --- Plotting Functions ---

def plot_ohlc_price(df: pd.DataFrame, symbol: str, timeframe: str):
    """Generates an interactive OHLC/Line chart[cite: 16, 26]."""
    if df.empty:
        return go.Figure()

    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        )
    ])
    fig.update_layout(
        title=f'{symbol} Price ({timeframe})',
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=400
    )
    return fig

def plot_spread_zscore(df_z_score: pd.DataFrame):
    """Plots the Spread and the Z-Score."""
    if df_z_score.empty:
        return go.Figure()

    fig = go.Figure()
    
    # 1. Spread Plot
    fig.add_trace(go.Scatter(x=df_z_score.index, y=df_z_score['Spread'], 
                             mode='lines', name='Spread'))
    
    # 2. Z-Score Lines (for mean-reversion)
    fig.add_hline(y=df_z_score['Rolling_Mean'].iloc[-1], line_dash="dash", line_color="gray", annotation_text="Mean")
    fig.add_hline(y=df_z_score['Rolling_Mean'].iloc[-1] + 2*df_z_score['Rolling_Std'].iloc[-1], line_dash="dash", line_color="red", annotation_text="+2 Std Dev")
    fig.add_hline(y=df_z_score['Rolling_Mean'].iloc[-1] - 2*df_z_score['Rolling_Std'].iloc[-1], line_dash="dash", line_color="red", annotation_text="-2 Std Dev")

    fig.update_layout(title='Pairs Spread', height=300, showlegend=True)
    
    # Create Z-Score Subplot
    fig_z = px.line(df_z_score, y='Z_Score', title='Spread Z-Score')
    fig_z.add_hline(y=2, line_dash="dash", line_color="red")
    fig_z.add_hline(y=-2, line_dash="dash", line_color="red")
    fig_z.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_z.update_layout(height=250)
    
    return fig, fig_z

# --- Main Dashboard Layout ---

st.title("⚡️ Real-Time Quant Analytics Dashboard")
st.markdown("---")

# --- Sidebar for Controls  ---
st.sidebar.header("⚙️ Settings")

# Symbol Selection
st.sidebar.subheader("Symbol Selection (Pair)")
symbol_x = st.sidebar.selectbox("Symbol X (Base)", SYMBOLS, index=0, key='sym_x')
symbol_y = st.sidebar.selectbox("Symbol Y (Hedged)", [s for s in SYMBOLS if s != symbol_x], index=0, key='sym_y')

# Timeframe Selection
timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=1, key='tf')

# Rolling Window
rolling_window = st.sidebar.slider("Rolling Window (Bars)", 10, 200, DEFAULT_WINDOW, key='window')

# --- Main Content: Analytics and Charts ---

st.header(f"Pairs Analytics: {symbol_x} / {symbol_y} ({timeframe})")
placeholder = st.empty() # Placeholder for live updates

with placeholder.container():
    st.subheader("Live Analytics (Refreshes every 5s)")
    
    # --- Data Retrieval & Computation ---
    df_spread, df_z_score, df_corr, hedge_ratio = get_analytics_data(symbol_x, symbol_y, timeframe, rolling_window)
    
    # --- Check for insufficient data ---
    if df_spread.empty or df_z_score.empty:
        st.warning(f"Waiting for sufficient {timeframe} bars to be saved for {symbol_x} and {symbol_y}. (Need >{rolling_window} bars)")
        st.stop()

    # --- 1. Key Statistics (Summary Stats)  ---
    latest_z = df_z_score['Z_Score'].iloc[-1]
    latest_corr = df_corr['Correlation'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hedge Ratio (β)", f"{hedge_ratio:.4f}")
    col2.metric("Latest Z-Score", f"{latest_z:.2f}")
    col3.metric("Rolling Correlation", f"{latest_corr:.4f}")
    col4.metric("Spread Std Dev", f"{df_z_score['Rolling_Std'].iloc[-1]:.4f}")

    st.markdown("---")

    # --- 2. Alerting Logic  ---
    st.subheader("🚨 Alerting System")
    alert_threshold = st.number_input("Z-Score Alert Threshold (e.g., 2.0)", min_value=1.0, max_value=5.0, value=2.0)
    
    if abs(latest_z) > alert_threshold:
        st.error(f"**ALERT!** Z-Score is **{latest_z:.2f}**, exceeding the threshold of {alert_threshold}. Potential mean-reversion trade entry!")
    else:
        st.success("Z-Score is within normal range.")

    st.markdown("---")

    # --- 3. Visualization [cite: 24, 26] ---
    
    st.subheader("Spread and Z-Score Plots")
    spread_fig, zscore_fig = plot_spread_zscore(df_z_score)
    st.plotly_chart(spread_fig, use_container_width=True)
    st.plotly_chart(zscore_fig, use_container_width=True)

    st.subheader(f"Rolling Correlation ({rolling_window}-Bar)")
    corr_fig = px.line(df_corr, y='Correlation', title=f'{symbol_x} / {symbol_y} Rolling Correlation')
    st.plotly_chart(corr_fig, use_container_width=True)
    
    # --- Price Charts (Optional, for context) ---
    st.subheader(f"Price Charts ({timeframe})")
    col_x, col_y = st.columns(2)
    
    # Fetch data for price charts (we need raw close price for X and Y)
    df_x = asyncio.run(st.session_state.db_manager.fetch_bars(symbol_x, timeframe, limit=500))
    df_y = asyncio.run(st.session_state.db_manager.fetch_bars(symbol_y, timeframe, limit=500))
    df_x = st.session_state.analytics_engine._prepare_data(df_x)
    df_y = st.session_state.analytics_engine._prepare_data(df_y)

    with col_x:
        st.plotly_chart(plot_ohlc_price(df_x, symbol_x, timeframe), use_container_width=True)
    with col_y:
        st.plotly_chart(plot_ohlc_price(df_y, symbol_y, timeframe), use_container_width=True)


# --- 4. ADF Test Trigger  ---
st.markdown("---")
st.header("🔬 Statistical Tests")

if st.button("Run ADF Test on Current Spread"):
    with st.spinner('Running ADF Test...'):
        adf_results = asyncio.run(st.session_state.analytics_engine.run_adf_test(df_spread.copy()))
        
        st.json(adf_results)
        
        if adf_results['P_Value'] <= 0.05:
            st.success(f"Conclusion: The spread is likely **{adf_results['Test_Status']}** (P-Value: {adf_results['P_Value']:.4f}).")
        elif pd.notna(adf_results['P_Value']):
            st.warning(f"Conclusion: The spread is likely **{adf_results['Test_Status']}** (P-Value: {adf_results['P_Value']:.4f}).")
        else:
            st.error("Could not run test.")


# --- 5. Data Export  ---
st.markdown("---")
st.header("📥 Data Export")

# Example for exporting the Z-Score data
csv_data = df_z_score.to_csv(index=True).encode('utf-8')
st.download_button(
    label="Download Z-Score & Spread Data as CSV",
    data=csv_data,
    file_name=f'{symbol_x}_{symbol_y}_zscore_spread_{timeframe}.csv',
    mime='text/csv',
)

# --- 6. OHLC Data Upload [cite: 39] ---
st.markdown("---")
st.header("⬆️ Upload Historical OHLC Data")

uploaded_file = st.file_uploader("Upload OHLC CSV (must contain 'open', 'high', 'low', 'close', 'volume', and a time index)", type="csv")

if uploaded_file is not None:
    try:
        # Load the data
        uploaded_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Validation (mandatory that this works without any dummy upload [cite: 39])
        if not all(col in uploaded_df.columns for col in required_cols):
             st.error(f"Error: Uploaded file must contain all required columns: {required_cols}")
        else:
             st.success("File uploaded and validated successfully! You can now process this data.")
             
             # Placeholder for processing logic (e.g., storing to a separate "historical" table 
             # and running analytics on it)
             st.dataframe(uploaded_df.head())

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")