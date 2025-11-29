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
from db_manager import DBManager
# Assuming 'analytics.py' imports the AnalyticsEngine class
from analytics import AnalyticsEngine 
from pykalman import KalmanFilter 

# --- Configuration ---
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"] 
TIMEFRAMES = ["1S", "1M", "5M"]
DEFAULT_WINDOW = 60 # Default lookback for rolling metrics

# --- Initialization and Setup ---
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DBManager()
    st.session_state.analytics_engine = AnalyticsEngine(st.session_state.db_manager)
    # Ensure the DB is initialized (run once)
    asyncio.run(st.session_state.db_manager.initialize())

@st.cache_data(show_spinner=False)
def get_analytics_data(symbol_x, symbol_y, timeframe, window, reg_type):
    """
    Function to fetch data and compute analytics, dynamically choosing OLS or Kalman Spread.
    The 'ts' dependency forces updates every few seconds.
    """
    refresh_key = int(time.time() / 5) 
    ae = st.session_state.analytics_engine
    
    # --- DYNAMIC REGRESSION LOGIC (KALMAN INTEGRATION) ---
    if reg_type == "Static OLS":
        # Assumes ae.get_spread_data now returns df_spread and static_hedge_ratio
        # You will need to ensure your ae.get_spread_data returns the necessary OLS info.
        df_spread, hedge_ratio = asyncio.run(ae.get_spread_data(symbol_x, symbol_y, timeframe))
        hedge_ratio_display = f"{hedge_ratio:.4f}"
    else: # Dynamic Kalman Filter
        # This assumes ae.get_kalman_spread_data is implemented in analytics_engine.py
        df_kalman, beta_series = asyncio.run(ae.get_kalman_spread_data(symbol_x, symbol_y, timeframe))
        df_spread = df_kalman.rename(columns={'Kalman_Spread': 'Spread'})
        # Use the latest beta for display
        hedge_ratio_display = f"{beta_series.iloc[-1]:.4f}" 
    # --- END DYNAMIC REGRESSION LOGIC ---

    # Z-Score and Correlation use the selected/computed df_spread
    df_z_score = asyncio.run(ae.get_z_score(df_spread.copy(), window))
    df_corr = asyncio.run(ae.get_rolling_correlation(symbol_x, symbol_y, timeframe, window))
    
    return df_spread, df_z_score, df_corr, hedge_ratio_display

# --- Plotting Functions (Unchanged) ---
def plot_ohlc_price(df: pd.DataFrame, symbol: str, timeframe: str):
    # ... (function body remains the same) ...
    if df.empty:
        return go.Figure()
    fig = go.Figure(data=[
        go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name=symbol)
    ])
    fig.update_layout(
        title=f'{symbol} Price ({timeframe})', xaxis_title="Time", yaxis_title="Price (USD)", xaxis_rangeslider_visible=False, height=400
    )
    return fig

def plot_spread_zscore(df_z_score: pd.DataFrame):
    # ... (function body remains the same) ...
    if df_z_score.empty:
        return go.Figure()

    fig = go.Figure()
    # 1. Spread Plot
    fig.add_trace(go.Scatter(x=df_z_score.index, y=df_z_score['Spread'], mode='lines', name='Spread'))
    
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

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Settings")

# --- Liquidity Filter Logic ---
st.sidebar.subheader("Liquidity Filter")
liquidity_stats = asyncio.run(st.session_state.analytics_engine.get_liquidity_stats("1M"))

min_volume = st.sidebar.number_input(
    "Min 1-Min Volume (Base Units)", 
    min_value=0, 
    value=1, 
    step=1, 
    help="Only show symbols where the last 1M bar volume exceeds this."
)

filtered_symbols = [
    sym for sym in SYMBOLS 
    if liquidity_stats.get(sym, 0) >= min_volume
]

default_x = st.session_state.get('sym_x', SYMBOLS[0])
default_y = st.session_state.get('sym_y', SYMBOLS[1])

symbol_x_list = filtered_symbols
symbol_y_list = [s for s in filtered_symbols if s != default_x]

idx_x = symbol_x_list.index(default_x) if default_x in symbol_x_list else 0
idx_y = symbol_y_list.index(default_y) if default_y in symbol_y_list else (0 if len(symbol_y_list)>0 else -1)

# --- CORRECTED Symbol Selection (Uses the filtered list) ---
st.sidebar.subheader("Symbol Selection (Pair)")
symbol_x = st.sidebar.selectbox("Symbol X (Base)", symbol_x_list, index=idx_x, key='sym_x')
symbol_y = st.sidebar.selectbox("Symbol Y (Hedged)", [s for s in filtered_symbols if s != symbol_x], index=idx_y, key='sym_y')

# --- NEW: Regression Type Control ---
regression_type = st.sidebar.selectbox("Regression Type", ["Static OLS", "Dynamic Kalman Filter"], index=0, key='reg_type')

# Timeframe Selection
timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=1, key='tf')

# Rolling Window
rolling_window = st.sidebar.slider("Rolling Window (Bars)", 10, 200, DEFAULT_WINDOW, key='window')

# --- Main Content: Analytics and Charts ---

st.header(f"Pairs Analytics: {symbol_x} / {symbol_y} ({timeframe})")
placeholder = st.empty() 

with placeholder.container():
    st.subheader("Live Analytics (Refreshes every 5s)")
    
    # --- Data Retrieval & Computation ---
    df_spread, df_z_score, df_corr, hedge_ratio_display = get_analytics_data(symbol_x, symbol_y, timeframe, rolling_window, regression_type)
    
    # --- Check for insufficient data ---
    if df_spread.empty or df_z_score.empty:
        st.warning(f"Waiting for sufficient {timeframe} bars to be saved for {symbol_x} and {symbol_y}. (Need >{rolling_window} bars)")
        st.stop()

    # --- 1. Key Statistics (Summary Stats) ---
    latest_z = df_z_score['Z_Score'].iloc[-1]
    latest_corr = df_corr['Correlation'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    # The display string now includes the regression type (Static or Dynamic)
    col1.metric(f"Hedge Ratio ({regression_type})", hedge_ratio_display) 
    col2.metric("Latest Z-Score", f"{latest_z:.2f}")
    col3.metric("Rolling Correlation", f"{latest_corr:.4f}")
    col4.metric("Spread Std Dev", f"{df_z_score['Rolling_Std'].iloc[-1]:.4f}")

    st.markdown("---")

    # --- 2. Alerting Logic ---
    st.subheader("🚨 Alerting System")
    alert_threshold = st.number_input("Z-Score Alert Threshold (e.g., 2.0)", min_value=1.0, max_value=5.0, value=2.0)
    
    if abs(latest_z) > alert_threshold:
        st.error(f"**ALERT!** Z-Score is **{latest_z:.2f}**, exceeding the threshold of {alert_threshold}. Potential mean-reversion trade entry!")
    else:
        st.success("Z-Score is within normal range.")

    st.markdown("---")

    # --- 3. Visualization ---
    
    st.subheader(f"Spread and Z-Score Plots ({regression_type})") # Dynamic title
    spread_fig, zscore_fig = plot_spread_zscore(df_z_score)
    st.plotly_chart(spread_fig, use_container_width=True)
    st.plotly_chart(zscore_fig, use_container_width=True)

    st.subheader(f"Rolling Correlation ({rolling_window}-Bar)")
    corr_fig = px.line(df_corr, y='Correlation', title=f'{symbol_x} / {symbol_y} Rolling Correlation')
    st.plotly_chart(corr_fig, use_container_width=True)
    
    # --- Price Charts (Optional, for context) ---
    st.subheader(f"Price Charts ({timeframe})")
    col_x, col_y = st.columns(2)
    
    df_x = asyncio.run(st.session_state.db_manager.fetch_bars(symbol_x, timeframe, limit=500))
    df_y = asyncio.run(st.session_state.db_manager.fetch_bars(symbol_y, timeframe, limit=500))
    df_x = st.session_state.analytics_engine._prepare_data(df_x)
    df_y = st.session_state.analytics_engine._prepare_data(df_y)

    with col_x:
        st.plotly_chart(plot_ohlc_price(df_x, symbol_x, timeframe), use_container_width=True)
    with col_y:
        st.plotly_chart(plot_ohlc_price(df_y, symbol_y, timeframe), use_container_width=True)

    st.markdown("---")

    # --- Cross-Correlation Heatmap (Discovery Tool) ---
    st.header("🔥 Cross-Correlation Heatmap (Discovery Tool)")
    heatmap_timeframe = st.selectbox("Timeframe for Correlation Matrix", ["1M", "5M"], index=0, key='heatmap_tf')
    heatmap_window = st.slider("Correlation Window (Bars)", 10, 200, 60, key='heatmap_window_slider')

    if st.button("Generate Correlation Matrix"):
        with st.spinner(f'Computing {heatmap_window}-bar correlation for all available symbols...'):
            corr_matrix_df = asyncio.run(
                st.session_state.analytics.get_cross_correlation_matrix(
                    symbol_x_list, heatmap_timeframe, heatmap_window
                )
            )
            
            if not corr_matrix_df.empty:
                fig = px.imshow(
                    corr_matrix_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title=f'Correlation Matrix ({heatmap_timeframe}, {heatmap_window} bars)'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("Look for values close to +1 (highly correlated) or -1 (inversely correlated) for potential pairs.")
            else:
                st.warning("Insufficient synchronized data to generate the heatmap.")


# --- 4. ADF Test Trigger ---
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


# --- 5. Data Export ---
st.markdown("---")
st.header("📥 Data Export")

csv_data = df_z_score.to_csv(index=True).encode('utf-8')
st.download_button(
    label="Download Z-Score & Spread Data as CSV",
    data=csv_data,
    file_name=f'{symbol_x}_{symbol_y}_zscore_spread_{timeframe}.csv',
    mime='text/csv',
)

# --- 6. OHLC Data Upload ---
st.markdown("---")
st.header("⬆️ Upload Historical OHLC Data")

uploaded_file = st.file_uploader("Upload OHLC CSV (must contain 'open', 'high', 'low', 'close', 'volume', and a time index)", type="csv")

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in uploaded_df.columns for col in required_cols):
             st.error(f"Error: Uploaded file must contain all required columns: {required_cols}")
        else:
             st.success("File uploaded and validated successfully! You can now process this data.")
             st.dataframe(uploaded_df.head())

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
