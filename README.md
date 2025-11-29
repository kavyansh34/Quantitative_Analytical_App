# ⚡️ Real-Time Quant Analytics Stack

This project delivers a small but complete end-to-end analytical application designed to provide **real-time pair trading insights** using Binance Futures tick data. The system covers data ingestion, multi-timeframe processing, advanced statistical analysis, and interactive visualization via a Streamlit dashboard.

# ℹ️ About This Project: Real-Time Quant Analytics Stack

This application is a complete, end-to-end **Real-Time Quantitative Analytics Stack** designed to provide actionable insights for a **mean-reversion pair trading strategy**. The system continuously monitors, processes, and analyzes live financial data from Binance Futures, demonstrating full-stack development capability from ingestion through advanced statistical modeling to interactive visualization.

***

## Key Highlights and Technical Achievements

### 📊 Quantitative Modeling Excellence

* **Dynamic Hedge Estimation (Kalman Filter):** Implements the advanced **Kalman Filter** to calculate a **time-varying, adaptive hedge ratio ($\beta_t$)**, which is essential for deriving a robust spread signal in volatile, non-stationary markets.
* **Static Baseline:** Includes **Static OLS Regression** for traditional hedge ratio calculation, providing a baseline for comparison.
* **Backtesting Utility:** Features a **Mini Mean-Reversion Backtest** module to simulate the performance of the $Z$-score strategy ($\text{Entry}: |Z| > 2, \text{Exit}: |Z| \approx 0$) on historical data, providing quantifiable feedback on the strategy's efficacy.
* **Statistical Validity:** Features the **Augmented Dickey-Fuller (ADF) Test** to allow the user to statistically confirm if the generated spread is stationary (mean-reverting).

### 🛠️ Architecture and Data Pipeline

* **Modular and Decoupled Design:** The architecture is separated into distinct, loosely coupled services (Ingestion, Processor, Analytics Engine) using Python classes, ensuring clarity, testability, and easy maintainability.
* **Asynchronous Processing:** Leverages Python's **`asyncio`** and **`websockets`** for high-performance, non-blocking I/O, allowing the system to handle concurrent, real-time tick data streams efficiently.
* **Data Integrity:** Implements a strict filter at the ingestion layer to reject corrupt data (e.g., price $\le 0$ or size $\le 0$), ensuring data quality before processing.
* **Flexible Sampling:** Aggregates raw tick data into user-selectable OHLCV timeframes (1S, 1M, 5M).

### 💻 User Interface and Utility

* **Interactive Dashboard:** Built with **Streamlit** and **Plotly** to provide visualizations that support zoom, pan, and hover functionality.
* **Risk & Discovery Tools:** Includes a **Liquidity Filter** (based on 1M Volume) for risk management and a **Cross-Correlation Heatmap** for quick pair discovery.
* **Alerting System:** Features simple, user-defined alerts based on the real-time $Z$-score signal (e.g., $Z > 2$), providing traders with immediate signals.

---

## 1. Architecture and Design Rationale

The architecture reflects a highly **modular and decoupled** design, utilizing Python's `asyncio` framework for high-performance, non-blocking data handling. This structure ensures **extensibility** (e.g., adding new data feeds or analytics requires minimal rework) and clarity.

### Component Breakdown

| Component | Technology | Role & Extensibility Rationale |
| :--- | :--- | :--- |
| **Data Ingestion** | Python (`asyncio`, `websockets`) | Connects to multiple Binance WebSocket streams concurrently. Includes **Data Integrity Checks** to filter corrupt ticks (Price/Size $\le 0$) at the source. |
| **Data Bus / Processor** | `asyncio.Queue` / Python Class | **Decouples** the fast Ingestion Client from the aggregation logic. Aggregates ticks into required OHLCV bars (1S, 1M, 5M). |
| **Data Storage** | **SQLite** (`aiosqlite`) | Chosen for simplicity and zero-setup persistence for a local prototype. |
| **Analytics Engine** | Python (`Pandas`, `Statsmodels`, `pykalman`) | Computes all quantitative metrics, reading directly from the SQLite DB. |
| **Frontend/Viz** | **Streamlit** + **Plotly** | Provides the required interactive dashboard, controls, and dynamic charts with zoom, pan, and hover functionality. |

---

## 2. Setup and Execution

### Dependencies

Ensure you have Python (3.9+) and the following libraries installed:

```bash
pip install streamlit websockets pandas numpy statsmodels aiosqlite plotly pykalman
```

### Execution (Runnable App)

The application requires two separate terminal commands for operation:
Start Backend (Ingestion & Analytics): (This must run first to populate the database.)

```bash
python app.py
```

## 3. Quantitative Analytics Methodology

The `AnalyticsEngine` implements core and advanced techniques for cointegration trading.

| Feature | Calculation Method | Purpose |
| :--- | :--- | :--- |
| **Hedge Ratio ($\beta$)** | **Static OLS Regression** | Provides the conventional, fixed ratio for spread calculation. |
| **Dynamic Hedge ($\beta_t$)**| **Kalman Filter** | Computes a time-varying, adaptive hedge ratio, resulting in a more robust spread signal. |
| **Spread & Z-score** | Calculated from the chosen hedge ratio and normalized by a rolling mean ($\mu$) and standard deviation ($\sigma$). | Generates the core mean-reversion trading signal. |
| **Stationarity Test**| **Augmented Dickey-Fuller (ADF) Test** | Used to statistically confirm if the spread is mean-reverting (stationary). |

### Advanced Features (Trader Utility)

* **Mini Mean-Reversion Backtest:** Simulates the performance of the simple Z-score strategy ($\text{Entry}: |Z| > 2, \text{Exit}: |Z| \approx 0$) on historical bars.
* **Liquidity Filters:** Dynamically limits available asset pairs based on a minimum **1-Minute Volume** to support risk-aware pair selection.
* **Cross-Correlation Heatmap:** Provides a visual tool for quick pair discovery among all available assets.
* **Data Export/Upload:** Allows downloading processed data and includes functionality to upload external OHLC data for analysis.

Start Frontend (Dashboard):
```bash
streamlit run dashboard.py
```

