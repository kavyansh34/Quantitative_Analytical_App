# ⚡️ Real-Time Quant Analytics Stack

This project delivers a small but complete end-to-end analytical application designed to provide **real-time pair trading insights** using Binance Futures tick data. The system covers data ingestion, multi-timeframe processing, advanced statistical analysis, and interactive visualization via a Streamlit dashboard.

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

Start Frontend (Dashboard):
```bash
streamlit run dashboard.py
```
