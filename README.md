# ClarityChains Model — Critical Materials Procurement Intelligence

A Streamlit platform for buyers of critical battery materials (lithium, cobalt, nickel, copper). It combines public market data with statistical models to forecast prices, measure supplier concentration risk, and plan hedged purchase schedules. It also includes an LLM agent that analyses supplier offers.

Prototype for [ClarityChains](https://github.com/OnsKoukiPro/ClarityChains).

## What it does

| Module | Method |
|---|---|
| **Price forecasting** (`src/models/baseline_forecaster.py`) | rolling statistics, momentum, trend and volatility-regime detection. P10/P50/P90 bands, adjusted with GDELT geopolitical risk and EV-adoption demand signals. Includes backtesting. |
| **Supplier diversification** (`src/models/diversification_engine.py`) | Herfindahl-Hirschman Index (HHI) on import shares, target-HHI optimisation, alternative-supplier search, scenario analysis |
| **Hedging optimiser** (`src/models/hedging_optimizer.py`) | purchase-schedule optimisation over the forecast, strategy comparison, VaR / CVaR |
| **Data pipeline** (`src/data_pipeline/`) | World Bank, UN Comtrade trade flows, Yahoo Finance prices, GDELT events, USGS minerals, EV adoption |
| **AI offer analysis** (`agent/`) | FastAPI + LangChain agent: offer ingestion, gap and risk analysis, scoring, chat |

App tabs: AI Offer Analysis · Tender Management · Global Dashboard · Forecasting · Geopolitical Risk · Procurement · Supply Chain · Data Sources.

## Run

```bash
cp .env.example .env
docker compose up --build
```

- Streamlit app: http://localhost:8501
- Agent API: http://localhost:8000
- Jupyter: http://localhost:8888

Or use `./deploy-integrated-platform.sh` (it creates the data directories, then builds and starts the stack). Sample data generation: `python scripts/generate-sample-data.py`.

## Stack

Python, Streamlit, pandas / NumPy / SciPy, FastAPI, LangChain, Docker Compose.
