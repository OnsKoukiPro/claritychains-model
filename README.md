<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/banner-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/banner-light.svg?v=r1" alt="claritychains-model: price forecasts, supplier-concentration risk and a hedging optimiser for the critical materials behind batteries and electric vehicles" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/stats-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/stats-light.svg?v=r1" alt="6 public data sources, P10/P90 forecast bands, HHI concentration risk, CVaR hedge scoring" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/s1-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/s1-light.svg?v=r1" alt="01 / Forecast: where prices go" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/forecast-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/forecast-light.svg?v=r1" alt="Six-month price forecasts with P10 to P90 bands for three critical battery materials" width="100%"></picture>

Real output of `BaselineForecaster` (no fundamentals) on monthly averages of the prices in `data/raw/real_prices.csv`: rolling mean and volatility, momentum and trend, then a P50 path with mean reversion and bands that widen with the square root of the horizon.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/s2-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/s2-light.svg?v=r1" alt="02 / Pipeline: from raw data to a buy plan" width="100%"></picture>

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/pipeline-dark.svg?v=r1"><img src="https://raw.githubusercontent.com/OnsKoukiPro/claritychains-model/main/.github/assets/pipeline-light.svg?v=r1" alt="Six public data sources feed a pipeline into a forecaster, diversification engine and hedging optimiser that together produce a buy plan, with an offer-scoring agent" width="100%"></picture>

<details>
<summary><b>What each module does</b></summary>

| Module | Method |
|---|---|
| **Price forecasting** `src/models/baseline_forecaster.py` | rolling statistics, momentum, trend and volatility-regime detection. P10/P50/P90 bands, adjusted with GDELT geopolitical risk and EV-adoption demand signals. Backtesting included. |
| **Supplier diversification** `src/models/diversification_engine.py` | Herfindahl-Hirschman Index on import shares, target-HHI optimisation, alternative-supplier search, scenarios |
| **Hedging optimiser** `src/models/hedging_optimizer.py` | purchase-schedule optimisation over the forecast, strategy comparison, VaR / CVaR |
| **Data pipeline** `src/data_pipeline/` | World Bank, UN Comtrade, Yahoo Finance, GDELT, USGS, EV adoption |
| **AI offer analysis** `agent/` | FastAPI + LangChain: offer ingestion, gap and risk analysis, scoring, chat |

</details>

<details>
<summary><b>Run it locally</b></summary>

```bash
cp .env.example .env
docker compose up --build
```

- Agent API: http://localhost:8000
- Jupyter: http://localhost:8888

Or `./deploy-integrated-platform.sh` (creates the data directories, then builds and starts the stack). Sample data: `python scripts/generate-sample-data.py`.

</details>

Prototype for [ClarityChains](https://github.com/OnsKoukiPro/ClarityChains).
