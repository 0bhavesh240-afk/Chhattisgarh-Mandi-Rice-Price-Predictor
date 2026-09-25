# Chhattisgarh Mandi Rice Price Predictor

A small, GitHub-ready Python project for exploring and forecasting rice prices in Chhattisgarh mandis. It trains a time-aware Random Forest model from historical daily market observations and predicts the next 7 days for a selected mandi and rice variety.

> **Data note:** `data/sample_rice_prices.csv` is synthetic demonstration data, not official mandi prices. Replace it with verified historical prices before using forecasts for decisions.

## Features

- Clean and validate daily mandi price data
- Create lag and rolling-window features without looking into the future
- Time-ordered train/test evaluation
- Forecast 7 days recursively from the most recent available history
- Save a forecast CSV and a price-history/forecast chart
- Optional Streamlit dashboard

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.cli --data data/sample_rice_prices.csv --mandi Raipur --variety Common
```

Outputs are written to `outputs/forecast.csv` and `outputs/price_forecast.png`.

Run the optional dashboard:

```bash
streamlit run app.py
```

## Use your own data

CSV must include these columns:

| Column | Meaning |
| --- | --- |
| `date` | Observation date (`YYYY-MM-DD`) |
| `mandi` | Market name, e.g. Raipur |
| `variety` | Rice variety/grade |
| `modal_price` | Modal price in INR per quintal |

Example:

```csv
date,mandi,variety,modal_price
2024-01-01,Raipur,Common,2180
```

Pass a replacement file with `--data path/to/your.csv`. Use a consistent unit and daily frequency. Missing days are filled by carrying forward the last observed price within each mandi/variety. Forecasts are experimental and accuracy depends on data quality, history length, seasonality, and policy or weather changes.

## Model and evaluation

The model uses calendar features, recent lags (1, 7, 14, 28 days), and rolling means. The last 20% of chronological observations are held out for evaluation; the project reports MAE, RMSE, and MAPE. This is a baseline, not a causal model: it does not include arrivals, weather, MSP changes, transport costs, or quality differences.

## Project layout

```text
├── app.py
├── data/sample_rice_prices.csv
├── outputs/                 # generated results
├── src/
│   ├── cli.py
│   └── predictor.py
├── requirements.txt
└── .gitignore
```

## Data sources for real deployment

Use a documented, permitted source such as the Government of India's Agmarknet market-price data. Check its current access terms and field definitions, and record the source and retrieval date alongside any dataset you publish.

## License

MIT. See `LICENSE`.
