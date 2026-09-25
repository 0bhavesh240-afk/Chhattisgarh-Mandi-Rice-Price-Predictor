from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.predictor import evaluate, forecast, load_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast rice prices for a Chhattisgarh mandi.")
    parser.add_argument("--data", default="data/sample_rice_prices.csv", help="Input CSV path")
    parser.add_argument("--mandi", default="Raipur", help="Mandi name")
    parser.add_argument("--variety", default="Common", help="Rice variety/grade")
    parser.add_argument("--days", type=int, default=7, help="Forecast horizon (1-30 days)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for forecast and chart")
    args = parser.parse_args()
    if not 1 <= args.days <= 30:
        parser.error("--days must be between 1 and 30")

    data = load_data(args.data)
    selection = data[(data.mandi.str.casefold() == args.mandi.casefold()) &
                     (data.variety.str.casefold() == args.variety.casefold())]
    if selection.empty:
        available = data[["mandi", "variety"]].drop_duplicates().to_dict("records")
        raise SystemExit(f"No data for {args.mandi}/{args.variety}. Available combinations: {available}")
    series = selection.set_index("date")["modal_price"].sort_index()
    metrics = evaluate(series)
    result = forecast(series, args.days)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "forecast.csv"
    chart_path = out / "price_forecast.png"
    result.insert(1, "mandi", args.mandi)
    result.insert(2, "variety", args.variety)
    result.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(series.iloc[-90:].index, series.iloc[-90:].values, label="Historical modal price")
    plt.plot(result.date, result.predicted_price, marker="o", linestyle="--", label="Forecast")
    plt.title(f"Rice price forecast — {args.mandi}, {args.variety}")
    plt.xlabel("Date")
    plt.ylabel("INR per quintal")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()

    print(f"Evaluation on chronological holdout: {metrics}")
    print(f"Forecast saved to {csv_path}")
    print(f"Chart saved to {chart_path}")
    print("Reminder: forecasts are experimental; verify against official market data.")


if __name__ == "__main__":
    main()
