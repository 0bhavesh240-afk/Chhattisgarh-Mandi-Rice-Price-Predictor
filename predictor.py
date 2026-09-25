from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

REQUIRED = {"date", "mandi", "variety", "modal_price"}
LAGS = (1, 7, 14, 28)


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    df = df[list(REQUIRED)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")
    df["mandi"] = df["mandi"].astype("string").str.strip()
    df["variety"] = df["variety"].astype("string").str.strip()
    df = df.dropna(subset=["date", "modal_price", "mandi", "variety"])
    df = df[df["modal_price"] > 0]
    df = df.sort_values(["mandi", "variety", "date"])
    df = df.drop_duplicates(["mandi", "variety", "date"], keep="last")
    if df.empty:
        raise ValueError("No valid positive price observations found.")
    return df.reset_index(drop=True)


def _series_frame(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"date": series.index, "price": series.values})
    frame["year"] = frame.date.dt.year
    frame["month"] = frame.date.dt.month
    frame["dayofyear"] = frame.date.dt.dayofyear
    frame["dayofweek"] = frame.date.dt.dayofweek
    frame["month_sin"] = np.sin(2 * np.pi * frame.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame.month / 12)
    for lag in LAGS:
        frame[f"lag_{lag}"] = frame.price.shift(lag)
    frame["rolling_7"] = frame.price.shift(1).rolling(7).mean()
    frame["rolling_28"] = frame.price.shift(1).rolling(28).mean()
    return frame.dropna().reset_index(drop=True)


def _make_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)


def evaluate(series: pd.Series) -> dict[str, float]:
    frame = _series_frame(series)
    if len(frame) < 20:
        raise ValueError("At least 48 daily observations are recommended for evaluation.")
    split = max(int(len(frame) * 0.8), len(frame) - 90)
    split = min(split, len(frame) - 1)
    features = [c for c in frame.columns if c not in {"date", "price"}]
    model = _make_model().fit(frame.loc[: split - 1, features], frame.loc[: split - 1, "price"])
    pred = model.predict(frame.loc[split:, features])
    actual = frame.loc[split:, "price"].to_numpy()
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAPE_percent": float(np.mean(np.abs((actual - pred) / actual)) * 100),
    }


def forecast(series: pd.Series, days: int = 7) -> pd.DataFrame:
    """Train on one daily price series and recursively forecast the next days."""
    series = series.sort_index().astype(float)
    series = series[~series.index.duplicated(keep="last")]
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full_index).ffill().dropna()
    if len(series) < 40:
        raise ValueError("Need at least 40 daily observations to create lag features.")
    train = _series_frame(series)
    features = [c for c in train.columns if c not in {"date", "price"}]
    model = _make_model().fit(train[features], train["price"])
    history = series.copy()
    rows = []
    for _ in range(days):
        date = history.index[-1] + pd.Timedelta(days=1)
        row = {
            "year": date.year, "month": date.month, "dayofyear": date.dayofyear,
            "dayofweek": date.dayofweek,
            "month_sin": np.sin(2 * np.pi * date.month / 12),
            "month_cos": np.cos(2 * np.pi * date.month / 12),
        }
        for lag in LAGS:
            row[f"lag_{lag}"] = history.iloc[-lag]
        row["rolling_7"] = history.iloc[-7:].mean()
        row["rolling_28"] = history.iloc[-28:].mean()
        value = max(0.0, float(model.predict(pd.DataFrame([row])[features])[0]))
        rows.append({"date": date, "predicted_price": round(value, 2)})
        history.loc[date] = value
    return pd.DataFrame(rows)
