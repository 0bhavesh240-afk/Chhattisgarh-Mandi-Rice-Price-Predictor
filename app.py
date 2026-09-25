from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predictor import evaluate, forecast, load_data

st.set_page_config(page_title="Chhattisgarh Rice Mandi Predictor", page_icon="🌾", layout="wide")
st.title("🌾 Chhattisgarh Rice Mandi Price Predictor")
st.caption("Experimental short-term forecast from historical modal prices (INR per quintal).")
st.warning("The included CSV is synthetic demonstration data, not official market data. Upload verified records before using forecasts.")

uploaded = st.file_uploader("Upload a CSV", type="csv")
default_path = Path("data/sample_rice_prices.csv")
try:
    if uploaded:
        data = load_data(uploaded)
    elif default_path.exists():
        data = load_data(default_path)
    else:
        st.info("Upload a CSV with date, mandi, variety, and modal_price columns.")
        st.stop()
except Exception as exc:
    st.error(str(exc))
    st.stop()

mandis = sorted(data.mandi.unique().tolist())
mandi = st.selectbox("Mandi", mandis)
varieties = sorted(data.loc[data.mandi == mandi, "variety"].unique().tolist())
variety = st.selectbox("Rice variety / grade", varieties)
days = st.slider("Forecast horizon (days)", 1, 30, 7)
series = data[(data.mandi == mandi) & (data.variety == variety)].set_index("date").modal_price.sort_index()

if len(series) < 40:
    st.error("At least 40 daily observations are needed for a forecast.")
    st.stop()

try:
    prediction = forecast(series, days)
    metrics = evaluate(series)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

left, mid, right = st.columns(3)
left.metric("Holdout MAE", f"₹{metrics['MAE']:,.0f}/quintal")
mid.metric("Holdout RMSE", f"₹{metrics['RMSE']:,.0f}/quintal")
right.metric("Holdout MAPE", f"{metrics['MAPE_percent']:.1f}%")
st.subheader("Forecast")
st.dataframe(prediction, use_container_width=True, hide_index=True)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.iloc[-90:].index, series.iloc[-90:].values, label="Historical")
ax.plot(prediction.date, prediction.predicted_price, marker="o", linestyle="--", label="Forecast")
ax.set_ylabel("INR per quintal")
ax.grid(alpha=0.25)
ax.legend()
st.pyplot(fig)
st.download_button("Download forecast CSV", prediction.to_csv(index=False), "rice_price_forecast.csv", "text/csv")
st.caption("This baseline does not account for arrivals, weather, MSP changes, transport costs, or quality differences.")
