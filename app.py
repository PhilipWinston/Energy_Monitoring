import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import holidays
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# 🔧 Streamlit Page Setup & Refresh
# -------------------------------
st.set_page_config(page_title="⚡ Realtime Energy Monitoring", layout="wide")
st.title("⚡ Realtime Energy Monitoring Dashboard")

# Refresh setup
st_autorefresh(interval=60000, limit=None, key="1min_refresh")
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔁 Refresh Now"):
        st.rerun()
with col2:
    st.markdown(f"⏱️ Last auto-refresh: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

st.markdown("---")

# -------------------------------
# 🔗 Load Google Sheet Data
# -------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_google_sheet_data():
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["google_service_account"], scopes=SCOPES
    )
    SHEET_ID = "19A2rlYT-Whb24UFcLGDn0ngDCBg8WAXR8N1PDl9F0LQ"
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

df_raw = load_google_sheet_data()

# -------------------------------
# 📅 Data Preparation
# -------------------------------
try:
    df_raw["DATETIME"] = pd.to_datetime(
        df_raw["DATE"] + " " + df_raw["TIME"],
        format="%Y-%m-%d %H:%M:%S"
    )
except Exception as e:
    st.error(f"Error creating datetime column: {e}")
    st.stop()

df_raw.sort_values("DATETIME", inplace=True)

st.markdown("### 📋 Latest Data Snapshot")
st.dataframe(df_raw.tail(), use_container_width=True)
st.success(f"✅ Latest data timestamp: `{df_raw['DATETIME'].max()}`")

st.markdown("---")

# -------------------------------
# 📊 Real-time Sensor Graphs
# -------------------------------
st.markdown("## 📈 Real-time Sensor Readings")

fig_voltage = px.line(
    df_raw,
    x="DATETIME",
    y="VOLTAGE",
    title="🔌 Voltage Over Time",
    labels={"VOLTAGE": "Voltage (V)"},
    template="plotly_white"
)
st.plotly_chart(fig_voltage, use_container_width=True)

fig_current = px.line(
    df_raw,
    x="DATETIME",
    y="CURRENT",
    title="🔋 Current Over Time",
    labels={"CURRENT": "Current (A)"},
    template="plotly_white"
)
st.plotly_chart(fig_current, use_container_width=True)

fig_energy = px.bar(
    df_raw,
    x="DATETIME",
    y="ENERGY (kWh)",
    title="⚙️ Energy Consumption Over Time",
    labels={"ENERGY (kWh)": "Energy (kWh)"},
    color_discrete_sequence=["#00CC96"],
    template="plotly_white"
)
fig_energy.update_layout(bargap=0.2)
st.plotly_chart(fig_energy, use_container_width=True)

st.markdown("---")

# -------------------------------
# 🔮 Forecasting with LSTM
# -------------------------------
st.markdown("## 🔮 10-Minute Energy Forecast")
st.caption("Using LSTM model with confidence spread ±5%")

df_raw["ENERGY (kWh)"] = pd.to_numeric(df_raw["ENERGY (kWh)"], errors="coerce")
df_energy = df_raw.dropna(subset=["ENERGY (kWh)"]).copy()
TIME_STEPS = 10

if len(df_energy) < TIME_STEPS:
    st.warning("Not enough data to generate a forecast.")
else:
    scaler = MinMaxScaler()
    energy_values = df_energy[["ENERGY (kWh)"]].values
    energy_scaled = scaler.fit_transform(energy_values)
    last_sequence = energy_scaled[-TIME_STEPS:].reshape((1, TIME_STEPS, 1))

    from tensorflow.keras.layers import LSTM as OriginalLSTM
    def LSTM_wrapper(*args, **kwargs):
        kwargs.pop("time_major", None)
        return OriginalLSTM(*args, **kwargs)

    @tf.keras.utils.register_keras_serializable()
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    try:
        model = load_model("lstm_energy_forecast_model.h5", custom_objects={
            "LSTM": LSTM_wrapper,
            "mse": mse
        })
    except Exception as e:
        st.error(f"Error loading the LSTM model: {e}")
        st.stop()

    predictions_scaled = []
    current_seq = last_sequence.copy()
    for _ in range(10):
        next_pred = model.predict(current_seq, verbose=0)
        predictions_scaled.append(next_pred[0, 0])
        current_seq = np.append(current_seq[:, 1:, :], [[[next_pred[0, 0]]]], axis=1)

    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions_inv = scaler.inverse_transform(predictions_scaled).flatten()
    conf_spread = predictions_inv * 0.05

    last_timestamp = df_energy["DATETIME"].iloc[-1]
    prev_timestamp = df_energy["DATETIME"].iloc[-2] if len(df_energy) >= 2 else timedelta(minutes=1)
    interval = last_timestamp - prev_timestamp
    forecast_times = [last_timestamp + interval * (i + 1) for i in range(10)]

    # Forecast Plot
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=df_energy.tail(50)["DATETIME"],
        y=df_energy.tail(50)["ENERGY (kWh)"],
        mode="lines",
        name="Historical"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_times,
        y=predictions_inv,
        mode="lines+markers",
        name="Forecast",
        error_y=dict(type="data", array=conf_spread, visible=True)
    ))
    fig_forecast.update_layout(
        title="📉 10-Minute Forecast with ±5% Confidence",
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("### 🧾 Forecast Table")
    forecast_df = pd.DataFrame({
        "Forecast Time": forecast_times,
        "Predicted Energy (kWh)": predictions_inv,
        "Confidence Spread (± kWh)": conf_spread
    })
    st.dataframe(forecast_df, use_container_width=True)

st.markdown("---")
st.info("📡 This dashboard auto-refreshes every 1 minute and allows manual refresh.\n\n📈 Displays real-time sensor readings and a 10-minute forecast powered by LSTM model.")

