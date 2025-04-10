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
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# 1. Streamlit Configuration & Auto-refresh
# -------------------------------
st.set_page_config(page_title="📊 Realtime Energy Monitoring Dashboard", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 2rem; }
    .stButton>button { background-color: #0d6efd; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #0b5ed7; }
    </style>
""", unsafe_allow_html=True)

st.title(":zap: Realtime Energy Monitoring Dashboard")

# Auto-refresh every 1 minute (60000 ms)
st_autorefresh(interval=60000, limit=None, key="1min_refresh")

col1, col2 = st.columns([0.15, 0.85])
with col1:
    if st.button("Refresh Now"):
        st.rerun()
with col2:
    st.caption("Auto-refreshes every 1 minute. Click above to refresh manually.")

# -------------------------------
# 2. Load Data from Google Sheets Using st.secrets
# -------------------------------
@st.cache_data(show_spinner=False)
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
    return pd.DataFrame(data)

df_raw = load_google_sheet_data()

# -------------------------------
# 3. Data Preparation & Cleaning
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

with st.expander("Latest Data Snapshot", expanded=True):
    st.dataframe(df_raw.tail(), use_container_width=True)

# -------------------------------
# 4. Real-time Graphs
# -------------------------------
st.markdown("### 🌟 Real-time Sensor Readings")
col1, col2, col3 = st.columns(3)

with col1:
    fig_voltage = px.line(
        df_raw, x="DATETIME", y="VOLTAGE", title="Voltage (V)",
        labels={"VOLTAGE": "Voltage (V)"}
    )
    st.plotly_chart(fig_voltage, use_container_width=True)

with col2:
    fig_current = px.line(
        df_raw, x="DATETIME", y="CURRENT", title="Current (A)",
        labels={"CURRENT": "Current (A)"}
    )
    st.plotly_chart(fig_current, use_container_width=True)

with col3:
    fig_energy = px.line(
        df_raw, x="DATETIME", y="ENERGY (kWh)", title="Energy (kWh)",
        labels={"ENERGY (kWh)": "Energy (kWh)"}
    )
    st.plotly_chart(fig_energy, use_container_width=True)

# -------------------------------
# 5. Forecasting
# -------------------------------
st.markdown("### 🕒 10-Minute Forecast")

# Ensure numeric
df_raw["ENERGY (kWh)"] = pd.to_numeric(df_raw["ENERGY (kWh)"], errors="coerce")
df_energy = df_raw.dropna(subset=["ENERGY (kWh)"]).copy()

TIME_STEPS = 10
if len(df_energy) >= TIME_STEPS:
    scaler = MinMaxScaler()
    energy_scaled = scaler.fit_transform(df_energy[["ENERGY (kWh)"]].values)
    last_sequence = energy_scaled[-TIME_STEPS:].reshape((1, TIME_STEPS, 1))

    from tensorflow.keras.layers import LSTM as OriginalLSTM
    def LSTM_wrapper(*args, **kwargs):
        kwargs.pop("time_major", None)
        return OriginalLSTM(*args, **kwargs)

    @tf.keras.utils.register_keras_serializable()
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    try:
        model = load_model("lstm_energy_forecast_model.h5", custom_objects={"LSTM": LSTM_wrapper, "mse": mse})
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    predictions_scaled = []
    current_seq = last_sequence.copy()
    for _ in range(10):
        next_pred = model.predict(current_seq, verbose=0)
        predictions_scaled.append(next_pred[0, 0])
        current_seq = np.append(current_seq[:, 1:, :], [[[next_pred[0, 0]]]], axis=1)

    predictions_inv = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    conf_spread = predictions_inv * 0.05

    last_timestamp = df_energy["DATETIME"].iloc[-1]
    prev_timestamp = df_energy["DATETIME"].iloc[-2] if len(df_energy) >= 2 else timedelta(minutes=1)
    interval = last_timestamp - prev_timestamp
    forecast_times = [last_timestamp + interval * (i + 1) for i in range(10)]

    recent_history = df_energy.tail(50)
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=recent_history["DATETIME"], y=recent_history["ENERGY (kWh)"], mode="lines", name="Historical Energy"))
    fig_forecast.add_trace(go.Scatter(x=forecast_times, y=predictions_inv, mode="lines+markers", name="10-Min Forecast",
                                      error_y=dict(type="data", array=conf_spread, visible=True)))
    fig_forecast.update_layout(title="10-Minute Energy Forecast with Confidence Spread",
                               xaxis_title="Time", yaxis_title="Energy (kWh)", template="plotly_white")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.dataframe(pd.DataFrame({
        "Forecast Time": forecast_times,
        "Predicted Energy (kWh)": predictions_inv,
        "Confidence Spread (± kWh)": conf_spread
    }), use_container_width=True)

else:
    st.warning("Not enough data for forecasting.")

st.markdown("---")
st.info("This dashboard auto-refreshes every 1 minute. View historical sensor data and a 10-minute forecast with confidence intervals.")
