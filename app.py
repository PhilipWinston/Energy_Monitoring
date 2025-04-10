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

# Optional auto-refresh
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# 1. Streamlit Configuration & Auto-refresh
# -------------------------------
st.set_page_config(page_title="Realtime Monitoring")
st.title("Realtime Monitoring")

# Auto-refresh every 1 minute (60000 ms)
st_autorefresh(interval=60000, limit=None, key="1min_refresh")

# Also show a manual "Refresh Now" button.
if st.button("Refresh Now"):
    st.rerun()

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
    df = pd.DataFrame(data)
    return df

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

st.subheader("Latest Data Snapshot")
st.write(df_raw.tail())

# -------------------------------
# 4. Real-time Graphs: Voltage, Current, and Energy
# -------------------------------
st.subheader("Real-time Sensor Readings")

fig_voltage = px.line(
    df_raw,
    x="DATETIME",
    y="VOLTAGE",
    title="Voltage Over Time",
    labels={"VOLTAGE": "Voltage (V)"}
)
st.plotly_chart(fig_voltage, use_container_width=True)

fig_current = px.line(
    df_raw,
    x="DATETIME",
    y="CURRENT",
    title="Current Over Time",
    labels={"CURRENT": "Current (A)"}
)
st.plotly_chart(fig_current, use_container_width=True)

fig_energy = px.bar(
    df_raw,
    x="DATETIME",
    y="ENERGY (kWh)",
    title="Energy Consumption Over Time (Column Chart)",
    labels={"ENERGY (kWh)": "Energy (kWh)"},
    color_discrete_sequence=["#00CC96"]
)
fig_energy.update_layout(
    bargap=0.2,
    xaxis_title="Time",
    yaxis_title="Energy (kWh)",
    template="plotly_white"
)
st.plotly_chart(fig_energy, use_container_width=True)

# -------------------------------
# 5. Time Series Forecasting for Next 10 Minutes using Pre-trained LSTM Model
# -------------------------------
st.subheader("Energy Consumption 10-Min Forecast with Confidence Spread")

df_raw["ENERGY (kWh)"] = pd.to_numeric(df_raw["ENERGY (kWh)"], errors="coerce")
df_energy = df_raw.dropna(subset=["ENERGY (kWh)"]).copy()

TIME_STEPS = 10

if len(df_energy) < TIME_STEPS:
    st.warning("Not enough data to generate a forecast.")
else:
    scaler = MinMaxScaler()
    energy_values = df_energy[["ENERGY (kWh)"]].values
    energy_scaled = scaler.fit_transform(energy_values)

    last_sequence = energy_scaled[-TIME_STEPS:]
    last_sequence = last_sequence.reshape((1, TIME_STEPS, 1))

    from tensorflow.keras.layers import LSTM as OriginalLSTM
    def LSTM_wrapper(*args, **kwargs):
        if "time_major" in kwargs:
            kwargs.pop("time_major")
        return OriginalLSTM(*args, **kwargs)

    @tf.keras.utils.register_keras_serializable()
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    try:
        custom_objects = {
            "LSTM": LSTM_wrapper,
            "mse": mse
        }
        model = load_model("lstm_energy_forecast_model.h5", custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Error loading the LSTM model: {e}")
        st.stop()

    num_steps = 10
    predictions_scaled = []
    current_seq = last_sequence.copy()
    for i in range(num_steps):
        next_pred = model.predict(current_seq)
        predictions_scaled.append(next_pred[0, 0])
        current_seq = np.append(current_seq[:, 1:, :], [[[next_pred[0,0]]]], axis=1)

    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions_inv = scaler.inverse_transform(predictions_scaled).flatten()
    conf_spread = predictions_inv * 0.05

    if len(df_energy) >= 2:
        last_timestamp = df_energy["DATETIME"].iloc[-1]
        prev_timestamp = df_energy["DATETIME"].iloc[-2]
        interval = last_timestamp - prev_timestamp
    else:
        interval = timedelta(minutes=1)

    forecast_times = [last_timestamp + interval * (i+1) for i in range(num_steps)]

    recent_history = df_energy.tail(50)
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=recent_history["DATETIME"],
        y=recent_history["ENERGY (kWh)"],
        mode="lines",
        name="Historical Energy"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_times,
        y=predictions_inv,
        mode="lines+markers",
        name="10-Min Forecast",
        error_y=dict(
            type="data",
            array=conf_spread,
            visible=True
        )
    ))
    fig_forecast.update_layout(
        title="10-Minute Energy Forecast with Confidence Spread",
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    forecast_df = pd.DataFrame({
        "Forecast Time": forecast_times,
        "Predicted Energy (kWh)": predictions_inv,
        "Confidence Spread (± kWh)": conf_spread
    })
    st.write("### Forecast Data", forecast_df)

# -------------------------------
# 6. End of App
# -------------------------------
st.info("This dashboard auto-refreshes every 1 minute and allows manual refresh. It displays real-time sensor readings and a 10-minute forecast with confidence spread.")
