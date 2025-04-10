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

# Optional: if using auto-refresh, ensure streamlit-autorefresh is installed.
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# 1. Streamlit Configuration & Optional Auto-refresh
# -------------------------------
st.set_page_config(page_title="Realtime Monitoring")
st.title("Realtime Monitoring")

# Uncomment the next line if you wish to auto-refresh every 10 minutes (600000 ms)
st_autorefresh(interval=600000, limit=None, key="10min_refresh")

# -------------------------------
# 2. Load Data from Google Sheets Using st.secrets
# -------------------------------
@st.cache_data(show_spinner=False)
def load_google_sheet_data():
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(st.secrets["google_service_account"], scopes=SCOPES)
    # Replace with your actual Sheet ID
    SHEET_ID = "19A2rlYT-Whb24UFcLGDn0ngDCBg8WAXR8N1PDl9F0LQ"
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1  # Open first sheet
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

df_raw = load_google_sheet_data()

# -------------------------------
# 3. Data Preparation & Cleaning
# -------------------------------
try:
    df_raw["DATETIME"] = pd.to_datetime(df_raw["DATE"] + " " + df_raw["TIME"], format="%Y-%m-%d %H:%M:%S")
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

fig_voltage = px.line(df_raw, x="DATETIME", y="VOLTAGE", title="Voltage Over Time",
                      labels={"VOLTAGE": "Voltage (V)"})
st.plotly_chart(fig_voltage, use_container_width=True)

fig_current = px.line(df_raw, x="DATETIME", y="CURRENT", title="Current Over Time",
                      labels={"CURRENT": "Current (A)"})
st.plotly_chart(fig_current, use_container_width=True)

fig_energy = px.line(df_raw, x="DATETIME", y="ENERGY (kWh)", title="Energy Consumption Over Time",
                     labels={"ENERGY (kWh)": "Energy (kWh)"})
st.plotly_chart(fig_energy, use_container_width=True)

# -------------------------------
# 5. Time Series Forecasting using Pre-trained LSTM Model
# -------------------------------
st.subheader("Energy Consumption Forecast")

# Ensure ENERGY (kWh) is numeric
df_raw["ENERGY (kWh)"] = pd.to_numeric(df_raw["ENERGY (kWh)"], errors="coerce")
df_energy = df_raw.dropna(subset=["ENERGY (kWh)"]).copy()

TIME_STEPS = 10
if len(df_energy) < TIME_STEPS:
    st.warning("Not enough data to generate a forecast.")
else:
    # Normalize the energy values using MinMaxScaler
    scaler = MinMaxScaler()
    energy_values = df_energy[["ENERGY (kWh)"]].values
    energy_scaled = scaler.fit_transform(energy_values)

    # Create the last sequence from the latest data points for prediction
    last_sequence = energy_scaled[-TIME_STEPS:]
    last_sequence = last_sequence.reshape((1, TIME_STEPS, 1))
    
    # ---- Define a custom LSTM wrapper to remove the 'time_major' keyword ----
    from tensorflow.keras.layers import LSTM as OriginalLSTM
    def LSTM_wrapper(*args, **kwargs):
        if "time_major" in kwargs:
            kwargs.pop("time_major")
        return OriginalLSTM(*args, **kwargs)
    
    # ---- Define and register a custom mse function ----
    @tf.keras.saving.register_keras_serializable()
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
    
    # Load the pre-trained model with the custom_objects mapping
    try:
        custom_objects = {
            "LSTM": LSTM_wrapper,
            "mse": mse
        }
        model = load_model("lstm_energy_forecast_model.h5", custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Error loading the LSTM model: {e}")
        st.stop()

    # Predict the next energy consumption value
    pred_scaled = model.predict(last_sequence)
    pred_energy = scaler.inverse_transform(pred_scaled)[0, 0]

    # Estimate the next timestamp (assuming constant measurement interval)
    if len(df_energy) >= 2:
        last_timestamp = df_energy["DATETIME"].iloc[-1]
        prev_timestamp = df_energy["DATETIME"].iloc[-2]
        interval = last_timestamp - prev_timestamp
    else:
        interval = timedelta(minutes=10)
    pred_timestamp = last_timestamp + interval

    # Create a forecast DataFrame for plotting
    forecast_df = pd.DataFrame({
        "DATETIME": [pred_timestamp],
        "ENERGY (kWh)_Forecast": [pred_energy]
    })

    # Plot historical energy data (last 50 records) with the forecast point
    recent_history = df_energy.tail(50)
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=recent_history["DATETIME"],
        y=recent_history["ENERGY (kWh)"],
        mode="lines",
        name="Historical Energy"
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df["DATETIME"],
        y=forecast_df["ENERGY (kWh)_Forecast"],
        mode="markers+lines",
        marker=dict(size=10, color="red"),
        name="Forecast"
    ))
    fig_forecast.update_layout(
        title="Energy Forecast",
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.write(f"**Next predicted energy consumption:** {pred_energy:.2f} kWh at {pred_timestamp}")

# -------------------------------
# 6. End of App
# -------------------------------
st.info("This dashboard auto-refreshes every 10 minutes to display real-time monitoring and forecast updates.")
