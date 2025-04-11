import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import pytz
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# 1. Streamlit Configuration & Auto-refresh
# -------------------------------
st.set_page_config(page_title="⚡ Realtime Energy Monitoring", layout="wide")
st.title("⚡ Realtime Energy Monitoring Dashboard")

# Auto-refresh every 1 minute (60000 ms)
st_autorefresh(interval=60000, limit=None, key="1min_refresh")

# Display IST time
ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
st.markdown(f"🕒 **Last Refreshed:** `{ist_now.strftime('%Y-%m-%d %H:%M:%S')} IST`")

# Manual refresh
if st.button("🔄 Refresh Now"):
    st.rerun()

# -------------------------------
# 2. Load Data from Google Sheets
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
# 3. Data Preparation
# -------------------------------
try:
    df_raw["DATETIME"] = pd.to_datetime(df_raw["DATE"] + " " + df_raw["TIME"], format="%Y-%m-%d %H:%M:%S")
except Exception as e:
    st.error(f"❌ Error parsing datetime: {e}")
    st.stop()

df_raw.sort_values("DATETIME", inplace=True)

# -------------------------------
# 4. Display Latest Snapshot
# -------------------------------
st.subheader("📈 Latest Data Snapshot")
st.dataframe(df_raw.tail(), use_container_width=True)

# -------------------------------
# 5. Real-time Graphs
# -------------------------------
st.subheader("📊 Real-time Sensor Readings")

col1, col2, col3 = st.columns(3)
with col1:
    fig_voltage = px.line(df_raw, x="DATETIME", y="VOLTAGE", title="🔌 Voltage Over Time", labels={"VOLTAGE": "Voltage (V)"})
    st.plotly_chart(fig_voltage, use_container_width=True)
with col2:
    fig_current = px.line(df_raw, x="DATETIME", y="CURRENT", title="🔋 Current Over Time", labels={"CURRENT": "Current (A)"})
    st.plotly_chart(fig_current, use_container_width=True)
with col3:
    fig_energy = px.bar(df_raw, x="DATETIME", y="ENERGY (kWh)", title="⚡ Energy Consumption Over Time", labels={"ENERGY (kWh)": "Energy (kWh)"}, color_discrete_sequence=["#00CC96"])
    fig_energy.update_layout(bargap=0.2)
    st.plotly_chart(fig_energy, use_container_width=True)

# -------------------------------
# 6. Meters (Gauges)
# -------------------------------
st.subheader("🧭 Live Meters (Gauges)")
latest = df_raw.iloc[-1]

gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
with gauge_col1:
    fig_v = go.Figure(go.Indicator(mode="gauge+number", value=latest["VOLTAGE"], title={"text": "Voltage (V)"}, gauge={"axis": {"range": [0, 300]}}))
    st.plotly_chart(fig_v, use_container_width=True)
with gauge_col2:
    fig_c = go.Figure(go.Indicator(mode="gauge+number", value=latest["CURRENT"], title={"text": "Current (A)"}, gauge={"axis": {"range": [0, 50]}}))
    st.plotly_chart(fig_c, use_container_width=True)
with gauge_col3:
    fig_e = go.Figure(go.Indicator(mode="gauge+number", value=latest["ENERGY (kWh)"], title={"text": "Energy (kWh)"}, gauge={"axis": {"range": [0, 10]}}))
    st.plotly_chart(fig_e, use_container_width=True)

# -------------------------------
# 7. Energy Cost Calculation
# -------------------------------
st.subheader("💸 Cost Estimation (in INR)")
cost_per_kwh = 7  # ₹7 per unit
total_energy = df_raw["ENERGY (kWh)"].sum()
total_cost = total_energy * cost_per_kwh
st.metric(label="Estimated Total Cost", value=f"₹ {total_cost:.2f}")

# -------------------------------
# 8. LSTM Forecast
# -------------------------------
st.subheader("🔮 Energy Consumption 10-Min Forecast")

df_raw["ENERGY (kWh)"] = pd.to_numeric(df_raw["ENERGY (kWh)"], errors="coerce")
df_energy = df_raw.dropna(subset=["ENERGY (kWh)"]).copy()

TIME_STEPS = 10
if len(df_energy) < TIME_STEPS:
    st.warning("⚠️ Not enough data to generate a forecast.")
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
        model = load_model("lstm_energy_forecast_model.h5", custom_objects={"LSTM": LSTM_wrapper, "mse": mse})
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    num_steps = 10
    predictions_scaled = []
    current_seq = last_sequence.copy()
    for i in range(num_steps):
        next_pred = model.predict(current_seq, verbose=0)
        predictions_scaled.append(next_pred[0, 0])
        current_seq = np.append(current_seq[:, 1:, :], [[[next_pred[0, 0]]]], axis=1)

    predictions_inv = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    conf_spread = predictions_inv * 0.05

    last_timestamp = df_energy["DATETIME"].iloc[-1]
    prev_timestamp = df_energy["DATETIME"].iloc[-2]
    interval = last_timestamp - prev_timestamp
    forecast_times = [last_timestamp + interval * (i + 1) for i in range(num_steps)]

    recent_history = df_energy.tail(50)
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=recent_history["DATETIME"], y=recent_history["ENERGY (kWh)"], mode="lines", name="History"))
    fig_forecast.add_trace(go.Scatter(x=forecast_times, y=predictions_inv, mode="lines+markers", name="Forecast", error_y=dict(type="data", array=conf_spread, visible=True)))
    fig_forecast.update_layout(title="⚡ 10-Minute Energy Forecast with Confidence Spread", xaxis_title="Time", yaxis_title="Energy (kWh)", template="plotly_white")
    st.plotly_chart(fig_forecast, use_container_width=True)

    forecast_df = pd.DataFrame({
        "Forecast Time": forecast_times,
        "Predicted Energy (kWh)": predictions_inv,
        "Confidence Spread (± kWh)": conf_spread
    })
    st.dataframe(forecast_df, use_container_width=True)

# -------------------------------
# 9. End of App
# -------------------------------
st.info("⏱️ This dashboard auto-refreshes every 1 minute. All readings are live and predictive models use LSTM with confidence spreads.")
