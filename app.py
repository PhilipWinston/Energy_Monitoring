import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from google.oauth2 import service_account
import gspread

# Set page config
st.set_page_config(page_title="⚡ Real-Time Energy Monitoring", layout="wide")

# ----------------------------
# Google Sheets authentication
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["google_credentials"],
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
gc = gspread.authorize(credentials)

# ----------------------------
# Load and clean data
def load_data():
    sheet_url = st.secrets["sheet_url"]
    spreadsheet = gc.open_by_url(sheet_url)
    worksheet = spreadsheet.get_worksheet(0)
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    # Clean and process
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], errors='coerce')
    df = df.dropna(subset=['DATETIME'])

    numeric_cols = ['VOLTAGE (V)', 'CURRENT (A)', 'ENERGY (kWh)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_cols)
    df = df.sort_values('DATETIME').reset_index(drop=True)
    return df

# ----------------------------
# LSTM Model Training
def train_lstm_model(df, time_steps=10):
    energy = df['ENERGY (kWh)'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    energy_scaled = scaler.fit_transform(energy)

    X, y = [], []
    for i in range(time_steps, len(energy_scaled)):
        X.append(energy_scaled[i-time_steps:i])
        y.append(energy_scaled[i])

    X, y = np.array(X), np.array(y)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model, scaler

# ----------------------------
# Predict next 10 minutes
def predict_next(model, df, scaler, time_steps=10):
    last_values = df['ENERGY (kWh)'].values[-time_steps:]
    last_scaled = scaler.transform(last_values.reshape(-1, 1))
    input_seq = last_scaled.reshape(1, time_steps, 1)
    prediction = model.predict(input_seq, verbose=0)
    predicted_value = scaler.inverse_transform(prediction)[0, 0]
    return predicted_value

# ----------------------------
# Main display
def main():
    st.title("⚡ Real-Time Energy Monitoring Dashboard")

    # Auto-refresh every 60 seconds
    st_autorefresh = st.experimental_data_editor({"Refresh Rate (s)": 60})
    st_autorefresh

    refresh_interval = 60  # seconds
    count = st.experimental_get_query_params().get("count", [0])[0]
    st.experimental_set_query_params(count=int(count) + 1)

    df = load_data()
    if df.empty:
        st.error("No data found or failed to clean the data.")
        return

    # Train model and predict
    model, scaler = train_lstm_model(df)
    next_10min = predict_next(model, df, scaler)

    # Latest values
    latest = df.iloc[-1]
    st.subheader("📍 Live Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Voltage (V)", f"{latest['VOLTAGE (V)']:.2f}")
    col2.metric("Current (A)", f"{latest['CURRENT (A)']:.2f}")
    col3.metric("Energy (kWh)", f"{latest['ENERGY (kWh)']:.4f}")

    # Forecast and metrics
    st.subheader("🔮 Prediction & Daily Summary")
    today = df[df['DATETIME'].dt.date == df['DATETIME'].max().date()]
    col4, col5 = st.columns(2)
    col4.metric("Total Energy Today", f"{today['ENERGY (kWh)'].sum():.2f} kWh")
    col5.metric("Predicted Next 10 Min", f"{next_10min:.4f} kWh")

    # Graphs
    st.subheader("📈 Live Graphs")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['DATETIME'], y=df['VOLTAGE (V)'], name="Voltage", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['DATETIME'], y=df['CURRENT (A)'], name="Current", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df['DATETIME'], y=df['ENERGY (kWh)'], name="Energy", line=dict(color='blue')))

    # Add prediction point
    predicted_time = df['DATETIME'].iloc[-1] + timedelta(minutes=10)
    fig.add_trace(go.Scatter(x=[predicted_time], y=[next_10min], mode='markers+text',
                             marker=dict(color='red', size=10),
                             text=["10-min Forecast"],
                             textposition="top center",
                             name="Prediction"))

    fig.update_layout(title="Energy Monitoring", xaxis_title="Time", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧾 Raw Data (Latest)")
    st.dataframe(df.tail(10), use_container_width=True)

# ----------------------------
if __name__ == "__main__":
    main()
