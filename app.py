import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import gspread
from google.oauth2.service_account import Credentials
import json
import os
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Realtime Energy Monitoring",
    page_icon="⚡",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-big {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066cc;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for Google Sheets
@st.cache_resource
def get_google_sheet_credentials():
    """Get Google Sheets credentials."""
    # For Streamlit Cloud, use secrets management
    if os.path.exists(".streamlit/secrets.toml"):
        creds_info = st.secrets["gcp_service_account"]
        creds_dict = {
            "type": creds_info["type"],
            "project_id": creds_info["project_id"],
            "private_key_id": creds_info["private_key_id"],
            "private_key": creds_info["private_key"].replace('\\n', '\n'),
            "client_email": creds_info["client_email"],
            "client_id": creds_info["client_id"],
            "auth_uri": creds_info["auth_uri"],
            "token_uri": creds_info["token_uri"],
            "auth_provider_x509_cert_url": creds_info["auth_provider_x509_cert_url"],
            "client_x509_cert_url": creds_info["client_x509_cert_url"]
        }
        return creds_dict
    else:
        # For local development, look for credentials file
        try:
            with open('credentials.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error("Google Sheets credentials not found. Please set up your credentials.")
            return None

def load_data_from_sheets(sheet_id):
    """Load data from Google Sheets."""
    creds_dict = get_google_sheet_credentials()
    
    if not creds_dict:
        # Return sample data if credentials aren't available
        return generate_sample_data()
    
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # Create credentials from dictionary
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(creds)
    
    try:
        # Try to open the specified sheet
        spreadsheet = client.open_by_key(sheet_id)
        sheet = spreadsheet.worksheet("cleaned_data")  # Use the cleaned data sheet
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Convert DATE / TIME to datetime
        df['DATE / TIME'] = pd.to_datetime(df['DATE / TIME'])
        
        # Convert numeric columns
        numeric_cols = ['VOLTAGE', 'CURRENT', 'POWER', 'ENERGY (kWh)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data for testing when Google Sheets is not available."""
    now = datetime.now()
    dates = [now - timedelta(minutes=i*10) for i in range(100, 0, -1)]
    
    # Create sample data with some realistic patterns
    df = pd.DataFrame({
        'DATE / TIME': dates,
        'VOLTAGE': [220 + np.random.normal(0, 5) for _ in range(100)],
        'CURRENT': [5 + 2*np.sin(i/10) + np.random.normal(0, 0.5) for i in range(100)],
        'POWER': [1100 + 200*np.sin(i/10) + np.random.normal(0, 50) for i in range(100)],
        'ENERGY (kWh)': [0.01 * i + np.random.normal(0, 0.005) for i in range(100)]
    })
    
    # Calculate cumulative energy
    df['CUMULATIVE_ENERGY'] = df['ENERGY (kWh)'].cumsum()
    
    return df

# Load and prepare model
@st.cache_resource
def load_lstm_model():
    """Load the pre-trained LSTM model."""
    try:
        model = load_model("lstm_energy_forecast_model.h5")
        return model
    except:
        st.warning("Pre-trained model not found. Using a mock model for demonstration.")
        # Create a simple mock model for demonstration
        mock_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10, 1)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1)
        ])
        mock_model.compile(optimizer='adam', loss='mse')
        return mock_model

def predict_energy(df, model, time_steps=10):
    """Make energy consumption predictions using the LSTM model."""
    if len(df) < time_steps:
        return None, None
    
    # Prepare data for prediction
    energy_data = df['ENERGY (kWh)'].values
    
    # Normalize the data
    scaler = MinMaxScaler()
    energy_scaled = scaler.fit_transform(energy_data.reshape(-1, 1))
    
    # Create the input sequence
    X_input = energy_scaled[-time_steps:].reshape(1, time_steps, 1)
    
    # Make prediction
    prediction_scaled = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction_scaled)
    
    # Generate future timestamps for predictions
    last_date = df['DATE / TIME'].iloc[-1]
    future_dates = [last_date + timedelta(minutes=(i+1)*10) for i in range(10)]
    
    # Create future values (simple forecast extending the last few points)
    last_values = energy_data[-5:]
    slope = (last_values[-1] - last_values[0]) / 4
    future_values = [energy_data[-1] + slope * (i+1) for i in range(10)]
    
    # Create a combined DataFrame
    historical_df = pd.DataFrame({
        'DATE / TIME': df['DATE / TIME'].values,
        'ENERGY (kWh)': energy_data,
        'Type': 'Historical'
    })
    
    forecast_df = pd.DataFrame({
        'DATE / TIME': future_dates,
        'ENERGY (kWh)': future_values,
        'Type': 'Forecast'
    })
    
    combined_df = pd.concat([historical_df, forecast_df])
    
    return combined_df, prediction[0][0]

# Main app
def main():
    # Main title
    st.markdown('<div class="main-header">Real-time Energy Monitoring System</div>', unsafe_allow_html=True)
    
    # Sheet ID from the original code
    SHEET_ID = "19A2rlYT-Whb24UFcLGDn0ngDCBg8WAXR8N1PDl9F0LQ"
    
    # Initialize session state for periodic updates
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now() - timedelta(minutes=10)  # Force initial update
        st.session_state.update_count = 0
    
    # Check if it's time for an update (every 10 minutes)
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_update).total_seconds() / 60
    
    # Placeholder for refresh button and last update info
    refresh_col, update_col = st.columns([1, 3])
    
    with refresh_col:
        if st.button("Refresh Data"):
            st.session_state.last_update = current_time
            st.session_state.update_count += 1
            st.experimental_rerun()
    
    with update_col:
        st.markdown(f"**Last updated:** {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        if time_diff < 10:
            st.markdown(f"Next update in: **{int(10 - time_diff)} minutes**")
    
    # Auto-refresh logic
    if time_diff >= 10:
        st.session_state.last_update = current_time
        st.session_state.update_count += 1
        st.experimental_rerun()
    
    # Load data
    with st.spinner("Loading energy monitoring data..."):
        df = load_data_from_sheets(SHEET_ID)
    
    if df is None or df.empty:
        st.error("No data available. Please check your data source.")
        return

    # Current metrics row
    st.markdown('<div class="sub-header">Current Readings</div>', unsafe_allow_html=True)
    
    # Get latest values
    latest = df.iloc[-1]
    
    metric1, metric2, metric3, metric4 = st.columns(4)
    
    with metric1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Voltage</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{latest["VOLTAGE"]:.1f} V</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Current</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{latest["CURRENT"]:.2f} A</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Power</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-big">{latest["POWER"]:.1f} W</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Energy Today</div>', unsafe_allow_html=True)
        
        # Calculate energy for today
        today = pd.Timestamp.now().floor('D')
        today_energy = df[df['DATE / TIME'].dt.date == today.date()]['ENERGY (kWh)'].sum()
        
        st.markdown(f'<div class="metric-big">{today_energy:.2f} kWh</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Real-time monitoring charts
    st.markdown('<div class="sub-header">Real-time Monitoring</div>', unsafe_allow_html=True)
    
    # Filter for the last 24 hours of data for the real-time view
    last_24h = datetime.now() - timedelta(hours=24)
    recent_df = df[df['DATE / TIME'] > last_24h]
    
    if len(recent_df) == 0:
        recent_df = df.tail(100)  # Fallback if no recent data
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Voltage", "Current", "Power"])
    
    with tab1:
        fig_voltage = px.line(
            recent_df, 
            x="DATE / TIME", 
            y="VOLTAGE", 
            title="Voltage Over Time",
            labels={"VOLTAGE": "Voltage (V)", "DATE / TIME": "Time"}
        )
        fig_voltage.update_layout(height=400)
        st.plotly_chart(fig_voltage, use_container_width=True)
    
    with tab2:
        fig_current = px.line(
            recent_df, 
            x="DATE / TIME", 
            y="CURRENT", 
            title="Current Over Time",
            labels={"CURRENT": "Current (A)", "DATE / TIME": "Time"}
        )
        fig_current.update_layout(height=400)
        st.plotly_chart(fig_current, use_container_width=True)
    
    with tab3:
        fig_power = px.line(
            recent_df, 
            x="DATE / TIME", 
            y="POWER", 
            title="Power Consumption Over Time",
            labels={"POWER": "Power (W)", "DATE / TIME": "Time"}
        )
        fig_power.update_layout(height=400)
        st.plotly_chart(fig_power, use_container_width=True)
    
    # Energy consumption analysis
    st.markdown('<div class="sub-header">Energy Consumption Analysis</div>', unsafe_allow_html=True)
    
    # Calculate cumulative energy if it doesn't exist
    if 'CUMULATIVE_ENERGY' not in df.columns:
        df['CUMULATIVE_ENERGY'] = df['ENERGY (kWh)'].cumsum()
    
    # Energy consumption chart
    fig_energy = px.line(
        df, 
        x="DATE / TIME", 
        y="CUMULATIVE_ENERGY", 
        title="Cumulative Energy Consumption",
        labels={"CUMULATIVE_ENERGY": "Energy (kWh)", "DATE / TIME": "Time"}
    )
    fig_energy.update_layout(height=400)
    st.plotly_chart(fig_energy, use_container_width=True)
    
    # Time Series Prediction
    st.markdown('<div class="sub-header">Time Series Analysis & Prediction</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading prediction model..."):
        model = load_lstm_model()
    
    # Make predictions
    prediction_df, next_value = predict_energy(df, model)
    
    if prediction_df is not None:
        # Create forecast visualization
        historical = prediction_df[prediction_df['Type'] == 'Historical'].tail(50)
        forecast = prediction_df[prediction_df['Type'] == 'Forecast']
        
        fig_forecast = go.Figure()
        
        # Add historical data
        fig_forecast.add_trace(go.Scatter(
            x=historical['DATE / TIME'],
            y=historical['ENERGY (kWh)'],
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecast data
        fig_forecast.add_trace(go.Scatter(
            x=forecast['DATE / TIME'],
            y=forecast['ENERGY (kWh)'],
            name='Forecast (Next 10 minutes)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_forecast.update_layout(
            title="Energy Consumption Forecast",
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            legend=dict(x=0, y=1),
            height=500
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Prediction insights
        st.markdown(f"**Predicted energy consumption for next interval:** {next_value:.4f} kWh")
        
        # Daily summary
        st.markdown('<div class="sub-header">Daily Energy Summary</div>', unsafe_allow_html=True)
        
        # Group by day and calculate daily totals
        df['Date'] = df['DATE / TIME'].dt.date
        daily_energy = df.groupby('Date')['ENERGY (kWh)'].sum().reset_index()
        daily_energy = daily_energy.tail(14)  # Last two weeks
        
        fig_daily = px.bar(
            daily_energy,
            x='Date',
            y='ENERGY (kWh)',
            title="Daily Energy Consumption (Last 14 Days)",
            labels={"ENERGY (kWh)": "Energy (kWh)", "Date": "Date"}
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
    else:
        st.warning("Not enough data for time series prediction")

if __name__ == "__main__":
    main()