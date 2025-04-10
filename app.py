import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, date, timedelta
import holidays
import time
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Realtime Energy Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚡ Realtime Energy Monitoring Dashboard")
st.markdown("This dashboard provides realtime monitoring of voltage, current, and energy consumption.")

# Create placeholder for connection status
connection_status = st.empty()

# Function to authenticate with Google Sheets
@st.cache_resource
def get_google_client():
    # For Streamlit Cloud, we'll use secrets management
    # Create a dictionary from secrets
    if os.path.exists(".streamlit/secrets.toml"):
        # Local development with secrets.toml
        credentials = st.secrets["google_credentials"]
        # Convert to a service account info dictionary
        service_account_info = {
            "type": credentials["type"],
            "project_id": credentials["project_id"],
            "private_key_id": credentials["private_key_id"],
            "private_key": credentials["private_key"].replace('\\n', '\n'),
            "client_email": credentials["client_email"],
            "client_id": credentials["client_id"],
            "auth_uri": credentials["auth_uri"],
            "token_uri": credentials["token_uri"],
            "auth_provider_x509_cert_url": credentials["auth_provider_x509_cert_url"],
            "client_x509_cert_url": credentials["client_x509_cert_url"],
        }
    else:
        # For Streamlit Cloud deployment
        credentials_json = st.secrets["google_credentials"]
        service_account_info = json.loads(credentials_json)
    
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    
    return client

# Function to load data
def load_data(client):
    SHEET_ID = st.secrets["sheet_id"]
    sheet = client.open_by_key(SHEET_ID).sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    # Clean and process data
    df_cleaned = df.copy()
    
    # Handle datetime conversion
    if "DATE" in df_cleaned.columns and "TIME" in df_cleaned.columns:
        # Option 1: Separate DATE and TIME columns
        df_cleaned["DATETIME"] = pd.to_datetime(df_cleaned["DATE"] + " " + df_cleaned["TIME"], errors='coerce')
        df_cleaned.drop(columns=["DATE", "TIME"], inplace=True)
    elif "DATE / TIME" in df_cleaned.columns:
        # Option 2: Combined DATE / TIME column
        df_cleaned['DATETIME'] = pd.to_datetime(df_cleaned['DATE / TIME'], errors='coerce')
        df_cleaned.drop(columns=['DATE / TIME'], inplace=True)
    
    # Convert numeric columns
    numeric_cols = ['VOLTAGE', 'CURRENT', 'POWER', 'ENERGY (kWh)']
    for col in numeric_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # Drop rows with missing values in important columns
    df_cleaned = df_cleaned.dropna(subset=['DATETIME'] + [col for col in numeric_cols if col in df_cleaned.columns])
    
    # Sort by datetime
    df_cleaned = df_cleaned.sort_values('DATETIME')
    
    return df_cleaned

# Function to create realtime monitoring graphs
def create_monitoring_graphs(df):
    # Create 3 columns for the graphs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'VOLTAGE' in df.columns:
            fig1 = px.line(
                df.iloc[-100:], 
                x="DATETIME", 
                y="VOLTAGE", 
                title="Voltage (Last 100 readings)", 
                labels={"VOLTAGE": "Voltage (V)", "DATETIME": "Time"}
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if 'CURRENT' in df.columns:
            fig2 = px.line(
                df.iloc[-100:], 
                x="DATETIME", 
                y="CURRENT", 
                title="Current (Last 100 readings)", 
                labels={"CURRENT": "Current (A)", "DATETIME": "Time"}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        if 'POWER' in df.columns:
            fig3 = px.line(
                df.iloc[-100:], 
                x="DATETIME", 
                y="POWER", 
                title="Power (Last 100 readings)", 
                labels={"POWER": "Power (W)", "DATETIME": "Time"}
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
    
    # Energy graph in a full width column
    if 'ENERGY (kWh)' in df.columns:
        # Calculate cumulative energy
        df_energy = df.copy()
        df_energy = df_energy.sort_values('DATETIME')
        df_energy["CUMULATIVE_ENERGY"] = df_energy["ENERGY (kWh)"].cumsum()
        
        fig4 = px.line(
            df_energy.iloc[-500:], 
            x="DATETIME", 
            y="CUMULATIVE_ENERGY", 
            title="Cumulative Energy Consumption (Last 500 readings)", 
            labels={"CUMULATIVE_ENERGY": "Energy (kWh)", "DATETIME": "Time"}
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

# Function to load the LSTM model
@st.cache_resource
def load_lstm_model():
    try:
        # Try to load the model from the local file
        model = load_model("lstm_energy_forecast_model.h5")
        return model
    except:
        st.warning("Could not load the LSTM model. Time series forecasting will not be available.")
        return None

# Function to prepare data for LSTM prediction
def prepare_data_for_lstm(df, time_steps=10):
    if 'ENERGY (kWh)' not in df.columns or len(df) < time_steps:
        return None, None
    
    # Get the energy data
    energy_data = df['ENERGY (kWh)'].values
    
    # Normalize data
    scaler = MinMaxScaler()
    energy_data_scaled = scaler.fit_transform(energy_data.reshape(-1, 1))
    
    # Get the last time_steps values for prediction
    X = energy_data_scaled[-time_steps:].reshape(1, time_steps, 1)
    
    return X, scaler

# Function to make predictions with LSTM model
def predict_with_lstm(model, df):
    if model is None or 'ENERGY (kWh)' not in df.columns:
        st.warning("LSTM model or energy data is not available for prediction.")
        return
    
    st.subheader("Energy Consumption Time Series Analysis")
    
    # Prepare data for prediction
    time_steps = 10
    X, scaler = prepare_data_for_lstm(df, time_steps)
    
    if X is None:
        st.warning("Not enough data points for prediction.")
        return
    
    # Make prediction
    prediction_scaled = model.predict(X)
    prediction = scaler.inverse_transform(prediction_scaled)
    
    # Get the last known datetime and add forecast points
    last_datetime = df['DATETIME'].iloc[-1]
    future_datetimes = [last_datetime + timedelta(minutes=10*i) for i in range(1, 25)]
    
    # Create forecast for the next 24 steps (4 hours if data is 10-min intervals)
    X_future = X.copy()
    future_predictions = []
    
    for _ in range(24):
        # Get prediction for next step
        next_pred = model.predict(X_future)
        future_predictions.append(next_pred[0, 0])
        
        # Update X_future by dropping the first value and adding the prediction
        X_future = np.append(X_future[:, 1:, :], [[next_pred[0, 0]]], axis=1)
    
    # Inverse transform the predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Create a dataframe for plotting
    historical_df = df.iloc[-100:].copy()
    future_df = pd.DataFrame({
        'DATETIME': future_datetimes,
        'ENERGY (kWh)': future_predictions.flatten()
    })
    
    # Create the plot
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_df['DATETIME'], 
        y=historical_df['ENERGY (kWh)'],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=future_df['DATETIME'], 
        y=future_df['ENERGY (kWh)'],
        mode='lines',
        name='Forecast (Next 4 hours)',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Energy Consumption Forecast",
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics
    st.subheader("Energy Consumption Metrics")
    col1, col2, col3 = st.columns(3)
    
    # Current hour consumption
    with col1:
        current_hour = df[df['DATETIME'] >= (df['DATETIME'].max() - timedelta(hours=1))]
        current_hour_consumption = current_hour['ENERGY (kWh)'].sum()
        st.metric("Last Hour Consumption", f"{current_hour_consumption:.2f} kWh")
    
    # Current day consumption
    with col2:
        current_day = df[df['DATETIME'].dt.date == df['DATETIME'].max().date()]
        current_day_consumption = current_day['ENERGY (kWh)'].sum()
        st.metric("Today's Consumption", f"{current_day_consumption:.2f} kWh")
    
    # Predicted consumption
    with col3:
        predicted_consumption = future_predictions.sum()
        st.metric("Predicted Next 4 Hours", f"{predicted_consumption[0]:.2f} kWh")

# Main function to run the app
def main():
    # Load LSTM model
    model = load_lstm_model()
    
    # Initialize session state for auto-refresh
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
        st.session_state.update_count = 0
    
    # Sidebar for controls
    st.sidebar.title("Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 10 minutes", value=True)
    
    if st.sidebar.button("Refresh Now"):
        st.session_state.last_update = datetime.now()
        st.session_state.update_count += 1
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    
    # Show last refresh time
    st.sidebar.write(f"Last refreshed: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.write(f"Total refreshes: {st.session_state.update_count}")
    
    # Auto-refresh logic
    if auto_refresh:
        current_time = datetime.now()
        time_diff = (current_time - st.session_state.last_update).total_seconds() / 60
        minutes_to_refresh = max(0, 10 - time_diff)
        
        st.sidebar.write(f"Next refresh in: {int(minutes_to_refresh)} minutes")
        
        if time_diff >= 10:
            st.session_state.last_update = current_time
            st.session_state.update_count += 1
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    
    # Load data
    with st.spinner("Loading energy monitoring data..."):
        try:
            # Get Google client and load data
            connection_status.info("Connecting to Google Sheets...")
            client = get_google_client()
            df = load_data(client)
            connection_status.success("Connected to Google Sheets successfully!")
            
            # Display summary info
            st.sidebar.subheader("Data Summary")
            st.sidebar.write(f"Total records: {len(df)}")
            st.sidebar.write(f"Date range: {df['DATETIME'].min().date()} to {df['DATETIME'].max().date()}")
            
            # Create monitoring graphs
            create_monitoring_graphs(df)
            
            # Make predictions with LSTM model
            predict_with_lstm(model, df)
        
        except Exception as e:
            connection_status.error(f"Error connecting to data source: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check your connection and credentials.")

if __name__ == "__main__":
    main()
