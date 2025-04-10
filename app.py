import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta
import time
import gspread
from google.oauth2.service_account import Credentials
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json
import os

# Set page config
st.set_page_config(
    page_title="Realtime Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chart-container {
        margin-top: 2rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Display main header
st.markdown("<h1 class='main-header'>⚡ Realtime Energy Monitoring Dashboard</h1>", unsafe_allow_html=True)

# Function to authenticate with Google Sheets
@st.cache_resource
def get_google_sheet_client():
    # For Streamlit Cloud, we'll use secrets
    if os.path.exists(".streamlit/secrets.toml"):
        # Local development using secrets.toml
        credentials = st.secrets["gcp_service_account"]
        credentials_dict = json.loads(credentials)
    else:
        # For GitHub deployment, get from environment variable
        credentials_dict = json.loads(st.secrets["gcp_credentials"])
    
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    creds = Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)
    return gspread.authorize(creds)

# Function to load data from Google Sheets
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_data():
    client = get_google_sheet_client()
    SHEET_ID = st.secrets["sheet_id"]  # Get sheet ID from secrets
    
    try:
        # Try to get the cleaned data worksheet
        sheet = client.open_by_key(SHEET_ID).worksheet("cleaned_data")
    except:
        # If cleaned_data doesn't exist, use the first sheet
        sheet = client.open_by_key(SHEET_ID).sheet1
    
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    # Convert DATE / TIME to datetime
    if 'DATE / TIME' in df.columns:
        df['DATE / TIME'] = pd.to_datetime(df['DATE / TIME'])
    elif 'DATETIME' in df.columns:
        df['DATE / TIME'] = pd.to_datetime(df['DATETIME'])
        df.drop(columns=['DATETIME'], inplace=True, errors='ignore')
    elif 'DATE' in df.columns and 'TIME' in df.columns:
        df['DATE / TIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
        df.drop(columns=['DATE', 'TIME'], inplace=True, errors='ignore')
    
    # Convert numeric columns
    numeric_cols = ['VOLTAGE', 'CURRENT', 'POWER', 'ENERGY (kWh)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by time
    df = df.sort_values('DATE / TIME')
    
    return df

# Load the ML model (we'll mock this for now)
@st.cache_resource
def load_ml_model():
    try:
        # In a real scenario, you would load your saved model
        # model = load_model("lstm_energy_forecast_model.h5")
        # For demo, we'll return None and handle predictions differently
        return "Model would be loaded here"
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to make predictions (mocked)
def predict_energy(df, time_steps=10):
    if len(df) < time_steps + 1:
        return np.array([]), np.array([])
    
    # In a real scenario with a loaded model:
    # 1. Normalize data
    # 2. Create sequences
    # 3. Use model.predict()
    
    # For demo, we'll generate mock predictions close to actual values
    last_points = df['ENERGY (kWh)'].iloc[-24:].values
    random_factor = np.random.normal(1, 0.05, size=len(last_points))
    predictions = last_points * random_factor
    
    # Return actual and predicted values
    return last_points, predictions

# Sidebar content
with st.sidebar:
    st.title("Dashboard Controls")
    
    # Refresh rate selection
    refresh_rate = st.slider(
        "Data Refresh Rate (seconds)", 
        min_value=10, 
        max_value=600, 
        value=60,
        step=10
    )
    
    # Time window selection
    time_window = st.selectbox(
        "Data Time Window",
        options=["Last Hour", "Last 6 Hours", "Last 12 Hours", "Last Day", "Last Week", "All Data"],
        index=3
    )
    
    # Additional controls
    st.subheader("Visualization Options")
    show_voltage = st.checkbox("Show Voltage", value=True)
    show_current = st.checkbox("Show Current", value=True)
    show_power = st.checkbox("Show Power", value=True)
    show_energy = st.checkbox("Show Energy", value=True)
    show_predictions = st.checkbox("Show Predictions", value=True)
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This dashboard displays real-time energy monitoring data from a Google Sheet. "
        "The data is refreshed automatically based on the selected refresh rate."
    )

# Create placeholder for refresh info
refresh_placeholder = st.empty()

# Function to filter data based on time window
def filter_by_time_window(df, window):
    now = datetime.now()
    if window == "Last Hour":
        start_time = now - timedelta(hours=1)
    elif window == "Last 6 Hours":
        start_time = now - timedelta(hours=6)
    elif window == "Last 12 Hours":
        start_time = now - timedelta(hours=12)
    elif window == "Last Day":
        start_time = now - timedelta(days=1)
    elif window == "Last Week":
        start_time = now - timedelta(days=7)
    else:  # All Data
        return df
    
    return df[df['DATE / TIME'] >= start_time]

# Main dashboard content
def update_dashboard():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    refresh_placeholder.info(f"Last updated: {current_time}")
    
    # Load and prepare data
    try:
        df = load_data()
        if df.empty:
            st.warning("No data available. Please check your Google Sheet.")
            return
        
        # Filter data by time window
        filtered_df = filter_by_time_window(df, time_window)
        if filtered_df.empty:
            st.warning(f"No data available for the selected time window: {time_window}")
            return
            
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric(
                "Current Voltage", 
                f"{filtered_df['VOLTAGE'].iloc[-1]:.2f} V",
                f"{filtered_df['VOLTAGE'].iloc[-1] - filtered_df['VOLTAGE'].iloc[-2]:.2f} V"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric(
                "Current Amperage", 
                f"{filtered_df['CURRENT'].iloc[-1]:.2f} A",
                f"{filtered_df['CURRENT'].iloc[-1] - filtered_df['CURRENT'].iloc[-2]:.2f} A"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric(
                "Current Power", 
                f"{filtered_df['POWER'].iloc[-1]:.2f} W",
                f"{filtered_df['POWER'].iloc[-1] - filtered_df['POWER'].iloc[-2]:.2f} W"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            # Calculate cumulative energy for the day
            today = datetime.now().date()
            today_data = filtered_df[pd.to_datetime(filtered_df['DATE / TIME']).dt.date == today]
            daily_energy = today_data['ENERGY (kWh)'].sum() if not today_data.empty else 0
            
            st.metric(
                "Today's Energy", 
                f"{daily_energy:.2f} kWh"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display charts based on user selections
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        if show_voltage:
            st.subheader("📈 Voltage Over Time")
            fig_voltage = px.line(
                filtered_df, 
                x="DATE / TIME", 
                y="VOLTAGE", 
                title="Voltage Trend",
                labels={"VOLTAGE": "Voltage (V)", "DATE / TIME": "Time"}
            )
            fig_voltage.update_layout(height=400)
            st.plotly_chart(fig_voltage, use_container_width=True)
        
        if show_current:
            st.subheader("📈 Current Over Time")
            fig_current = px.line(
                filtered_df, 
                x="DATE / TIME", 
                y="CURRENT", 
                title="Current Trend",
                labels={"CURRENT": "Current (A)", "DATE / TIME": "Time"}
            )
            fig_current.update_layout(height=400)
            st.plotly_chart(fig_current, use_container_width=True)
        
        if show_power:
            st.subheader("📈 Power Over Time")
            fig_power = px.line(
                filtered_df, 
                x="DATE / TIME", 
                y="POWER", 
                title="Power Consumption Trend",
                labels={"POWER": "Power (W)", "DATE / TIME": "Time"}
            )
            fig_power.update_layout(height=400)
            st.plotly_chart(fig_power, use_container_width=True)
        
        if show_energy:
            st.subheader("📈 Energy Consumption")
            # Calculate cumulative energy
            filtered_df['CUMULATIVE_ENERGY'] = filtered_df['ENERGY (kWh)'].cumsum()
            
            fig_energy = px.line(
                filtered_df, 
                x="DATE / TIME", 
                y="CUMULATIVE_ENERGY", 
                title="Cumulative Energy Consumption",
                labels={"CUMULATIVE_ENERGY": "Energy (kWh)", "DATE / TIME": "Time"}
            )
            fig_energy.update_layout(height=400)
            st.plotly_chart(fig_energy, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Time Series Analysis (if enabled)
        if show_predictions:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.subheader("🔮 Energy Consumption Prediction")
            
            # Get predictions
            actual_values, predicted_values = predict_energy(filtered_df)
            if len(actual_values) > 0:
                # Create dates for prediction visualization
                dates = filtered_df['DATE / TIME'].iloc[-len(actual_values):].reset_index(drop=True)
                
                # Create figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, 
                    y=actual_values, 
                    name='Actual', 
                    mode='lines',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=dates, 
                    y=predicted_values, 
                    name='Predicted', 
                    mode='lines',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title="Energy Consumption: Actual vs Predicted",
                    xaxis_title="Time",
                    yaxis_title="Energy (kWh)",
                    legend=dict(x=0, y=1),
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction accuracy metrics
                mse = np.mean((actual_values - predicted_values) ** 2)
                mae = np.mean(np.abs(actual_values - predicted_values))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                with col2:
                    st.metric("Mean Absolute Error", f"{mae:.4f}")
            else:
                st.warning("Not enough data for predictions.")
            st.markdown("</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error updating dashboard: {e}")

# Load model at startup
model = load_ml_model()

# Auto-refresh mechanism
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0

# Initial update
update_dashboard()

# Auto-refresh using an empty container and script to reload
refresh_container = st.empty()
with refresh_container.container():
    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    if auto_refresh:
        st.write(f"Auto-refreshing every {refresh_rate} seconds. Current count: {st.session_state.refresh_count}")
        time.sleep(refresh_rate)
        st.session_state.refresh_count += 1
        st.experimental_rerun()
