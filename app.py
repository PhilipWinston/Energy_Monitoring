import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, date, timedelta
import time
import json
import os
import warnings

# Filter TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

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
def get_google_client():
    try:
        # For Streamlit Cloud deployment - use secrets directly
        service_account_info = json.loads(st.secrets["google_credentials"])
        
        SCOPES = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
        client = gspread.authorize(creds)
        
        return client
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        st.write("Please check your secrets.toml file format. Make sure it's properly formatted JSON.")
        return None

# Function to load data
def load_data(client):
    if client is None:
        return None
    
    try:
        SHEET_ID = st.secrets["sheet_id"]
        sheet = client.open_by_key(SHEET_ID).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Clean and process data
        df_cleaned = clean_data(df)
        
        return df_cleaned
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None

# Function to clean data
def clean_data(df):
    try:
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
        
        # Remove outliers using IQR for each numeric column
        for col in numeric_cols:
            if col in df_cleaned.columns:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
        # Remove zero or negative values for POWER and ENERGY
        if 'POWER' in df_cleaned.columns and 'ENERGY (kWh)' in df_cleaned.columns:
            df_cleaned = df_cleaned[(df_cleaned['POWER'] > 0) & (df_cleaned['ENERGY (kWh)'] > 0)]
        
        # Sort by datetime
        df_cleaned = df_cleaned.sort_values('DATETIME')
        
        return df_cleaned
    except Exception as e:
        st.error(f"Data cleaning error: {str(e)}")
        return df  # Return original data if cleaning fails

# Function to create realtime monitoring graphs
def create_monitoring_graphs(df):
    if df is None or len(df) == 0:
        st.warning("No data available to display")
        return
    
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

# Function to train and save LSTM model
def train_lstm_model(df):
    try:
        st.info("LSTM Model not found. Training a new model...")
        
        # Import necessary libraries for model training
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
        
        # Extract energy data
        energy_data = df[['DATETIME', 'ENERGY (kWh)']].copy()
        energy_data = energy_data.sort_values('DATETIME')
        
        # Normalize data
        scaler = MinMaxScaler()
        energy_scaled = scaler.fit_transform(energy_data['ENERGY (kWh)'].values.reshape(-1, 1))
        
        # Create sequences
        def create_sequences(data, time_steps=10):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:i + time_steps])
                y.append(data[i + time_steps])
            return np.array(X), np.array(y)
        
        time_steps = 10
        X, y = create_sequences(energy_scaled, time_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM expects 3D input
        
        # Train-test split
        split_idx = int(0.8 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        # Build model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(time_steps, 1)),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Display training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Custom callback to update progress
        class StreamlitCallback:
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
                
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.total_epochs
                progress_bar.progress(progress)
                status_text.text(f"Training LSTM: Epoch {epoch+1}/{total_epochs}, Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")
                time.sleep(0.1)  # Small delay to show progress
        
        total_epochs = 20
        streamlit_callback = StreamlitCallback(total_epochs)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=total_epochs,
            batch_size=32,
            callbacks=[early_stop, streamlit_callback],
            verbose=0
        )
        
        # Save model
        model.save("lstm_energy_forecast_model.h5")
        status_text.text("Model training complete! Model saved as 'lstm_energy_forecast_model.h5'")
        progress_bar.progress(1.0)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # MSE and R2 score calculation
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        st.write(f"Model Evaluation - Mean Squared Error: {mse:.4f}, R² Score: {r2:.4f}")
        
        # Plot training history
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Training Loss'
        ))
        fig.add_trace(go.Scatter(
            y=history.history['val_loss'],
            name='Validation Loss'
        ))
        fig.update_layout(
            title="Model Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return model
    except Exception as e:
        st.error(f"Error training LSTM model: {str(e)}")
        return None

# Function to load the LSTM model
@st.cache_resource
def load_lstm_model(df):
    try:
        # Try to load the model
        from tensorflow.keras.models import load_model
        model = load_model("lstm_energy_forecast_model.h5")
        return model
    except Exception as e:
        st.warning(f"Could not load the LSTM model: {str(e)}")
        # Train a new model if file not found
        return train_lstm_model(df)

# Function to prepare data for LSTM prediction
def prepare_data_for_lstm(df, time_steps=10):
    if df is None or 'ENERGY (kWh)' not in df.columns or len(df) < time_steps:
        return None, None
    
    try:
        # Import in function to avoid initial TF errors
        from sklearn.preprocessing import MinMaxScaler
        
        # Get the energy data
        energy_data = df['ENERGY (kWh)'].values
        
        # Normalize data
        scaler = MinMaxScaler()
        energy_data_scaled = scaler.fit_transform(energy_data.reshape(-1, 1))
        
        # Get the last time_steps values for prediction
        X = energy_data_scaled[-time_steps:].reshape(1, time_steps, 1)
        
        return X, scaler
    except Exception as e:
        st.warning(f"Error preparing data for prediction: {str(e)}")
        return None, None

# Function to make predictions with LSTM model
def predict_with_lstm(model, df):
    if model is None or df is None or 'ENERGY (kWh)' not in df.columns:
        st.warning("LSTM model or energy data is not available for prediction.")
        return
    
    try:
        # Import numpy only when needed
        import numpy as np
        
        st.subheader("Energy Consumption Time Series Analysis")
        
        # Prepare data for prediction
        time_steps = 10
        X, scaler = prepare_data_for_lstm(df, time_steps)
        
        if X is None:
            st.warning("Not enough data points for prediction.")
            return
        
        # Make prediction
        with st.spinner("Making predictions..."):
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
    
    except Exception as e:
        st.error(f"Error in time series prediction: {str(e)}")

# Main function to run the app
def main():
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
        
        refresh_progress = st.sidebar.progress(0)
        refresh_status = st.sidebar.empty()
        refresh_status.write(f"Next refresh in: {int(minutes_to_refresh)} minutes")
        
        # Update progress bar
        progress_value = min(1.0, time_diff / 10)
        refresh_progress.progress(progress_value)
        
        if time_diff >= 10:
            st.session_state.last_update = current_time
            st.session_state.update_count += 1
            refresh_status.write("Refreshing now...")
            st.rerun()
    
    # Load data
    with st.spinner("Loading energy monitoring data..."):
        try:
            # Get Google client and load data
            connection_status.info("Connecting to Google Sheets...")
            client = get_google_client()
            
            if client:
                df = load_data(client)
                if df is not None and not df.empty:
                    connection_status.success("Connected to Google Sheets successfully!")
                    
                    # Display summary info
                    st.sidebar.subheader("Data Summary")
                    st.sidebar.write(f"Total records: {len(df)}")
                    st.sidebar.write(f"Date range: {df['DATETIME'].min().date()} to {df['DATETIME'].max().date()}")
                    
                    # Add data quality checks
                    st.sidebar.subheader("Data Quality")
                    if 'VOLTAGE' in df.columns:
                        voltage_ok = (df['VOLTAGE'].mean() > 200) and (df['VOLTAGE'].mean() < 250)
                        st.sidebar.write(f"Voltage quality: {'✅' if voltage_ok else '⚠️'}")
                    
                    if 'CURRENT' in df.columns:
                        current_ok = df['CURRENT'].max() < 20  # Assuming max 20A for safety
                        st.sidebar.write(f"Current quality: {'✅' if current_ok else '⚠️'}")
                    
                    # Create monitoring graphs
                    create_monitoring_graphs(df)
                    
                    # Add data download option
                    csv = df.to_csv(index=False)
                    st.sidebar.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name="energy_data.csv",
                        mime="text/csv",
                    )
                    
                    # Load model and make predictions
                    model = load_lstm_model(df)
                    predict_with_lstm(model, df)
                else:
                    connection_status.error("Connected to Google Sheets but no valid data found")
            else:
                connection_status.error("Failed to connect to Google Sheets")
        
        except Exception as e:
            connection_status.error(f"Error connecting to data source: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check your connection and credentials.")

if __name__ == "__main__":
    main()
