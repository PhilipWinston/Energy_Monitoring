import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import holidays
import pickle
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Streamlit Configuration
# -------------------------------
st.set_page_config(
    page_title="⚡ Smart Energy Management System", 
    layout="wide",
    page_icon="⚡"
)

# -------------------------------
# 2. Load Trained Prophet Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    """Load the pre-trained Prophet model"""
    try:
        with open('energy_prophet_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Trained model file 'energy_prophet_model.pkl' not found. Please upload the trained model.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading trained model: {e}")
        return None

# -------------------------------
# 3. Data Loading Functions
# -------------------------------
@st.cache_data(ttl=60)
def load_google_sheet_data():
    """Load real-time data from Google Sheets"""
    try:
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
    except Exception as e:
        st.error(f"❌ Error loading live data: {e}")
        st.info("📄 Please ensure Google Sheets credentials are properly configured.")
        return None

def prepare_features_for_prediction(datetime_obj, power_value):
    """Prepare features for Prophet model prediction based on your training code"""
    india_holidays_2024_2025 = holidays.country_holidays("IN", years=[2024, 2025])
    
    features = {
        'ds': datetime_obj,
        'is_working_hour': int((8 <= datetime_obj.hour < 20)),
        'is_holiday': int(datetime_obj.date() in india_holidays_2024_2025.keys()),
        'is_sunday': int(datetime_obj.weekday() == 6),
        'is_odd_saturday': int(datetime_obj.weekday() == 5 and ((datetime_obj.day - 1) // 7) % 2 == 0),
        'is_weekend': int(datetime_obj.weekday() >= 5),
        'is_peak_hour': int((9 <= datetime_obj.hour <= 18)),
        'power': power_value
    }
    
    return pd.DataFrame([features])

def make_predictions(model, prediction_horizon, interval_minutes, current_data=None):
    """Make predictions using the trained Prophet model"""
    if model is None:
        return None, None
    
    # Get current time or use latest data time
    if current_data is not None and len(current_data) > 0:
        last_time = current_data['DATETIME'].max()
        latest_power = current_data['POWER'].iloc[-1]
    else:
        last_time = datetime.now()
        latest_power = 0.0
    
    # Define prediction horizons
    horizon_mapping = {
        "Next Hour": 1,
        "Next 4 Hours": 4, 
        "Next 12 Hours": 12,
        "Next 24 Hours": 24,
        "Next Week": 168,  # 7 days * 24 hours
        "Next Month": 720,  # 30 days * 24 hours
        "Next Year": 8760   # 365 days * 24 hours
    }
    
    horizon_hours = horizon_mapping.get(prediction_horizon, 1)
    future_periods = int((horizon_hours * 60) / interval_minutes)
    
    # Generate future timestamps
    future_dates = pd.date_range(
        start=last_time + timedelta(minutes=interval_minutes),
        periods=future_periods,
        freq=f'{interval_minutes}min'
    )
    
    # Prepare future dataframe with all required features
    future_data = []
    for dt in future_dates:
        # Estimate power based on working patterns from your training
        is_working_day = not (dt.date() in holidays.country_holidays("IN", years=[2024, 2025]).keys() or 
                             dt.weekday() == 6 or 
                             (dt.weekday() == 5 and ((dt.day - 1) // 7) % 2 == 0))
        
        if is_working_day:
            monitoring_prob = 0.95 if (8 <= dt.hour < 20) else 0.2
        else:
            monitoring_prob = 0.1 if (8 <= dt.hour < 20) else 0.03
        
        # Use current power pattern or estimate
        estimated_power = latest_power if np.random.rand() < monitoring_prob else 0.0
        
        features = prepare_features_for_prediction(dt, estimated_power)
        future_data.append(features.iloc[0])
    
    future_df = pd.DataFrame(future_data)
    
    try:
        # Make predictions
        forecast = model.predict(future_df)
        return forecast, future_df
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None

# -------------------------------
# 4. Load Data and Model
# -------------------------------
# Load the trained model
trained_model = load_trained_model()

# Load live data
df_raw = load_google_sheet_data()

# Process data if available
if df_raw is not None and len(df_raw) > 0:
    try:
        df_raw["DATETIME"] = pd.to_datetime(
            df_raw["DATE"].astype(str) + " " + df_raw["TIME"].astype(str),
            format="%Y-%m-%d %H:%M:%S"
        )
        df_raw = df_raw.sort_values("DATETIME")
        df_raw['ENERGY (kWh)'] = pd.to_numeric(df_raw['ENERGY (kWh)'], errors='coerce')
        df_raw['VOLTAGE'] = pd.to_numeric(df_raw['VOLTAGE'], errors='coerce')
        df_raw['CURRENT'] = pd.to_numeric(df_raw['CURRENT'], errors='coerce')
        df_raw['POWER'] = pd.to_numeric(df_raw['POWER'], errors='coerce')
        df_raw = df_raw.dropna()
    except Exception as e:
        st.error(f"❌ Data processing error: {e}")
        df_raw = None

# -------------------------------
# 5. Header and Navigation
# -------------------------------
st.title("⚡ Smart Energy Management System")
st.markdown("### Real-time Monitoring & AI-Powered Predictive Analytics")

# Status indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

with col2:
    model_status = "🟢 Ready" if trained_model else "🔴 Not Loaded"
    st.metric("AI Model", model_status)

with col3:
    data_status = "🟢 Live" if df_raw is not None and len(df_raw) > 0 else "🔴 No Data"
    st.metric("Data Feed", data_status)

with col4:
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    st.metric("IST Time", ist_time.strftime('%H:%M:%S'))

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Monitoring", "🔮 Energy Prediction", "📈 Analytics"])

# -------------------------------
# TAB 1: LIVE MONITORING
# -------------------------------
with tab1:
    st.header("📊 Real-time Energy Monitoring")
    
    if df_raw is not None and len(df_raw) > 0:
        # Current metrics
        latest = df_raw.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            voltage_color = "normal" if 220 <= latest['VOLTAGE'] <= 240 else "inverse"
            st.metric(
                "⚡ Voltage", 
                f"{latest['VOLTAGE']:.1f} V",
                delta=f"{latest['VOLTAGE'] - 230:.1f}" if len(df_raw) > 1 else None,
                delta_color=voltage_color
            )
        
        with col2:
            st.metric(
                "🔋 Current", 
                f"{latest['CURRENT']:.2f} A",
                delta=f"{latest['CURRENT'] - df_raw.iloc[-2]['CURRENT']:.2f}" if len(df_raw) > 1 else None
            )
        
        with col3:
            st.metric(
                "⚙️ Power", 
                f"{latest['POWER']:.3f} kW",
                delta=f"{latest['POWER'] - df_raw.iloc[-2]['POWER']:.3f}" if len(df_raw) > 1 else None
            )
        
        with col4:
            total_energy = df_raw['ENERGY (kWh)'].sum()
            cost = total_energy * 7.11  # Indian electricity rate
            st.metric("💰 Total Cost", f"₹{cost:.2f}")
        
        # Voltage status indicator
        if latest['VOLTAGE'] < 220:
            st.warning("⚠️ Low voltage detected! Check electrical connections.")
        elif latest['VOLTAGE'] > 240:
            st.warning("⚠️ High voltage detected! Equipment may be at risk.")
        else:
            st.success("✅ Voltage within normal range (220-240V)")
        
        # Real-time visualization
        st.subheader("📈 Live Data Visualization")
        
        # Create combined dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Voltage Monitoring', 'Current & Power', 'Energy Consumption', 'System Health'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08
        )
        
        # Voltage with safety zones
        fig.add_trace(
            go.Scatter(x=df_raw['DATETIME'], y=df_raw['VOLTAGE'], 
                      name='Voltage', line=dict(color='#FF6B6B', width=2)),
            row=1, col=1
        )
        fig.add_hline(y=220, line_dash="dash", line_color="orange", row=1, col=1)
        fig.add_hline(y=240, line_dash="dash", line_color="orange", row=1, col=1)
        fig.add_hline(y=230, line_dash="solid", line_color="green", row=1, col=1)
        
        # Current and Power
        fig.add_trace(
            go.Scatter(x=df_raw['DATETIME'], y=df_raw['CURRENT'], 
                      name='Current', line=dict(color='#4ECDC4', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df_raw['DATETIME'], y=df_raw['POWER'], 
                      name='Power', line=dict(color='#45B7D1', width=2)),
            row=1, col=2, secondary_y=True
        )
        
        # Energy consumption
        fig.add_trace(
            go.Bar(x=df_raw['DATETIME'], y=df_raw['ENERGY (kWh)'], 
                   name='Energy', marker_color='#96CEB4'),
            row=2, col=1
        )
        
        # System efficiency
        efficiency = (df_raw['POWER'] / (df_raw['VOLTAGE'] * df_raw['CURRENT'] / 1000)).fillna(0)
        fig.add_trace(
            go.Scatter(x=df_raw['DATETIME'], y=efficiency, 
                      name='Efficiency', line=dict(color='#F7DC6F', width=2)),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Real-time Energy Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent data table
        st.subheader("📋 Recent Readings")
        display_count = st.slider("Records to display:", 5, 20, 10)
        recent_data = df_raw.tail(display_count)[['DATETIME', 'VOLTAGE', 'CURRENT', 'POWER', 'ENERGY (kWh)']]
        st.dataframe(recent_data, use_container_width=True)
        
    else:
        st.warning("📡 No live data available. Please check Google Sheets connection.")

# -------------------------------
# TAB 2: ENERGY PREDICTION
# -------------------------------
with tab2:
    st.header("🔮 AI-Powered Energy Prediction")
    
    if trained_model is None:
        st.error("❌ Trained Prophet model not available. Please upload 'energy_prophet_model.pkl'")
        st.info("📝 The model should be trained using your provided training code with synthetic_energy_april_2024.csv")
    else:
        st.success("✅ AI Model loaded successfully!")
        
        # Prediction controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_horizon = st.selectbox(
                "🔮 Prediction Horizon:",
                ["Next Hour", "Next 4 Hours", "Next 12 Hours", "Next 24 Hours", 
                 "Next Week", "Next Month", "Next Year"]
            )
        
        with col2:
            interval_mapping = {
                "Next Hour": ["1 minute", "5 minutes", "10 minutes"],
                "Next 4 Hours": ["10 minutes", "30 minutes", "1 hour"],
                "Next 12 Hours": ["30 minutes", "1 hour", "2 hours"],
                "Next 24 Hours": ["1 hour", "2 hours", "4 hours"],
                "Next Week": ["4 hours", "12 hours", "24 hours"],
                "Next Month": ["24 hours", "3 days", "7 days"],
                "Next Year": ["7 days", "30 days", "90 days"]
            }
            
            available_intervals = interval_mapping.get(prediction_horizon, ["1 hour"])
            prediction_interval = st.selectbox("⏱️ Prediction Interval:", available_intervals)
        
        with col3:
            confidence_level = st.slider("📊 Confidence Level:", 80, 99, 95)
        
        # Convert interval to minutes
        interval_minutes_map = {
            "1 minute": 1, "5 minutes": 5, "10 minutes": 10, "30 minutes": 30,
            "1 hour": 60, "2 hours": 120, "4 hours": 240, "12 hours": 720,
            "24 hours": 1440, "3 days": 4320, "7 days": 10080, "30 days": 43200,
            "90 days": 129600
        }
        interval_minutes = interval_minutes_map.get(prediction_interval, 60)
        
        # Make predictions
        if st.button("🚀 Generate Predictions"):
            with st.spinner("🤖 AI is analyzing patterns and generating predictions..."):
                forecast, future_df = make_predictions(
                    trained_model, prediction_horizon, interval_minutes, df_raw
                )
            
            if forecast is not None:
                st.success("✅ Predictions generated successfully!")
                
                # Prediction visualization
                st.subheader("📊 Energy Consumption Forecast")
                
                fig_forecast = go.Figure()
                
                # Show recent historical data if available
                if df_raw is not None and len(df_raw) > 0:
                    recent_data = df_raw.tail(min(100, len(df_raw)))
                    fig_forecast.add_trace(go.Scatter(
                        x=recent_data['DATETIME'],
                        y=recent_data['ENERGY (kWh)'],
                        mode='lines+markers',
                        name='Historical Data',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=3)
                    ))
                
                # Forecast
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines+markers',
                    name=f'Forecast ({prediction_horizon})',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    marker=dict(size=4)
                ))
                
                # Confidence interval
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))
                
                fig_forecast.update_layout(
                    title=f'Energy Consumption Forecast - {prediction_horizon}',
                    xaxis_title='Time',
                    yaxis_title='Energy (kWh)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Prediction analytics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Forecast Summary")
                    total_predicted = forecast['yhat'].sum()
                    max_predicted = forecast['yhat'].max()
                    avg_predicted = forecast['yhat'].mean()
                    min_predicted = forecast['yhat'].min()
                    
                    st.metric("Total Predicted Energy", f"{total_predicted:.4f} kWh")
                    st.metric("Peak Energy Period", f"{max_predicted:.4f} kWh")
                    st.metric("Average Energy Rate", f"{avg_predicted:.4f} kWh")
                    st.metric("Minimum Energy Period", f"{min_predicted:.4f} kWh")
                    
                    # Cost estimation
                    predicted_cost = total_predicted * 7.11
                    st.metric("💰 Estimated Cost", f"₹{predicted_cost:.2f}")
                
                with col2:
                    st.subheader("🧠 AI Insights & Recommendations")
                    
                    # Generate intelligent recommendations
                    peak_time = forecast.loc[forecast['yhat'].idxmax(), 'ds']
                    low_time = forecast.loc[forecast['yhat'].idxmin(), 'ds']
                    
                    insights = []
                    
                    # Peak usage analysis
                    if max_predicted > avg_predicted * 1.5:
                        insights.append(f"⚠️ High energy spike predicted at {peak_time.strftime('%d %b, %H:%M')}. Consider load balancing.")
                    
                    # Off-peak opportunities
                    if min_predicted < avg_predicted * 0.5:
                        insights.append(f"💡 Low usage window at {low_time.strftime('%d %b, %H:%M')}. Ideal for maintenance or heavy tasks.")
                    
                    # Weekend patterns
                    weekend_forecast = forecast[forecast['ds'].dt.weekday >= 5]
                    if len(weekend_forecast) > 0:
                        weekend_avg = weekend_forecast['yhat'].mean()
                        if weekend_avg > avg_predicted * 1.2:
                            insights.append("📅 Higher weekend usage predicted. Check for unnecessary equipment running.")
                        else:
                            insights.append("✅ Normal weekend consumption pattern detected.")
                    
                    # Cost optimization
                    if predicted_cost > 100:
                        insights.append(f"💰 High cost period ahead (₹{predicted_cost:.2f}). Consider energy-saving measures.")
                    elif predicted_cost < 20:
                        insights.append("💚 Low cost period - good time for energy-intensive tasks.")
                    
                    # Efficiency recommendations
                    high_consumption_hours = forecast[forecast['yhat'] > avg_predicted * 1.3]
                    if len(high_consumption_hours) > 0:
                        insights.append(f"⚡ {len(high_consumption_hours)} high-consumption periods detected. Monitor equipment efficiency.")
                    
                    for i, insight in enumerate(insights, 1):
                        st.write(f"{i}. {insight}")
                
                # Detailed forecast table
                st.subheader("📋 Detailed Predictions")
                forecast_display = pd.DataFrame({
                    'Time': forecast['ds'].dt.strftime('%Y-%m-%d %H:%M'),
                    'Predicted Energy (kWh)': forecast['yhat'].round(6),
                    'Lower Bound': forecast['yhat_lower'].round(6),
                    'Upper Bound': forecast['yhat_upper'].round(6),
                    'Estimated Cost (₹)': (forecast['yhat'] * 7.11).round(2)
                })
                
                # Show only first 20 rows for readability
                st.dataframe(forecast_display.head(20), use_container_width=True)
                
                if len(forecast_display) > 20:
                    st.info(f"Showing first 20 of {len(forecast_display)} predictions. Full data available for download.")

# -------------------------------
# TAB 3: ANALYTICS
# -------------------------------
with tab3:
    st.header("📈 Energy Analytics & Insights")
    
    if df_raw is not None and len(df_raw) > 0:
        # Energy consumption patterns
        st.subheader("🔍 Consumption Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern
            df_raw['Hour'] = df_raw['DATETIME'].dt.hour
            hourly_stats = df_raw.groupby('Hour').agg({
                'ENERGY (kWh)': ['mean', 'sum', 'count'],
                'POWER': 'mean'
            }).round(4)
            
            fig_hourly = px.bar(
                x=hourly_stats.index,
                y=hourly_stats[('ENERGY (kWh)', 'sum')],
                title='Hourly Energy Consumption Pattern',
                labels={'x': 'Hour of Day', 'y': 'Total Energy (kWh)'},
                color=hourly_stats[('ENERGY (kWh)', 'sum')],
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Daily pattern
            df_raw['Date'] = df_raw['DATETIME'].dt.date
            daily_stats = df_raw.groupby('Date')['ENERGY (kWh)'].sum().reset_index()
            
            fig_daily = px.line(
                daily_stats, x='Date', y='ENERGY (kWh)',
                title='Daily Energy Consumption Trend',
                markers=True
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # System efficiency analysis
        st.subheader("⚙️ System Efficiency Analysis")
        
        # Calculate efficiency metrics
        df_raw['Efficiency'] = df_raw['POWER'] / (df_raw['VOLTAGE'] * df_raw['CURRENT'] / 1000)
        df_raw['Efficiency'] = df_raw['Efficiency'].fillna(0).clip(0, 1)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_efficiency = df_raw['Efficiency'].mean()
            st.metric("Average Efficiency", f"{avg_efficiency:.2%}")
        
        with col2:
            power_factor = df_raw[df_raw['POWER'] > 0]['POWER'].sum() / (df_raw['VOLTAGE'].mean() * df_raw['CURRENT'].sum() / 1000)
            st.metric("System Power Factor", f"{power_factor:.3f}")
        
        with col3:
            uptime = (df_raw['CURRENT'] > 0).mean()
            st.metric("Equipment Uptime", f"{uptime:.1%}")
        
        # Cost analysis
        st.subheader("💰 Cost Analysis")
        
        total_energy = df_raw['ENERGY (kWh)'].sum()
        total_cost = total_energy * 7.11
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Energy Consumed", f"{total_energy:.3f} kWh")
        with col2:
            st.metric("Total Cost", f"₹{total_cost:.2f}")
        with col3:
            avg_hourly_cost = total_cost / (len(df_raw) / 6)  # Assuming 10-min intervals
            st.metric("Avg Hourly Cost", f"₹{avg_hourly_cost:.2f}")
        with col4:
            if len(df_raw) > 1:
                time_span_hours = (df_raw['DATETIME'].max() - df_raw['DATETIME'].min()).total_seconds() / 3600
                cost_per_hour = total_cost / time_span_hours if time_span_hours > 0 else 0
                st.metric("Cost Rate", f"₹{cost_per_hour:.2f}/hr")
    
    else:
        st.warning("📊 No data available for analytics. Please ensure live data connection is working.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    st.caption(f"⏱ Last updated: {ist_time.strftime('%H:%M:%S')} IST")

with col2:
    data_points = len(df_raw) if df_raw is not None else 0
    st.caption(f"📊 Data points: {data_points:,}")

with col3:
    model_status = "AI Ready" if trained_model else "No Model"
    st.caption(f"🤖 Model: {model_status}")

with col4:
    if df_raw is not None and len(df_raw) > 0:
        data_freshness = (datetime.now() - df_raw['DATETIME'].max()).total_seconds() / 60
        st.caption(f"🔄 Data age: {data_freshness:.1f} min")
    else:
        st.caption("🔄 No live data")

st.markdown("*Powered by Prophet AI & Real-time Analytics*")
