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
from prophet import Prophet
import pickle
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Streamlit Configuration & Auto-refresh
# -------------------------------
st.set_page_config(
    page_title="⚡ Smart Energy Management System", 
    layout="wide",
    page_icon="⚡"
)

# Auto-refresh setup (uncomment when streamlit_autorefresh is available)
# from streamlit_autorefresh import st_autorefresh
# st_autorefresh(interval=60000, limit=None, key="1min_refresh")

# -------------------------------
# 2. Data Loading Functions
# -------------------------------
@st.cache_data(ttl=60)  # Cache for 1 minute
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
        st.error(f"Error loading data: {e}")
        # Return sample data for demo purposes
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data for demo purposes"""
    dates = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                         end=datetime.now(), freq='10min')
    np.random.seed(42)
    
    data = []
    for date in dates:
        voltage = 220 + np.random.normal(0, 5)
        current = 2 + np.random.normal(0, 0.5) if np.random.random() > 0.3 else 0
        power = voltage * current / 1000  # kW
        energy = power * (10/60)  # kWh for 10 minutes
        
        data.append({
            'DATE': date.strftime('%Y-%m-%d'),
            'TIME': date.strftime('%H:%M:%S'),
            'VOLTAGE': max(0, voltage),
            'CURRENT': max(0, current),
            'POWER': max(0, power),
            'ENERGY (kWh)': max(0, energy)
        })
    
    return pd.DataFrame(data)

def prepare_prophet_data(df):
    """Prepare data for Prophet model"""
    prophet_df = pd.DataFrame({
        'ds': df['DATETIME'],
        'y': df['ENERGY (kWh)']
    })
    
    # Add regressors
    prophet_df['is_working_hour'] = ((df['DATETIME'].dt.hour >= 8) & 
                                   (df['DATETIME'].dt.hour < 20)).astype(int)
    prophet_df['is_holiday'] = df['DATETIME'].dt.date.apply(
        lambda x: x in holidays.country_holidays("IN", years=[2024, 2025]).keys()
    ).astype(int)
    prophet_df['is_sunday'] = (df['DATETIME'].dt.dayofweek == 6).astype(int)
    prophet_df['is_weekend'] = ((df['DATETIME'].dt.dayofweek == 5) | 
                               (df['DATETIME'].dt.dayofweek == 6)).astype(int)
    prophet_df['is_peak_hour'] = ((df['DATETIME'].dt.hour >= 9) & 
                                 (df['DATETIME'].dt.hour <= 18)).astype(int)
    prophet_df['power'] = df['POWER'].fillna(0)
    
    return prophet_df

# -------------------------------
# 3. Load and Prepare Data
# -------------------------------
df_raw = load_google_sheet_data()

# Data preparation
try:
    df_raw["DATETIME"] = pd.to_datetime(
        df_raw["DATE"] + " " + df_raw["TIME"],
        format="%Y-%m-%d %H:%M:%S"
    )
except Exception as e:
    st.error(f"Error creating datetime column: {e}")
    st.stop()

df_raw = df_raw.sort_values("DATETIME")
df_raw['ENERGY (kWh)'] = pd.to_numeric(df_raw['ENERGY (kWh)'], errors='coerce')
df_raw = df_raw.dropna()

# -------------------------------
# 4. Header and Navigation
# -------------------------------
st.title("⚡ Smart Energy Management System")
st.markdown("### Real-time Monitoring & Predictive Analytics")

# Manual refresh button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

with col2:
    st.metric("Status", "🟢 Live", help="System is actively monitoring")

# Create tabs
tab1, tab2 = st.tabs(["📊 Live Monitoring", "🔮 Energy Prediction"])

# -------------------------------
# TAB 1: LIVE MONITORING
# -------------------------------
with tab1:
    st.header("📊 Real-time Energy Monitoring")
    
    # Current metrics row
    if len(df_raw) > 0:
        latest = df_raw.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "⚡ Current Voltage", 
                f"{latest['VOLTAGE']:.1f} V",
                delta=f"{latest['VOLTAGE'] - 220:.1f}" if len(df_raw) > 1 else None
            )
        
        with col2:
            st.metric(
                "🔋 Current Draw", 
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
            cost = total_energy * 7.11
            st.metric(
                "💰 Total Cost", 
                f"₹{cost:.2f}",
                delta=f"₹{latest['ENERGY (kWh)'] * 7.11:.2f}"
            )
    
    # Visualization options
    st.subheader("📈 Live Data Visualization")
    
    viz_option = st.selectbox(
        "Choose visualization type:",
        ["Combined View", "Individual Metrics", "Power Analysis", "24-Hour Overview"]
    )
    
    if viz_option == "Combined View":
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Voltage Over Time', 'Current Over Time', 
                          'Power Consumption', 'Energy Consumption'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Voltage
        fig.add_trace(
            go.Scatter(x=df_raw['DATETIME'], y=df_raw['VOLTAGE'], 
                      name='Voltage', line=dict(color='#FF6B6B', width=2)),
            row=1, col=1
        )
        
        # Current
        fig.add_trace(
            go.Scatter(x=df_raw['DATETIME'], y=df_raw['CURRENT'], 
                      name='Current', line=dict(color='#4ECDC4', width=2)),
            row=1, col=2
        )
        
        # Power
        fig.add_trace(
            go.Scatter(x=df_raw['DATETIME'], y=df_raw['POWER'], 
                      name='Power', line=dict(color='#45B7D1', width=2)),
            row=2, col=1
        )
        
        # Energy
        fig.add_trace(
            go.Bar(x=df_raw['DATETIME'], y=df_raw['ENERGY (kWh)'], 
                   name='Energy', marker_color='#96CEB4'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600, 
            showlegend=False,
            title_text="Real-time Energy Monitoring Dashboard",
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Individual Metrics":
        # Individual metric charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_v = px.line(df_raw, x='DATETIME', y='VOLTAGE', 
                           title='🔌 Voltage Monitoring',
                           color_discrete_sequence=['#FF6B6B'])
            fig_v.add_hline(y=220, line_dash="dash", line_color="green", 
                           annotation_text="Nominal Voltage")
            fig_v.add_hline(y=200, line_dash="dash", line_color="orange", 
                           annotation_text="Low Voltage Warning")
            st.plotly_chart(fig_v, use_container_width=True)
            
            fig_c = px.line(df_raw, x='DATETIME', y='CURRENT', 
                           title='🔋 Current Monitoring',
                           color_discrete_sequence=['#4ECDC4'])
            st.plotly_chart(fig_c, use_container_width=True)
        
        with col2:
            fig_p = px.area(df_raw, x='DATETIME', y='POWER', 
                           title='⚙️ Power Consumption',
                           color_discrete_sequence=['#45B7D1'])
            st.plotly_chart(fig_p, use_container_width=True)
            
            fig_e = px.bar(df_raw, x='DATETIME', y='ENERGY (kWh)', 
                          title='📊 Energy Usage',
                          color_discrete_sequence=['#96CEB4'])
            st.plotly_chart(fig_e, use_container_width=True)
    
    elif viz_option == "Power Analysis":
        # Power analysis with statistics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_power = go.Figure()
            fig_power.add_trace(go.Scatter(
                x=df_raw['DATETIME'], 
                y=df_raw['POWER'],
                mode='lines+markers',
                name='Power Consumption',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=4)
            ))
            
            # Add power zones
            avg_power = df_raw['POWER'].mean()
            fig_power.add_hline(y=avg_power, line_dash="dash", line_color="blue", 
                               annotation_text=f"Average: {avg_power:.3f} kW")
            fig_power.add_hline(y=avg_power * 1.5, line_dash="dash", line_color="orange", 
                               annotation_text="High Usage Zone")
            
            fig_power.update_layout(title='🔍 Detailed Power Analysis')
            st.plotly_chart(fig_power, use_container_width=True)
        
        with col2:
            st.subheader("📊 Power Statistics")
            power_stats = df_raw['POWER'].describe()
            for stat, value in power_stats.items():
                st.metric(stat.title(), f"{value:.4f} kW")
    
    elif viz_option == "24-Hour Overview":
        # 24-hour energy pattern
        df_raw['Hour'] = df_raw['DATETIME'].dt.hour
        hourly_energy = df_raw.groupby('Hour')['ENERGY (kWh)'].sum().reset_index()
        
        fig_hourly = px.bar(
            hourly_energy, x='Hour', y='ENERGY (kWh)',
            title='24-Hour Energy Consumption Pattern',
            color='ENERGY (kWh)',
            color_continuous_scale='viridis'
        )
        fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Energy (kWh)")
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Data table
    st.subheader("📋 Recent Data Log")
    display_count = st.slider("Number of recent records to display:", 5, 50, 10)
    st.dataframe(df_raw.tail(display_count), use_container_width=True)

# -------------------------------
# TAB 2: ENERGY PREDICTION
# -------------------------------
with tab2:
    st.header("🔮 Energy Prediction & Analytics")
    
    # Prediction controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_horizon = st.selectbox(
            "Prediction Horizon:",
            ["Next Hour", "Next 4 Hours", "Next 12 Hours", "Next 24 Hours"]
        )
    
    with col2:
        prediction_interval = st.selectbox(
            "Prediction Interval:",
            ["10 minutes", "30 minutes", "1 hour"]
        )
    
    with col3:
        confidence_level = st.slider("Confidence Level:", 80, 99, 95)
    
    # Create and train Prophet model (simplified for demo)
    @st.cache_resource
    def create_prophet_model(df):
        """Create and train Prophet model"""
        try:
            prophet_df = prepare_prophet_data(df)
            
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=0.1,
                holidays_prior_scale=0.1,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=confidence_level/100
            )
            
            # Add regressors
            model.add_regressor('is_working_hour', prior_scale=0.5)
            model.add_regressor('is_holiday', prior_scale=0.3)
            model.add_regressor('is_weekend', prior_scale=0.3)
            model.add_regressor('is_peak_hour', prior_scale=0.4)
            model.add_regressor('power', prior_scale=0.8)
            
            # Add holidays
            india_holidays = holidays.country_holidays("IN", years=[2024, 2025])
            holiday_df = pd.DataFrame([
                {'holiday': 'holiday', 'ds': pd.to_datetime(date), 
                 'lower_window': 0, 'upper_window': 0}
                for date in india_holidays.keys()
            ])
            model.holidays = holiday_df
            
            model.fit(prophet_df)
            return model, prophet_df
        except Exception as e:
            st.error(f"Error creating model: {e}")
            return None, None
    
    # Train model
    with st.spinner("🤖 Training prediction model..."):
        model, prophet_df = create_prophet_model(df_raw)
    
    if model is not None:
        # Generate predictions
        horizon_hours = {"Next Hour": 1, "Next 4 Hours": 4, 
                        "Next 12 Hours": 12, "Next 24 Hours": 24}[prediction_horizon]
        
        interval_minutes = {"10 minutes": 10, "30 minutes": 30, "1 hour": 60}[prediction_interval]
        
        # Create future dataframe
        last_time = df_raw['DATETIME'].max()
        future_periods = int((horizon_hours * 60) / interval_minutes)
        
        future_dates = pd.date_range(
            start=last_time + timedelta(minutes=interval_minutes),
            periods=future_periods,
            freq=f'{interval_minutes}min'
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Add regressors for future predictions
        future_df['is_working_hour'] = ((future_df['ds'].dt.hour >= 8) & 
                                       (future_df['ds'].dt.hour < 20)).astype(int)
        future_df['is_holiday'] = future_df['ds'].dt.date.apply(
            lambda x: x in holidays.country_holidays("IN", years=[2024, 2025]).keys()
        ).astype(int)
        future_df['is_weekend'] = ((future_df['ds'].dt.dayofweek == 5) | 
                                  (future_df['ds'].dt.dayofweek == 6)).astype(int)
        future_df['is_peak_hour'] = ((future_df['ds'].dt.hour >= 9) & 
                                    (future_df['ds'].dt.hour <= 18)).astype(int)
        
        # Estimate future power based on patterns
        avg_working_power = df_raw[df_raw['DATETIME'].dt.hour.between(8, 20)]['POWER'].mean()
        avg_non_working_power = df_raw[~df_raw['DATETIME'].dt.hour.between(8, 20)]['POWER'].mean()
        
        future_df['power'] = np.where(
            future_df['is_working_hour'], 
            avg_working_power, 
            avg_non_working_power
        )
        
        # Make predictions
        forecast = model.predict(future_df)
        
        # Visualization
        st.subheader("📊 Energy Consumption Forecast")
        
        # Combine historical and forecast data
        fig_forecast = go.Figure()
        
        # Historical data
        recent_data = df_raw.tail(50)  # Show last 50 points
        fig_forecast.add_trace(go.Scatter(
            x=recent_data['DATETIME'],
            y=recent_data['ENERGY (kWh)'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecast data
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
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
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Prediction summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Forecast Summary")
            total_predicted = forecast['yhat'].sum()
            max_predicted = forecast['yhat'].max()
            avg_predicted = forecast['yhat'].mean()
            
            st.metric("Total Predicted Energy", f"{total_predicted:.4f} kWh")
            st.metric("Peak Predicted Energy", f"{max_predicted:.4f} kWh")
            st.metric("Average Predicted Energy", f"{avg_predicted:.4f} kWh")
            
            # Cost estimation
            predicted_cost = total_predicted * 7.11
            st.metric("Estimated Cost", f"₹{predicted_cost:.2f}")
        
        with col2:
            st.subheader("💡 Smart Recommendations")
            
            # Generate recommendations based on predictions
            peak_hour = forecast.loc[forecast['yhat'].idxmax(), 'ds'].hour
            low_hour = forecast.loc[forecast['yhat'].idxmin(), 'ds'].hour
            
            recommendations = []
            
            if max_predicted > avg_predicted * 1.5:
                recommendations.append(f"⚠️ High energy usage predicted at {peak_hour}:00. Consider load balancing.")
            
            if any(forecast['yhat'] < 0.001):
                recommendations.append("💡 Low usage periods detected. Good for maintenance scheduling.")
            
            weekend_forecast = forecast[forecast['ds'].dt.weekday >= 5]
            if len(weekend_forecast) > 0 and weekend_forecast['yhat'].mean() > avg_predicted:
                recommendations.append("📅 Higher weekend usage predicted. Check for unnecessary equipment.")
            
            if predicted_cost > 50:  # Arbitrary threshold
                recommendations.append(f"💰 High cost predicted (₹{predicted_cost:.2f}). Consider energy optimization.")
            else:
                recommendations.append("✅ Energy usage within optimal range.")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        
        # Detailed forecast table
        st.subheader("📊 Detailed Forecast Data")
        forecast_display = pd.DataFrame({
            'Time': forecast['ds'],
            'Predicted Energy (kWh)': forecast['yhat'].round(6),
            'Lower Bound': forecast['yhat_lower'].round(6),
            'Upper Bound': forecast['yhat_upper'].round(6),
            'Confidence': f"{confidence_level}%"
        })
        
        st.dataframe(forecast_display, use_container_width=True)
        
        # Download predictions
        csv_data = forecast_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv_data,
            file_name=f"energy_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    else:
        st.error("Unable to create prediction model. Please check your data.")

# -------------------------------
# 5. Footer with System Info
# -------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    st.caption(f"⏱ Last updated: {ist_time.strftime('%Y-%m-%d %H:%M:%S')} IST")

with col2:
    st.caption(f"📊 Data points: {len(df_raw)}")

with col3:
    if len(df_raw) > 0:
        data_freshness = (datetime.now() - df_raw['DATETIME'].max()).total_seconds() / 60
        st.caption(f"🔄 Data freshness: {data_freshness:.1f} min ago")
