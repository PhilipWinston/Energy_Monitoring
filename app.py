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
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Streamlit Configuration
# -------------------------------
st.set_page_config(
    page_title="⚡ Smart Energy Management System", 
    layout="wide",
    page_icon="⚡"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric > div > div > div > div {
        font-size: 1.2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-danger { color: #dc3545; }
    .energy-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Data Loading Functions
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
        st.error(f"⚠️ Error loading data: {e}")
        return None

def get_voltage_status(voltage):
    """Get voltage status with Indian standards (220-240V normal)"""
    if 220 <= voltage <= 240:
        return "🟢 Normal", "status-good"
    elif 200 <= voltage < 220 or 240 < voltage <= 250:
        return "🟡 Warning", "status-warning"
    else:
        return "🔴 Critical", "status-danger"

# -------------------------------
# Load and Prepare Data
# -------------------------------
df_raw = load_google_sheet_data()

if df_raw is not None:
    try:
        df_raw["DATETIME"] = pd.to_datetime(
            df_raw["DATE"] + " " + df_raw["TIME"],
            format="%Y-%m-%d %H:%M:%S"
        )
        df_raw = df_raw.sort_values("DATETIME")
        df_raw['ENERGY (kWh)'] = pd.to_numeric(df_raw['ENERGY (kWh)'], errors='coerce')
        df_raw = df_raw.dropna()
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        st.stop()
else:
    st.error("❌ Unable to load data. Please check your connection.")
    st.stop()

# -------------------------------
# Header Section
# -------------------------------
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='color: #2E86AB; font-size: 3rem; margin-bottom: 0;'>⚡ Smart Energy Management System</h1>
    <p style='color: #6C757D; font-size: 1.2rem; margin-top: 0.5rem;'>Real-time Monitoring & Predictive Analytics for India</p>
</div>
""", unsafe_allow_html=True)

# Control Panel
col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
with col1:
    if st.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

with col2:
    st.markdown("**Status:** 🟢 Live")

with col3:
    if len(df_raw) > 0:
        data_age = (datetime.now() - df_raw['DATETIME'].max()).total_seconds() / 60
        st.markdown(f"**Freshness:** {data_age:.1f}min")

# Create tabs with custom styling
tab1, tab2 = st.tabs(["📊 Live Monitoring", "🔮 Energy Prediction"])

# -------------------------------
# TAB 1: LIVE MONITORING
# -------------------------------
with tab1:
    if len(df_raw) > 0:
        latest = df_raw.iloc[-1]
        
        # Current Status Dashboard
        st.markdown("### 📊 Current System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            voltage_status, voltage_class = get_voltage_status(latest['VOLTAGE'])
            st.markdown(f"""
            <div class="energy-card">
                <h3>⚡ Voltage</h3>
                <h2>{latest['VOLTAGE']:.1f} V</h2>
                <p class="{voltage_class}">{voltage_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="energy-card">
                <h3>🔋 Current</h3>
                <h2>{latest['CURRENT']:.2f} A</h2>
                <p>Load Active</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="energy-card">
                <h3>⚙️ Power</h3>
                <h2>{latest['POWER']:.3f} kW</h2>
                <p>Instantaneous</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_energy = df_raw['ENERGY (kWh)'].sum()
            cost = total_energy * 7.11
            st.markdown(f"""
            <div class="energy-card">
                <h3>💰 Total Cost</h3>
                <h2>₹{cost:.2f}</h2>
                <p>{total_energy:.4f} kWh</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Visualization
        st.markdown("### 📈 Real-time Analytics")
        
        viz_type = st.selectbox(
            "Select Visualization:",
            ["📊 Complete Dashboard", "⚡ Voltage Analysis", "🔋 Power Trends", "📉 Energy Pattern"],
            key="viz_selector"
        )
        
        if viz_type == "📊 Complete Dashboard":
            # Multi-metric dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('⚡ Voltage Monitoring', '🔋 Current Flow', 
                              '⚙️ Power Consumption', '📊 Energy Usage'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                vertical_spacing=0.08
            )
            
            # Voltage with reference lines
            fig.add_trace(
                go.Scatter(x=df_raw['DATETIME'], y=df_raw['VOLTAGE'], 
                          name='Voltage', line=dict(color='#FF6B35', width=3)),
                row=1, col=1
            )
            fig.add_hline(y=230, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_hline(y=220, line_dash="dot", line_color="orange", row=1, col=1)
            fig.add_hline(y=240, line_dash="dot", line_color="orange", row=1, col=1)
            
            # Current
            fig.add_trace(
                go.Scatter(x=df_raw['DATETIME'], y=df_raw['CURRENT'], 
                          name='Current', line=dict(color='#4ECDC4', width=3),
                          fill='tonexty' if len(df_raw) > 1 else None),
                row=1, col=2
            )
            
            # Power
            fig.add_trace(
                go.Scatter(x=df_raw['DATETIME'], y=df_raw['POWER'], 
                          name='Power', line=dict(color='#45B7D1', width=3),
                          mode='lines+markers', marker=dict(size=4)),
                row=2, col=1
            )
            
            # Energy bars
            fig.add_trace(
                go.Bar(x=df_raw['DATETIME'], y=df_raw['ENERGY (kWh)'], 
                       name='Energy', marker_color='#96CEB4', opacity=0.7),
                row=2, col=2
            )
            
            fig.update_layout(
                height=700, 
                showlegend=False,
                title_text="🏠 Smart Energy Monitoring Dashboard",
                title_x=0.5,
                title_font_size=20
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "⚡ Voltage Analysis":
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig_voltage = go.Figure()
                fig_voltage.add_trace(go.Scatter(
                    x=df_raw['DATETIME'], 
                    y=df_raw['VOLTAGE'],
                    mode='lines+markers',
                    name='Voltage',
                    line=dict(color='#FF6B35', width=3),
                    marker=dict(size=5)
                ))
                
                # Indian voltage standards
                fig_voltage.add_hline(y=230, line_dash="solid", line_color="green", 
                                     annotation_text="Ideal (230V)", annotation_position="top right")
                fig_voltage.add_hline(y=220, line_dash="dash", line_color="orange", 
                                     annotation_text="Min Normal (220V)")
                fig_voltage.add_hline(y=240, line_dash="dash", line_color="orange", 
                                     annotation_text="Max Normal (240V)")
                fig_voltage.add_hline(y=200, line_dash="dot", line_color="red", 
                                     annotation_text="Critical Low (200V)")
                
                fig_voltage.update_layout(
                    title="⚡ Voltage Quality Analysis (Indian Standards)",
                    xaxis_title="Time",
                    yaxis_title="Voltage (V)",
                    height=500
                )
                st.plotly_chart(fig_voltage, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Voltage Stats")
                voltage_stats = df_raw['VOLTAGE'].describe()
                
                for stat, value in [("Current", latest['VOLTAGE']), 
                                   ("Average", voltage_stats['mean']),
                                   ("Maximum", voltage_stats['max']),
                                   ("Minimum", voltage_stats['min'])]:
                    status_class = get_voltage_status(value)[1]
                    st.markdown(f"""
                    <div style='padding: 0.5rem; margin: 0.25rem 0; border-left: 4px solid #FF6B35;'>
                        <strong>{stat}:</strong> <span class='{status_class}'>{value:.1f}V</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        elif viz_type == "🔋 Power Trends":
            # Power analysis with advanced metrics
            fig_power = make_subplots(
                rows=2, cols=1,
                subplot_titles=('🔋 Power Consumption Over Time', '📊 Power Distribution'),
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
                vertical_spacing=0.1
            )
            
            # Power trend
            fig_power.add_trace(
                go.Scatter(x=df_raw['DATETIME'], y=df_raw['POWER'],
                          mode='lines+markers', name='Power',
                          line=dict(color='#45B7D1', width=3),
                          fill='tonexty'),
                row=1, col=1
            )
            
            avg_power = df_raw['POWER'].mean()
            fig_power.add_hline(y=avg_power, line_dash="dash", line_color="blue",
                               annotation_text=f"Avg: {avg_power:.3f}kW", row=1, col=1)
            
            # Power histogram
            fig_power.add_trace(
                go.Histogram(x=df_raw['POWER'], nbinsx=20, name='Distribution',
                            marker_color='#96CEB4', opacity=0.7),
                row=2, col=1
            )
            
            fig_power.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_power, use_container_width=True)
        
        elif viz_type == "📉 Energy Pattern":
            # Energy consumption patterns
            df_raw['Hour'] = df_raw['DATETIME'].dt.hour
            hourly_energy = df_raw.groupby('Hour')['ENERGY (kWh)'].sum().reset_index()
            
            fig_pattern = px.bar(
                hourly_energy, x='Hour', y='ENERGY (kWh)',
                title='🕐 24-Hour Energy Consumption Pattern',
                color='ENERGY (kWh)',
                color_continuous_scale='plasma',
                labels={'Hour': 'Hour of Day', 'ENERGY (kWh)': 'Energy Consumed (kWh)'}
            )
            
            fig_pattern.update_layout(height=500)
            st.plotly_chart(fig_pattern, use_container_width=True)
        
        # Recent data with better formatting
        st.markdown("### 📋 Recent Activity Log")
        display_count = st.slider("Records to display:", 5, 25, 10, key="data_display")
        
        recent_data = df_raw.tail(display_count).copy()
        recent_data['Status'] = recent_data['VOLTAGE'].apply(lambda x: get_voltage_status(x)[0])
        recent_data = recent_data[['DATETIME', 'VOLTAGE', 'CURRENT', 'POWER', 'ENERGY (kWh)', 'Status']]
        
        st.dataframe(
            recent_data,
            use_container_width=True,
            column_config={
                "DATETIME": st.column_config.DatetimeColumn("Time", format="DD/MM/YY HH:mm:ss"),
                "VOLTAGE": st.column_config.NumberColumn("Voltage (V)", format="%.1f"),
                "CURRENT": st.column_config.NumberColumn("Current (A)", format="%.2f"),
                "POWER": st.column_config.NumberColumn("Power (kW)", format="%.3f"),
                "ENERGY (kWh)": st.column_config.NumberColumn("Energy (kWh)", format="%.6f"),
            }
        )

# -------------------------------
# TAB 2: ENERGY PREDICTION
# -------------------------------
with tab2:
    st.markdown("### 🔮 AI-Powered Energy Forecasting")
    
    # Prediction controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        horizon = st.selectbox(
            "🕐 Forecast Period:",
            ["Next 2 Hours", "Next 6 Hours", "Next 12 Hours", "Next 24 Hours"],
            key="forecast_horizon"
        )
    
    with col2:
        interval = st.selectbox(
            "📊 Data Interval:",
            ["15 minutes", "30 minutes", "1 hour"],
            key="forecast_interval"
        )
    
    with col3:
        confidence = st.slider("🎯 Confidence Level:", 85, 99, 95, key="confidence")
    
    # Prophet model setup
    @st.cache_resource
    def create_forecast_model(df):
        """Create Prophet forecasting model"""
        try:
            # Prepare data
            prophet_df = pd.DataFrame({
                'ds': df['DATETIME'],
                'y': df['ENERGY (kWh)']
            })
            
            # Add features
            prophet_df['is_working_hour'] = ((df['DATETIME'].dt.hour >= 8) & 
                                           (df['DATETIME'].dt.hour <= 20)).astype(int)
            prophet_df['is_weekend'] = (df['DATETIME'].dt.dayofweek >= 5).astype(int)
            
            # Create model
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=0.1,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=confidence/100
            )
            
            model.add_regressor('is_working_hour')
            model.add_regressor('is_weekend')
            
            model.fit(prophet_df)
            return model, prophet_df
        except Exception as e:
            return None, str(e)
    
    # Generate forecast
    with st.spinner("🤖 Training AI model and generating predictions..."):
        model, error = create_forecast_model(df_raw)
    
    if model is not None:
        # Create future dataframe
        horizon_map = {"Next 2 Hours": 2, "Next 6 Hours": 6, "Next 12 Hours": 12, "Next 24 Hours": 24}
        interval_map = {"15 minutes": 15, "30 minutes": 30, "1 hour": 60}
        
        hours = horizon_map[horizon]
        minutes = interval_map[interval]
        periods = int((hours * 60) / minutes)
        
        last_time = df_raw['DATETIME'].max()
        future_dates = pd.date_range(
            start=last_time + timedelta(minutes=minutes),
            periods=periods,
            freq=f'{minutes}min'
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['is_working_hour'] = ((future_df['ds'].dt.hour >= 8) & 
                                       (future_df['ds'].dt.hour <= 20)).astype(int)
        future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
        
        # Make predictions
        forecast = model.predict(future_df)
        
        # Visualization
        st.markdown("### 📈 Energy Consumption Forecast")
        
        # Enhanced forecast plot
        fig_forecast = go.Figure()
        
        # Historical data (last 50 points)
        recent = df_raw.tail(50)
        fig_forecast.add_trace(go.Scatter(
            x=recent['DATETIME'],
            y=recent['ENERGY (kWh)'],
            mode='lines+markers',
            name='📊 Historical Data',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        # Forecast
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines+markers',
            name='🔮 AI Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Confidence interval
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,127,14,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'🎯 {confidence}% Confidence Band'
        ))
        
        fig_forecast.update_layout(
            title=f'🔮 Energy Forecast - {horizon}',
            xaxis_title='Time',
            yaxis_title='Energy Consumption (kWh)',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Forecast Summary")
            
            total_predicted = forecast['yhat'].sum()
            peak_consumption = forecast['yhat'].max()
            avg_consumption = forecast['yhat'].mean()
            predicted_cost = total_predicted * 7.11
            
            metrics_data = [
                ("🔋 Total Energy", f"{total_predicted:.4f} kWh"),
                ("⚡ Peak Usage", f"{peak_consumption:.4f} kWh"),
                ("📊 Average", f"{avg_consumption:.4f} kWh"),
                ("💰 Estimated Cost", f"₹{predicted_cost:.2f}")
            ]
            
            for label, value in metrics_data:
                st.markdown(f"""
                <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                           padding: 1rem; margin: 0.5rem 0; border-radius: 10px; color: white;'>
                    <strong>{label}:</strong> {value}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💡 Smart Insights")
            
            peak_hour = forecast.loc[forecast['yhat'].idxmax(), 'ds'].hour
            min_hour = forecast.loc[forecast['yhat'].idxmin(), 'ds'].hour
            
            insights = []
            
            if peak_consumption > avg_consumption * 1.8:
                insights.append(f"⚠️ Peak usage at {peak_hour}:00 - Consider load shifting")
            
            if predicted_cost > 100:
                insights.append(f"💸 High cost alert: ₹{predicted_cost:.2f} - Optimize usage")
            else:
                insights.append("✅ Cost within reasonable range")
            
            weekend_periods = forecast[forecast['ds'].dt.weekday >= 5]
            if len(weekend_periods) > 0 and weekend_periods['yhat'].mean() > avg_consumption * 1.2:
                insights.append("📅 High weekend usage predicted")
            
            insights.append(f"🌙 Lowest usage at {min_hour}:00 - Best for maintenance")
            insights.append(f"📈 Confidence level: {confidence}% - High reliability")
            
            for insight in insights:
                st.info(insight)
        
        # Download forecast
        forecast_export = pd.DataFrame({
            'DateTime': forecast['ds'],
            'Predicted_Energy_kWh': forecast['yhat'].round(6),
            'Lower_Bound': forecast['yhat_lower'].round(6),
            'Upper_Bound': forecast['yhat_upper'].round(6)
        })
        
        csv_data = forecast_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv_data,
            file_name=f"energy_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            type="primary"
        )
    
    else:
        st.error(f"❌ Unable to create forecast model: {error}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    st.caption(f"🕐 Last updated: {ist_time.strftime('%d/%m/%Y %H:%M:%S')} IST")

with col2:
    st.caption(f"📊 Total records: {len(df_raw):,}")

with col3:
    if len(df_raw) > 0:
        uptime = (df_raw['DATETIME'].max() - df_raw['DATETIME'].min()).total_seconds() / 3600
        st.caption(f"⏱ Monitoring duration: {uptime:.1f} hours")
