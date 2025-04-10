# Realtime Energy Monitoring Dashboard

A Streamlit application that provides real-time monitoring of voltage, current, power, and energy consumption data from a Google Sheet. The application includes time series analysis and prediction capabilities.

## Features

- Real-time data visualization of electrical parameters (voltage, current, power, energy)
- Time window selection for data analysis
- Energy consumption prediction using a pre-trained LSTM model
- Customizable refresh rate
- Responsive design for all device sizes

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- A Google Cloud Platform (GCP) service account with access to Google Sheets
- A Google Sheet containing your energy monitoring data

### 2. Streamlit Cloud Deployment

1. Fork this repository
2. Add your secrets to the Streamlit Cloud dashboard:
   - Go to your app dashboard on Streamlit Cloud
   - Navigate to "Settings" > "Secrets"
   - Add your GCP service account credentials and Google Sheet ID

   Your secrets should include:
   ```
   sheet_id = "your-google-sheet-id"
   gcp_credentials = "{\"type\":\"service_account\",\"project_id\":\"your-project-id\",...}"
   ```

3. Deploy the app on Streamlit Cloud by connecting to your GitHub repository

### 3. Local Development

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/realtime-energy-monitoring.git
   cd realtime-energy-monitoring
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.streamlit/secrets.toml` file with your Google Sheets credentials:
   ```
   sheet_id = "your-google-sheet-id"
   gcp_service_account = """
   {
     "type": "service_account",
     "project_id": "your-project-id",
     ...
   }
   """
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Data Format

The application expects the following columns in your Google Sheet:
- `DATE` and `TIME` or a combined `DATE / TIME` column
- `VOLTAGE` - Voltage readings in volts (V)
- `CURRENT` - Current readings in amperes (A)
- `POWER` - Power consumption in watts (W)
- `ENERGY (kWh)` - Energy consumption in kilowatt-hours

## Model Training

The prediction model uses an LSTM neural network to forecast energy consumption. While the app uses a pre-trained model, you can retrain it by running the training script:

```
python train_model.py
```

## License

Forge
