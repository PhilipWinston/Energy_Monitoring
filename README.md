

# ⚡ Smart Energy Management System

<img width="2535" height="1206" alt="image" src="https://github.com/user-attachments/assets/b55ee84a-b164-4edc-9222-638265d6faea" />


A real-time energy monitoring and predictive analytics dashboard built with Streamlit, designed to provide insights into energy consumption, voltage stability, and future energy needs for Indian households and businesses. This system leverages Google Sheets for live data input and Facebook Prophet for accurate energy forecasting.

-----

## ✨ Features

This application offers a comprehensive suite of tools for effective energy management:

### 📊 Live Monitoring

  * **Real-time Metrics:** Displays current Voltage, Current, Power, and estimated Total Cost based on live data.
  * **System Status:** Indicates if the data feed is "Live," "Not Live," or "No Data" with a freshness indicator.
  * **Voltage Quality Analysis:** Visualizes voltage trends against Indian standards (220-240V normal) with warning and critical thresholds.
  * **Power Consumption Trends:** Tracks instantaneous power usage over time and shows its distribution.
  * **24-Hour Energy Pattern:** Identifies daily energy consumption habits to pinpoint peak and off-peak hours.
  * **Interactive Charts:** Dynamic Plotly visualizations for in-depth analysis.
  * **Activity Log:** A searchable and filterable table of recent energy readings.

### 🔮 Energy Prediction

  * **AI-Powered Forecasting:** Utilizes Facebook Prophet to predict future energy consumption (kWh) for various horizons (2, 6, 12, 24 hours).
  * **Configurable Forecast:** Adjust the forecast period, data interval, and confidence level for predictions.
  * **Confidence Intervals:** Visualizes the predicted range with a customizable confidence band.
  * **Smart Insights:** Provides actionable recommendations based on predicted usage patterns (e.g., peak usage alerts, cost optimization tips, best times for maintenance).
  * **Data Export:** Download predicted energy consumption data in CSV format.

-----

## 🚀 Getting Started

Follow these steps to set up and run the Smart Energy Management System locally.

### Prerequisites

Before you begin, ensure you have the following installed:

  * **Python 3.8+**
  * **pip** (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/smart-energy-system.git
    cd smart-energy-system
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

    *Create a `requirements.txt` file in your project root with the following content:*

    ```
    streamlit>=1.30.0
    pandas>=2.0.0
    numpy>=1.20.0
    gspread>=5.0.0
    google-auth-oauthlib>=0.4.6
    plotly>=5.0.0
    prophet>=1.1.0
    pytz>=2024.1
    ```

### Google Sheets Integration

This application uses Google Sheets as its data source. You'll need to set up a Google Service Account and share your sheet with it.

1.  **Create a Google Sheet:**

      * Create a new Google Sheet named `EnergyData` (or any name you prefer).
      * Ensure your data has columns named `DATE`, `TIME`, `VOLTAGE`, `CURRENT`, `POWER`, and `ENERGY (kWh)`. The `DATETIME` column will be generated by combining `DATE` and `TIME` in the app.
      * **Crucially, the first sheet in your Google Sheet must contain your energy data.**

2.  **Set up a Google Service Account:**

      * Go to the [Google Cloud Console](https://console.cloud.google.com/).
      * Create a new project (or select an existing one).
      * Navigate to **APIs & Services \> Credentials**.
      * Click **+ CREATE CREDENTIALS** and choose **Service Account**.
      * Give your service account a name (e.g., `energy-app-service`).
      * Grant it the `Editor` role for your project or specifically for your Google Sheets and Google Drive APIs.
      * After creation, click on the service account email, then go to the **Keys** tab, and click **ADD KEY \> Create new key \> JSON**.
      * This will download a JSON file. Rename this file to `google_service_account.json`.

3.  **Securely store your credentials in Streamlit Secrets:**

      * Open your Streamlit app folder.

      * Create a `.streamlit` folder if it doesn't exist.

      * Inside `.streamlit`, create a file named `secrets.toml`.

      * Copy the content of your `google_service_account.json` file into `secrets.toml` under a `[google_service_account]` section. It should look like this:

        ```toml
        # .streamlit/secrets.toml
        [google_service_account]
        type = "service_account"
        project_id = "your-project-id"
        private_key_id = "your-private-key-id"
        private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
        client_email = "your-service-account-email@your-project-id.iam.gserviceaccount.com"
        client_id = "your-client-id"
        auth_uri = "https://accounts.google.com/o/oauth2/auth"
        token_uri = "https://oauth2.googleapis.com/token"
        auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
        client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account-email%40your-project-id.iam.gserviceaccount.com"
        universe_domain = "googleapis.com"
        ```

      * **Important:** Share your Google Sheet with the `client_email` (the service account email) from your `secrets.toml` file. This grants the service account permission to read data from your sheet.

### Running the Application

1.  **Ensure your virtual environment is active.**

2.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py # Assuming your main script is named app.py
    ```

    This will open the application in your web browser, typically at `http://localhost:8501`.

-----

## 🛠️ Configuration

  * **Google Sheet ID:** The `SHEET_ID` variable in `app.py` is hardcoded. If your sheet has a different ID, update this line:
    ```python
    SHEET_ID = "19A2rlYT-Whb24UFcLGDn0ngDCBg8WAXR8N1PDl9F0LQ" # Replace with your Sheet ID
    ```
  * **Energy Cost:** The current cost per kWh is set to `7.11` (₹). You can adjust this value based on your local electricity tariff:
    ```python
    cost = total_energy * 7.11 # Update this value if needed
    ```
  * **Voltage Standards:** The voltage status logic is based on Indian standards (220-240V normal). You can modify the `get_voltage_status` function if different standards apply:
    ```python
    def get_voltage_status(voltage):
        if 220 <= voltage <= 240:
            return "🟢 Normal", "status-good"
        elif 200 <= voltage < 220 or 240 < voltage <= 250:
            return "🟡 Warning", "status-warning"
        else:
            return "🔴 Critical", "status-danger"
    ```

-----

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

-----
