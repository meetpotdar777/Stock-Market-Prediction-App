📈 Stock Market Prediction App

This Python application leverages historical stock data, technical indicators, and economic data to predict future stock prices using machine learning models. It supports training a new model or loading an existing one, and visualizes the model's performance along with a future price prediction.

✨ Features

Historical Data Fetching: Downloads historical stock data (Open, High, Low, Close, Volume, Adjusted Close) using yfinance. 📊

Economic Data Integration: Fetches key economic indicators (Federal Funds Rate, CPI, Unemployment Rate) from the FRED API to enrich the feature set. 💰

Comprehensive Feature Engineering:

Calculates various technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic Oscillator, ROC, CMF, PVT, Rolling Volatility). 📉

Generates lagged features for better time-series forecasting. 🕰️

Extracts temporal features like Day of Week, Day of Month, Week of Year, Month of Year, and Quarter of Year.  कैलेंडर

Calculates Rate of Change (ROC) for economic indicators. 📈

Data Preprocessing: Scales features using StandardScaler for optimal model performance. 🧹

Machine Learning Models: Supports different regression models, with a focus on LGBMRegressor for its performance and speed, including hyperparameter tuning with RandomizedSearchCV and TimeSeriesSplit for robust time-series validation. 🧠

Model Persistence: Saves and loads trained models, scalers, and feature columns using joblib to avoid retraining. 💾

Prediction: Predicts the stock price for a specified number of days into the future. 🔮

Visualization: Plots actual vs. predicted historical prices and clearly marks the single future prediction on the same chart. 📈📊

🛠️ Installation

To run this application, you'll need Python 3.7+ and several libraries.

Clone the repository (if applicable) or save the main.py file:

git clone <repository-url>
cd Stock Market Prediction App

(Or simply save the Python script as main.py or main6.py as per your current file name)

Install the required Python packages:

pip install pandas numpy matplotlib yfinance scikit-learn lightgbm ta fredapi

Obtain a FRED API Key:

Go to the FRED API website. 🔑

Register for a free API key.

Update the FRED_API_KEY variable in the main.py script with your obtained key. This is crucial if you want to use economic indicators for predictions.

# stock_predictor_version_1.py (or stock_predictor_version_9.py)
FRED_API_KEY = "YOUR_FRED_API_KEY_HERE" # Replace with your actual FRED API key

🚀 Usage

You can run the script directly from your terminal.

python stock_predictor_version_1.py
# or if you named it stock_predictor_version_9.py
python stock_predictor_version_9.py

The script will:

Check for existing trained models. If found, it will load them. Otherwise, it will train a new model. 🔄

Fetch historical stock data for the specified STOCK_TICKER. 📥

Fetch economic data from FRED (if FRED_API_KEY is set). 💰

Perform feature engineering and data scaling. ⚙️

Train (or load) and evaluate the chosen machine learning model. 📈

Display performance metrics (RMSE, R-squared). 📊

Predict the future stock price. 🔮

Generate a plot visualizing historical actual vs. predicted prices, and the future prediction. 📉✨

⚙️ Configuration

You can customize the application's behavior by modifying the variables in the --- Configuration --- section of the script:

STOCK_TICKER: The stock symbol (e.g., "AAPL", "MSFT", "GOOG"). 🍎

START_DATE: The start date for fetching historical data (e.g., "2020-01-01"). 🗓️

CUSTOM_END_DATE: Set to a specific date string ("YYYY-MM-DD") if you want to define the end of your historical data. Set to None to use today's date. 📅

N_LAG_DAYS: Number of past days to use for creating lagged features. ⏪

PREDICT_N_DAYS_FUTURE: Number of days into the future to predict the close price. ➡️

MODEL_CHOICE: Choose your machine learning model: 'LinearRegression', 'RandomForestRegressor', or 'LGBMRegressor'. 🤖

FRED_API_KEY: Your personal API key for FRED data. 🔑

📂 File Structure

The main logic is contained within a single Python script (e.g., main.py or main6.py).
Upon execution, it will create (or use) the following files in the same directory:

AAPL_stock_prediction_model_lgbmregressor.joblib: The trained machine learning model. 💾

AAPL_scaler_lgbmregressor.joblib: The fitted StandardScaler object used for feature scaling. ⚖️

AAPL_feature_columns_lgbmregressor.joblib: A list of the feature column names used during training, ensuring consistency for future predictions. 📋
(Note: The exact names will depend on your STOCK_TICKER and MODEL_CHOICE.)

🧠 Model Details

The application primarily uses LGBMRegressor, a gradient boosting framework that uses tree-based learning algorithms. It's chosen for its efficiency, speed, and accuracy, especially with large datasets. Hyperparameter tuning is performed using RandomizedSearchCV combined with TimeSeriesSplit cross-validation to ensure the model generalizes well to new, unseen time-series data. 🚀

🚧 Troubleshooting

DNSError or Failed download from yfinance: This indicates a network or DNS resolution issue on your machine. 📡

Check your internet connection. 🌐

Try ping query1.finance.yahoo.com in your terminal. 💻

Temporarily disable your firewall/antivirus. 🛡️

Consult your network administrator if you are on a corporate network. 🏢

FRED_API_KEY warning/error: Ensure you have obtained a valid FRED API key and correctly replaced "YOUR_FRED_API_KEY_HERE" with your actual key in the script. If your model was trained with economic features but you run it without a valid key, it will lead to errors during prediction. ⚠️

"Not enough processed data points to make a prediction": This means the START_DATE is too recent or required_history_days is not large enough for all technical indicators and lag features to be calculated without resulting in all NaN rows after dropna(). Consider extending START_DATE further back in time. ⏳

"AttributeError: module 'matplotlib.pyplot' has no attribute 'autoscale_view'": This has been addressed in the latest script version by removing the problematic call. Ensure you are using the most up-to-date code. ✅

Chart Dates: The chart visualization is designed to correctly align historical predictions with their actual dates and extend a line to the single future predicted date. Ensure your data ranges are reasonable for clear visualization. 🗓️📈

💡 Future Enhancements

Multiple Future Day Predictions: Extend the prediction functionality to forecast multiple days into the future, possibly showing a prediction curve. 📈➡️➡️

Interactive Visualization: Implement an interactive plot using libraries like Plotly or Bokeh for zoom, pan, and tooltip functionalities. 🖱️

User Interface: Develop a simple web-based UI using Flask/Django or Streamlit for easier interaction. 🖥️

Model Ensemble: Combine predictions from multiple models for potentially better accuracy. 🤝

News Sentiment Analysis: Integrate news sentiment data as an additional feature. 📰

Robust Error Handling: More granular error handling and user feedback. 🐛

Cloud Deployment: Deploy the model as an API for easy access. ☁️