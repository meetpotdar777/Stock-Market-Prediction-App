import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import datetime
import joblib
import os
import lightgbm as lgb
import ta # Technical analysis library
from fredapi import Fred # Importing Fred directly as installation is assumed

# --- Configuration ---
# Define the stock ticker symbol you want to predict
STOCK_TICKER = "AAPL" # Example: Apple Inc.
# Define the start and end dates for historical data
START_DATE = "2025-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d") # Today's date
# Number of days to look back for features (lag features)
N_LAG_DAYS = 5
# Number of days into the future to predict
PREDICT_N_DAYS_FUTURE = 5
# Choose your model: 'LinearRegression', 'RandomForestRegressor', or 'LGBMRegressor'
MODEL_CHOICE = 'LGBMRegressor'

# Max window for technical indicator calculations (e.g., MACD, Bollinger Bands)
MAX_INDICATOR_CALCULATION_WINDOW = max(26, 20, 14) # Max window from MACD, BB, RSI, ATR, Stoch, CMF, etc.

# FRED API Key: Obtain a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
# It's highly recommended to set this as an environment variable (e.g., export FRED_API_KEY="YOUR_KEY")
# or load it securely. For demonstration, it's placed directly.
FRED_API_KEY = "PUT YOUR FRED API KEY HERE" # <<< IMPORTANT: Replaced with your actual FRED API key


# File paths for saving/loading the model, scaler, and feature columns
MODEL_FILE = f'{STOCK_TICKER}_stock_prediction_model_{MODEL_CHOICE.lower()}.joblib'
SCALER_FILE = f'{STOCK_TICKER}_scaler_{MODEL_CHOICE.lower()}.joblib'
FEATURE_COLUMNS_FILE = f'{STOCK_TICKER}_feature_columns_{MODEL_CHOICE.lower()}.joblib'


# --- Data Collection Functions ---
def fetch_stock_data(ticker, start, end):
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        start (str): The start date in "YYYY-MM-DD" format.
        end (str): The end date in "YYYY-MM-DD" format.

    Returns:
        pd.DataFrame: A DataFrame containing historical stock data,
                      or None if data fetching fails.
    """
    print(f"Fetching historical stock data for {ticker} from {start} to {end}...")
    try:
        # Set auto_adjust=False to get original 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'
        # This prevents yfinance from modifying 'Close' and removing 'Adj Close'.
        data = yf.download(ticker, start=start, end=end, auto_adjust=False)
        if data.empty:
            print(f"No data found for {ticker} in the specified range. Please check ticker or date range.")
            return None
        print("Stock data fetched successfully.")

        # Print initial columns for debugging
        print(f"Initial columns after yfinance download: {data.columns.tolist()}")

        # Reliably flatten MultiIndex columns, if present (common for single ticker with auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Explicitly rename standard Yahoo Finance columns to consistent names
        data = data.rename(columns={
            'Adj Close': 'Adj_Close',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })

        # Further clean any remaining non-standard characters from column names for LightGBM compatibility
        data.columns = [
            col.replace(' ', '_').replace('.', '_').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(',', '')
            for col in data.columns
        ]
        
        # Ensure required columns exist after renaming/cleaning
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
        for col in required_cols:
            if col not in data.columns:
                print(f"Error: Required column '{col}' not found in fetched data after processing. Available columns: {data.columns.tolist()}")
                return None

        print(f"Final stock data columns after processing: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def fetch_economic_data(start, end, api_key):
    """
    Fetches economic indicators from the FRED API.

    Args:
        start (str): Start date for data inYYYY-MM-DD format.
        end (str): End date for data inYYYY-MM-DD format.
        api_key (str): Your FRED API key.

    Returns:
        pd.DataFrame: DataFrame containing economic indicators, or None if failed.
    """
    print("Fetching economic data from FRED...")
    try:
        fred = Fred(api_key=api_key)
        
        # Define economic series IDs. You can find more at https://fred.stlouisfed.org/
        # FEDFUNDS: Federal Funds Effective Rate (Daily)
        # CPIAUCSL: Consumer Price Index for All Urban Consumers: All Items (Monthly)
        # UNRATE: Unemployment Rate (Monthly)
        economic_series_ids = {
            'FEDFUNDS': 'Federal_Funds_Rate',
            'CPIAUCSL': 'CPI',
            'UNRATE': 'Unemployment_Rate'
        }
        
        economic_data = {}
        for fred_id, col_name in economic_series_ids.items():
            series = fred.get_series(fred_id, observation_start=start, observation_end=end)
            if series is not None:
                economic_data[col_name] = series
            else:
                print(f"Warning: Could not fetch FRED series '{fred_id}'. It might not exist for the specified date range or API key is invalid.")

        if not economic_data:
            print("No economic data could be fetched.")
            return None

        economic_df = pd.DataFrame(economic_data)
        
        # FRED data often has different frequencies. We need to handle this.
        # Resample to daily, forward-fill missing values.
        economic_df = economic_df.resample('D').mean().ffill()
        
        print(f"Economic data fetched successfully. Columns: {economic_df.columns.tolist()}")
        return economic_df
    except Exception as e:
        print(f"Error fetching economic data: {e}")
        print("Please ensure your FRED_API_KEY is valid and the 'fredapi' library is installed (`pip install fredapi`).")
        return None


# --- Feature Engineering Function ---
def add_features(df, economic_df, lag_days):
    """
    Adds daily returns, technical indicators, temporal features, and economic indicators
    to the DataFrame. Also creates lagged features for all relevant columns.
    Optimized to ensure features are added to df_temp directly before subsequent use.

    Args:
        df (pd.DataFrame): The input DataFrame with historical stock data.
        economic_df (pd.DataFrame): DataFrame with economic indicators.
        lag_days (int): Number of previous days' data to use for creating lagged features.

    Returns:
        pd.DataFrame: DataFrame with added features.
    """
    df_temp = df.copy()

    # Convert df index to datetime if not already
    if not isinstance(df_temp.index, pd.DatetimeIndex):
        df_temp.index = pd.to_datetime(df_temp.index)

    # --- Merge Economic Data ---
    economic_cols_present = []
    if economic_df is not None and not economic_df.empty:
        df_temp = df_temp.merge(
            economic_df,
            left_index=True,
            right_index=True,
            how='left'
        )
        df_temp.ffill(inplace=True)
        economic_cols_present = economic_df.columns.tolist()
    else:
        print("Warning: No economic data provided or it was empty. Skipping economic features.")

    # Ensure 'Close', 'High', 'Low' are numeric, 1-dimensional Series for ta calculations
    close_series = df_temp['Close'].astype(float).squeeze()
    high_series = df_temp['High'].astype(float).squeeze()
    low_series = df_temp['Low'].astype(float).squeeze()
    volume_series = df_temp['Volume'].astype(float).squeeze()

    # --- Directly add primary features to df_temp ---
    df_temp['Daily_Return'] = close_series.pct_change()
    df_temp['SMA_10'] = ta.trend.sma_indicator(close=close_series, window=10, fillna=False)
    df_temp['SMA_20'] = ta.trend.sma_indicator(close=close_series, window=20, fillna=False)
    df_temp['RSI'] = ta.momentum.rsi(close=close_series, window=14, fillna=False)
    df_temp['MACD'] = ta.trend.macd(close=close_series, fillna=False)
    df_temp['BB_High'] = ta.volatility.bollinger_hband(close=close_series, fillna=False)
    df_temp['BB_Low'] = ta.volatility.bollinger_lband(close=close_series, fillna=False)
    df_temp['BB_Width'] = ta.volatility.bollinger_wband(close=close_series, window=20, fillna=False)
    df_temp['BB_Percent_B'] = ta.volatility.bollinger_pband(close=close_series, window=20, fillna=False)
    df_temp['ATR'] = ta.volatility.average_true_range(high=high_series, low=low_series, close=close_series, window=14, fillna=False)
    df_temp['OBV'] = ta.volume.on_balance_volume(close=close_series, volume=volume_series, fillna=False)
    df_temp['Stoch_K'] = ta.momentum.stoch(high=high_series, low=low_series, close=close_series, window=14, fillna=False)
    df_temp['Stoch_D'] = ta.momentum.stoch_signal(high=high_series, low=low_series, close=close_series, window=3, fillna=False)
    df_temp['ROC'] = ta.momentum.roc(close=close_series, window=12, fillna=False)
    df_temp['CMF'] = ta.volume.chaikin_money_flow(high=high_series, low=low_series, close=close_series, volume=volume_series, window=20, fillna=False)
    
    # Rolling Volatility (Standard Deviation of Daily Returns) - Now safe to access Daily_Return
    df_temp['Rolling_Vol_20'] = df_temp['Daily_Return'].rolling(window=20).std()

    # Price-Volume Trend (PVT)
    pvt_series = (close_series.diff() / close_series.shift(1)) * volume_series
    df_temp['PVT'] = pvt_series.cumsum().fillna(0)

    # Temporal Features
    df_temp['Day_Of_Week'] = df_temp.index.dayofweek # Monday=0, Sunday=6
    df_temp['Day_Of_Month'] = df_temp.index.day
    df_temp['Week_Of_Year'] = df_temp.index.isocalendar().week.astype(int)
    df_temp['Month_Of_Year'] = df_temp.index.month
    df_temp['Quarter_Of_Year'] = df_temp.index.quarter

    print(f"DEBUG (add_features): Columns after primary feature creation and merge: {df_temp.columns.tolist()}")
    if 'Daily_Return' in df_temp.columns:
        print(f"DEBUG (add_features): Daily_Return head:\n{df_temp['Daily_Return'].head()}")
        print(f"DEBUG (add_features): Daily_Return tail:\n{df_temp['Daily_Return'].tail()}")


    # --- Generate Lagged Features for all relevant columns ---
    features_to_lag_candidates = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return',
        'SMA_10', 'SMA_20', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'BB_Width', 'BB_Percent_B',
        'ATR', 'OBV', 'Stoch_K', 'Stoch_D', 'ROC', 'CMF', 'Rolling_Vol_20', 'PVT',
        'Day_Of_Week', 'Day_Of_Month', 'Month_Of_Year', 'Quarter_Of_Year', 'Week_Of_Year'
    ]
    features_to_lag_candidates.extend(economic_cols_present)

    features_to_lag = [f for f in features_to_lag_candidates if f in df_temp.columns]

    print(f"Generating lag features for {len(features_to_lag)} features with {lag_days} lags...")
    lagged_features_dict = {}
    for feature in features_to_lag:
        for i in range(1, lag_days + 1):
            lagged_features_dict[f'{feature}_Lag_{i}'] = df_temp[feature].shift(i)
    
    if lagged_features_dict:
        df_temp = pd.concat([df_temp, pd.DataFrame(lagged_features_dict, index=df_temp.index)], axis=1)
    
    print("Lag features generated.")
    
    # Add rate of change for economic indicators (if applicable)
    for col in economic_cols_present:
        if f'{col}_ROC_1' not in df_temp.columns:
            df_temp[f'{col}_ROC_1'] = df_temp[col].pct_change()

    return df_temp

# --- Data Preparation Function ---
def prepare_data(df, lag_days, predict_future_days):
    """
    Prepares the DataFrame by creating various features and the target variable.

    Args:
        df (pd.DataFrame): The input DataFrame with historical stock data and potentially economic data.
        lag_days (int): Number of previous days' 'Close' prices to use as features.
        predict_future_days (int): Number of days into the future to predict.

    Returns:
        tuple: A tuple containing:
                - pd.DataFrame: Scaled Features (X_scaled).
                - pd.Series: Target variable (y).
                - pd.DataFrame: The original DataFrame with added features and target.
                - sklearn.preprocessing.StandardScaler: The fitted scaler object.
                - list: Names of the features columns used (X.columns.tolist()).
    """
    # Create the target variable (future 'Close' price)
    df_with_features = df.copy() # df here is already with all features
    df_with_features['Target'] = df_with_features['Close'].shift(-predict_future_days)

    # Drop rows with NaN values introduced by shifting and indicator calculations
    initial_rows = len(df_with_features)
    df_with_features.dropna(inplace=True)
    if len(df_with_features) == 0:
        print(f"Warning: After feature engineering and dropping NaNs, no data remains. Original rows: {initial_rows}")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), None, []

    # Select features (X) and target (y)
    # Ensure 'Adj_Close' is also excluded as it often correlates highly with 'Close'
    features_to_exclude = ['Target', 'Adj_Close', 'Close'] # Explicitly exclude Close, as its lagged versions are features
    features_list = [col for col in df_with_features.columns if col not in features_to_exclude]
    
    # Filter features_list to ensure all are numeric before selecting X
    numeric_features_list = []
    for col in features_list:
        if pd.api.types.is_numeric_dtype(df_with_features[col]):
            numeric_features_list.append(col)
        else:
            print(f"Warning: Non-numeric feature '{col}' found. Excluding from X.")
            
    X = df_with_features[numeric_features_list]
    y = df_with_features['Target']

    # Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    # Fit the scaler on the features and transform them
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    print("Features scaled successfully.")

    print(f"Data prepared with {len(numeric_features_list)} features (including technical indicators, lags, and economic data) and '{predict_future_days}'-day future target.")
    print(f"First 5 rows of scaled features (X_scaled):\n{X_scaled.head()}")
    print(f"First 5 rows of target (y):\n{y.head()}")
    return X_scaled, y, df_with_features, scaler, X.columns.tolist()


# --- Model Training and Prediction Function ---
def train_and_predict(X_scaled, y, model_choice):
    """
    Trains a selected machine learning model and makes predictions.
    Includes TimeSeriesSplit for CV and RandomizedSearchCV for tuning if LGBM.

    Args:
        X_scaled (pd.DataFrame): Scaled Features.
        y (pd.Series): Target variable.
        model_choice (str): Name of the model to use ('LinearRegression', 'RandomForestRegressor', 'LGBMRegressor').

    Returns:
        tuple: A tuple containing:
                - sklearn.base.Estimator: The trained model.
                - np.ndarray: Training set predictions.
                - np.ndarray: Test set predictions.
                - pd.DataFrame: X_train (features for training).
                - pd.DataFrame: X_test (features for testing).
                - pd.Series: y_train (target for training).
                - pd.Series: y_test (target for testing).
    """
    # Split the data into training and testing sets
    # Using shuffle=False to maintain time series order
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"Data split: Training size={len(X_train)}, Testing size={len(X_test)}")

    model = None
    if model_choice == 'LinearRegression':
        model = LinearRegression()
        print("Using Linear Regression model.")
    elif model_choice == 'RandomForestRegressor':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        print("Using RandomForestRegressor model.")
    elif model_choice == 'LGBMRegressor':
        print("Using LGBMRegressor model with RandomizedSearchCV for tuning...")
        # Define the parameter distribution for RandomizedSearchCV
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [20, 31, 40, 50],
            'max_depth': [-1, 5, 8, 10], # -1 means no limit
            'min_child_samples': [20, 30, 50],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0], # L1 regularization
            'reg_lambda': [0, 0.1, 0.5, 1.0], # L2 regularization
        }

        lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1) # verbose=-1 suppresses output
        
        # TimeSeriesSplit for cross-validation
        # n_splits determines the number of train/test splits.
        # Max_train_size limits the size of the training set in each split, useful for large datasets.
        # gap ensures a gap between training and validation set, reflecting real-world scenarios.
        tscv = TimeSeriesSplit(n_splits=5) # 5 splits

        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_dist,
            n_iter=50, # Number of parameter settings that are sampled. Reduce for faster runs.
            scoring='neg_mean_squared_error', # Using negative MSE, as GridSearchCV maximizes score
            cv=tscv,
            verbose=1,
            random_state=42,
            n_jobs=-1 # Use all available cores
        )

        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        print(f"Best parameters found: {random_search.best_params_}")
        print(f"Best RMSE found during CV: {np.sqrt(-random_search.best_score_):.2f}")
    else:
        raise ValueError("Invalid MODEL_CHOICE. Choose 'LinearRegression', 'RandomForestRegressor', or 'LGBMRegressor'.")

    # In case of GridSearchCV/RandomizedSearchCV, the best estimator is already fitted.
    # We only call fit here if we are not using RandomizedSearchCV or to ensure it's fully fitted
    # if the best_estimator_ from search itself isn't directly usable for subsequent predictions
    # without an explicit fit (which it usually is). For simplicity, keep it.
    model.fit(X_train, y_train) # Fit the best model (or the chosen model if not LGBM)
    print("Model trained successfully.")

    # Feature Importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        print("\n--- Feature Importances ---")
        print(feature_importances.nlargest(10)) # Print top 10 most important features
        print("---------------------------\n")

    # Make predictions on the training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate the model
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\n--- Model Performance ({model_choice}) ---")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}") # Corrected this line to print test_rmse
    print(f"Training R-squared: {train_r2:.2f}")
    print(f"Test R-squared: {test_r2:.2f}")
    print("-------------------------\n")

    return model, y_train_pred, y_test_pred, X_train, X_test, y_train, y_test

# --- Visualization Function ---
def visualize_predictions(original_df, y_train, y_test, y_train_pred, y_test_pred, predict_future_days):
    """
    Visualizes the actual and predicted stock prices.

    Args:
        original_df (pd.DataFrame): The DataFrame with features and target, after dropna.
        y_train (pd.Series): Actual target values for the training set.
        y_test (pd.Series): Actual target values for the test set.
        y_train_pred (np.ndarray): Predicted values for the training set.
        y_test_pred (np.ndarray): Predicted values for the test set.
        predict_future_days (int): Number of days into the future predicted.
    """
    plt.figure(figsize=(14, 7))

    all_predictions = np.concatenate((y_train_pred, y_test_pred))
    prediction_dates = original_df.index[-len(all_predictions):]

    plt.plot(original_df.index[-len(y_test) - len(y_train):],
             pd.concat([y_train, y_test]),
             label='Actual Future Close Price', color='blue', linewidth=2)

    plt.plot(prediction_dates,
             all_predictions,
             label='Model Predictions', color='red', linestyle='--', alpha=0.7)


    plt.title(f'{STOCK_TICKER} Stock Price Prediction ({predict_future_days}-day future price) using {MODEL_CHOICE}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    trained_model = None
    fitted_scaler = None
    feature_columns_loaded = [] # To store the feature columns from training

    # Adjust file paths based on MODEL_CHOICE for proper persistence
    MODEL_FILE = f'{STOCK_TICKER}_stock_prediction_model_{MODEL_CHOICE.lower()}.joblib'
    SCALER_FILE = f'{STOCK_TICKER}_scaler_{MODEL_CHOICE.lower()}.joblib'
    FEATURE_COLUMNS_FILE = f'{STOCK_TICKER}_feature_columns_{MODEL_CHOICE.lower()}.joblib'

    # Option to load existing model, scaler, and feature columns
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(FEATURE_COLUMNS_FILE):
        print(f"Loading existing model from {MODEL_FILE}, scaler from {SCALER_FILE}, and feature columns from {FEATURE_COLUMNS_FILE}...")
        try:
            trained_model = joblib.load(MODEL_FILE)
            fitted_scaler = joblib.load(SCALER_FILE)
            feature_columns_loaded = joblib.load(FEATURE_COLUMNS_FILE)
            print("Model, scaler, and feature columns loaded successfully. Skipping retraining.")

            # Check if the loaded model was trained with economic features
            # This is a heuristic check based on expected column names
            model_trained_with_economic_features = any(
                col.startswith(('Federal_Funds_Rate', 'CPI', 'Unemployment_Rate')) for col in feature_columns_loaded
            )

            # Only warn if model was trained with economic features BUT FRED_API_KEY is empty/placeholder
            if model_trained_with_economic_features and (not FRED_API_KEY or FRED_API_KEY == "YOUR_FRED_API_KEY_HERE"):
                print("\n--- CRITICAL ERROR: FRED API Key Not Set for Loaded Model ---")
                print("Your loaded model was trained WITH economic data, but your FRED_API_KEY is missing or a placeholder.")
                print("To ensure consistent predictions, you MUST update FRED_API_KEY with your actual key.")
                print("If you prefer NOT to use economic data, delete the saved model files (joblib files) and rerun the script.")
                print("-------------------------------------------------------------\n")
                exit("Exiting due to FRED_API_KEY mismatch for loaded model.")

        except Exception as e:
            print(f"Error loading saved files: {e}. Proceeding with fresh training.")
            trained_model = None # Reset to trigger retraining
    else:
        print("No existing model, scaler, or feature columns found. Proceeding with data fetching and training.")
    
    # If model was not loaded (either not found or error during loading), proceed with training
    if trained_model is None:
        # 1. Fetch Stock Data
        stock_df = fetch_stock_data(STOCK_TICKER, START_DATE, END_DATE)
        if stock_df is None:
            print("Failed to fetch stock data. Exiting.")
            exit()

        # 2. Fetch Economic Data
        # Ensure FRED_API_KEY is present and not a placeholder
        if not FRED_API_KEY or FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
            print("WARNING: FRED_API_KEY is not set or is a placeholder. Economic data will not be fetched during training.")
            economic_df = None
        else:
            economic_df = fetch_economic_data(START_DATE, END_DATE, FRED_API_KEY)
            if economic_df is None:
                print("Failed to fetch economic data during training. Proceeding with stock data only.")

        # 3. Add Features (including technical, temporal, and economic)
        # Ensure that add_features can handle None for economic_df
        df_with_all_features = add_features(stock_df, economic_df, N_LAG_DAYS)


        # 4. Prepare Data (create target, drop NaNs, and scale)
        X_scaled, y, processed_df, fitted_scaler, feature_columns_for_saving = prepare_data(df_with_all_features, N_LAG_DAYS, PREDICT_N_DAYS_FUTURE)

        # 5. Train and Predict
        if not X_scaled.empty and not y.empty:
            trained_model, y_train_pred, y_test_pred, X_train, X_test, y_train, y_test = train_and_predict(X_scaled, y, MODEL_CHOICE)

            # Save the trained model, scaler, and feature columns
            print(f"Saving trained model to {MODEL_FILE}, scaler to {SCALER_FILE}, and feature columns to {FEATURE_COLUMNS_FILE}...")
            joblib.dump(trained_model, MODEL_FILE)
            joblib.dump(fitted_scaler, SCALER_FILE)
            joblib.dump(feature_columns_for_saving, FEATURE_COLUMNS_FILE)
            print("Model, scaler, and feature columns saved.")

            # 6. Visualize Results (only if training happened)
            visualize_predictions(processed_df, y_train, y_test, y_train_pred, y_test_pred, PREDICT_N_DAYS_FUTURE)
            # Set loaded feature columns for future predictions
            feature_columns_loaded = feature_columns_for_saving
        else:
            print("Data preparation resulted in empty features or target. Cannot train model.")
            exit()

    # --- Predict the next 'PREDICT_N_DAYS_FUTURE' days (works with loaded or newly trained model) ---
    if trained_model is not None and fitted_scaler is not None and feature_columns_loaded:
        print(f"\n--- Predicting {PREDICT_N_DAYS_FUTURE} day(s) into the future ---")
        try:
            # Always fetch fresh stock data for prediction to ensure we have the very latest
            # Fetch enough historical data to compute all necessary features and lags.
            # We need data up to today to compute today's features for tomorrow's prediction.
            # The start date needs to go back far enough to allow all rolling windows and lags to fill.
            # MAX_INDICATOR_CALCULATION_WINDOW covers the longest indicator window (e.g., 26 for MACD, 20 for BB/CMF/Rolling_Vol).
            # N_LAG_DAYS covers the maximum lag for features.
            # An extra buffer (e.g., +10 days) is good for non-trading days or data sparsity.
            required_history_days = MAX_INDICATOR_CALCULATION_WINDOW + N_LAG_DAYS + 30 # Increased buffer
            prediction_start_date = (datetime.date.today() - datetime.timedelta(days=required_history_days)).strftime("%Y-%m-%d")

            latest_stock_df = fetch_stock_data(STOCK_TICKER, prediction_start_date, END_DATE)
            if latest_stock_df is None:
                print("Could not fetch latest stock data for future prediction.")
                exit()
            
            # Fetch latest economic data for prediction
            if any(col.startswith(('Federal_Funds_Rate', 'CPI', 'Unemployment_Rate')) for col in feature_columns_loaded):
                if not FRED_API_KEY or FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
                    print("\nWARNING: FRED_API_KEY is not set for prediction. Model was likely trained with economic data.")
                    print("This may lead to inconsistent predictions or errors. Please update FRED_API_KEY.")
                    latest_economic_df = None
                else:
                    latest_economic_df = fetch_economic_data(prediction_start_date, END_DATE, FRED_API_KEY)
                    if latest_economic_df is None:
                        print("Failed to fetch latest economic data for prediction. Proceeding with stock data only, but feature set may mismatch trained model.")
            else:
                latest_economic_df = None

            # Apply the same feature engineering function used for training data
            engineered_features_for_pred = add_features(latest_stock_df.copy(), latest_economic_df, N_LAG_DAYS)

            # Drop any NaNs that might appear at the start due to lag/indicator calculations
            engineered_features_for_pred.dropna(inplace=True)
            print(f"DEBUG (Prediction): Columns after dropna: {engineered_features_for_pred.columns.tolist()}")
            print(f"DEBUG (Prediction): Shape after dropna: {engineered_features_for_pred.shape}")
            nan_summary = engineered_features_for_pred.isnull().sum()
            print(f"DEBUG (Prediction): NaNs per column after dropna:\n{nan_summary[nan_summary > 0]}")

            if 'Daily_Return' in engineered_features_for_pred.columns:
                print(f"DEBUG (Prediction): Is 'Daily_Return' present in columns after dropna? {'Daily_Return' in engineered_features_for_pred.columns}")
                if not engineered_features_for_pred['Daily_Return'].empty:
                    print(f"DEBUG (Prediction): Last Daily_Return value after dropna: {engineered_features_for_pred['Daily_Return'].iloc[-1]}")
                    print(f"DEBUG (Prediction): Is Last Daily_Return NaN? {pd.isna(engineered_features_for_pred['Daily_Return'].iloc[-1])}")
                else:
                    print("DEBUG (Prediction): Daily_Return column is empty after dropna (all values might be NaN).")
            else:
                print(f"DEBUG (Prediction): 'Daily_Return' not found in engineered_features_for_pred columns after dropna.")


            if not engineered_features_for_pred.empty:
                # Filter `engineered_features_for_pred` to only include columns that `feature_columns_loaded` expects
                missing_cols_in_pred = set(feature_columns_loaded) - set(engineered_features_for_pred.columns)
                if missing_cols_in_pred:
                    print(f"ERROR: Missing expected features for prediction: {missing_cols_in_pred}")
                    print("This often happens if the FRED_API_KEY was set during training but not for prediction, or vice-versa.")
                    print("Please ensure your FRED_API_KEY is correct and consistent with your trained model.")
                    exit("Exiting due to feature mismatch for prediction.")
                
                extra_cols_in_pred = set(engineered_features_for_pred.columns) - set(feature_columns_loaded)
                if extra_cols_in_pred:
                    print(f"WARNING: Extra features found in prediction data not used during training: {extra_cols_in_pred}. These will be ignored.")
                    # Drop extra columns to avoid issues with scaler or model if it's strict
                    engineered_features_for_pred = engineered_features_for_pred.drop(columns=list(extra_cols_in_pred))

                # Select features and ensure order
                # Extract the last row as a Series and convert to a DataFrame.
                # Then, reindex to explicitly ensure the column order matches 'feature_columns_loaded'.
                last_row_features = engineered_features_for_pred.iloc[-1]
                next_day_features_df = pd.DataFrame([last_row_features.reindex(feature_columns_loaded)], columns=feature_columns_loaded)
                
                # Debug print for the last row before scaling
                print(f"DEBUG (Prediction): next_day_features_df before scaling:\n{next_day_features_df}")
                print(f"DEBUG (Prediction): Columns in next_day_features_df: {next_day_features_df.columns.tolist()}")
                if 'Daily_Return' in next_day_features_df.columns:
                    print(f"DEBUG (Prediction): Is 'Daily_Return' in next_day_features_df columns? True")
                    print(f"DEBUG (Prediction): Daily_Return value in next_day_features_df: {next_day_features_df['Daily_Return'].iloc[0]}")
                    print(f"DEBUG (Prediction): Is Daily_Return NaN in next_day_features_df? {pd.isna(next_day_features_df['Daily_Return'].iloc[0])}")
                else:
                    print(f"DEBUG (Prediction): 'Daily_Return' not found in next_day_features_df columns.")


                # Scale these new features using the *fitted scaler*
                next_day_features_scaled = fitted_scaler.transform(next_day_features_df)
                next_future_price_pred = trained_model.predict(next_day_features_scaled)

                last_known_date = latest_stock_df.index[-1]
                prediction_date = last_known_date + pd.Timedelta(days=PREDICT_N_DAYS_FUTURE)

                print(f"Predicted {STOCK_TICKER} close price for {prediction_date.strftime('%Y-%m-%d')} ({PREDICT_N_DAYS_FUTURE} day(s) from last known data): ${next_future_price_pred[0]:.2f}")
            else:
                print("Not enough processed data points to make a prediction after feature engineering and dropping NaNs.")
                print("Consider extending the `required_raw_data_points` or the `START_DATE`.")
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            print("Please review the data fetching and feature engineering steps for inconsistencies.")
            print(f"DEBUG: The error occurred likely when trying to access columns in 'engineered_features_for_pred' or 'next_day_features_df'.")
    else:
        print("\nSkipping future prediction as model, scaler, or feature columns are not available or not successfully loaded/trained.")

