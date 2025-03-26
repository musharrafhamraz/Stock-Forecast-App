from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd

def run_model(data, model_name, horizon):
    """
    Run the selected forecasting model.
    :param data: DataFrame containing stock data
    :param model_name: Name of the model to run (e.g., "ARIMA")
    :param horizon: Number of months to predict
    :return: Forecasted values, MAE, RMSE
    """
    # Check if data contains multiple stock symbols
    unique_symbols = data['Symbol'].unique()
    if len(unique_symbols) == 1:
        # Single stock: Use 'Close' prices directly
        close_prices = data['Close']
    elif len(unique_symbols) == 2:
        # Two stocks: Calculate the spread (price difference)
        stock1_data = data[data['Symbol'] == unique_symbols[0]]['Close']
        stock2_data = data[data['Symbol'] == unique_symbols[1]]['Close']
        close_prices = stock1_data.values - stock2_data.values
    else:
        raise ValueError("Unsupported number of stock symbols. Please enter 1 or 2 symbols.")

    # Handle each model
    if model_name == "ARIMA":
        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon)

    elif model_name == "SARIMA":
        model = SARIMAX(close_prices, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=horizon)

    elif model_name == "ADF":
        # Perform Augmented Dickey-Fuller test
        result = adfuller(close_prices)
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        return None, None, None  # ADF does not produce forecasts

    elif model_name == "STL":
        # Decompose time series using STL
        stl = STL(close_prices, seasonal=13)
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal
        residual = res.resid
        print("STL Decomposition Complete")
        return None, None, None  # STL does not produce forecasts

    elif model_name == "VECM":
        # Vector Error Correction Model for two stocks
        if len(unique_symbols) != 2:
            raise ValueError("VECM requires exactly two stock symbols.")
        
        # Combine data for both stocks
        combined_data = pd.concat([stock1_data, stock2_data], axis=1)
        combined_data.columns = ['Stock1', 'Stock2']
        
        # Select cointegration rank
        coint_rank = select_coint_rank(combined_data, det_order=0, k_ar_diff=1)
        
        # Fit VECM model
        vecm = VECM(combined_data, k_ar_diff=1, coint_rank=coint_rank.rank)
        vecm_fit = vecm.fit()
        
        # Generate forecast
        forecast = vecm_fit.predict(steps=horizon)
        forecast = forecast[:, 0] - forecast[:, 1]  # Spread forecast

    elif model_name == "LSTM":
        # Prepare data for LSTM
        def create_dataset(dataset, look_back=1):
            X, y = [], []
            for i in range(len(dataset) - look_back):
                X.append(dataset[i:(i + look_back)])
                y.append(dataset[i + look_back])
            return np.array(X), np.array(y)

        look_back = 10
        dataset = close_prices.values.reshape(-1, 1)
        X, y = create_dataset(dataset, look_back)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(look_back, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

        # Predict
        forecast = model.predict(X_test[-horizon:])
        forecast = forecast.flatten()

    else:
        raise NotImplementedError(f"Model {model_name} is not implemented yet.")

    # Calculate evaluation metrics (if applicable)
    if model_name not in ["ADF", "STL"]:
        actual = close_prices[-horizon:]  # Last 'horizon' points as actual
        mae = mean_absolute_error(actual, forecast[:len(actual)])
        rmse = np.sqrt(mean_squared_error(actual, forecast[:len(actual)]))
    else:
        mae, rmse = None, None

    return forecast, mae, rmse