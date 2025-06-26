import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
from xgboost import XGBRegressor
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

class ModelRunnerAgent:
    def __init__(self):
        self.results = {}
    
    def preprocess_for_prophet(self, df):
        df_prophet = df[['Date', 'Close']].rename(columns = {'Date': 'ds', 'Close': 'y'})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)
        return df_prophet
    
    def preprocess_for_xgboost(self, df, n_lags = 5):
        df = df.copy()
        for lag in range(1, n_lags + 1):
            df[f'lag_{lag}'] = df['Close'].shift(lag)
        df.dropna(inplace = True)
        return df
    
    def preprocess_for_arima(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace = True)
        return df
    
    def train_prophet_model(self, df):
        df_prophet = self.preprocess_for_prophet(df)
        model = Prophet(daily_seasonality = True)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods = 30)
        forecast = model.predict(future)
        pred = forecast[["ds", "yhat"]].set_index("ds")["yhat"][-30:]
        true = df_prophet.set_index("ds")["y"][-30:]
        error = mean_squared_error(true[-30:], pred[:30])
        error = np.sqrt(error)
        return model, error
    
    def train_xgboost(self, df):
        df_xgboost = self.preprocess_for_xgboost(df)
        X = df_xgboost[[col for col in df_xgboost.columns if "lag_" in col]]
        y = df_xgboost["Close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)
        model = XGBRegressor(n_estimators = 100, learning_rate = 0.1, random_state = 24)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        error = mean_squared_error(y_test, pred)
        error = np.sqrt(error)
        return model, error, X.columns.tolist()
    
    def train_arima(self, df):
        df_arima = self.preprocess_for_arima(df)
        series = df_arima["Close"]
        train, test = series[:-30], series[-30:]
        model = auto_arima(train, seasonal = False, suppress_warnings = True)
        pred = model.predict(n_periods = len(test))
        error = mean_squared_error(test, pred)
        error = np.sqrt(error)
        return model, error
    
    def train_models(self, df):
        model_prophet, error_prophet = self.train_prophet_model(df)
        model_xgboost, error_xgboost, xgb_features = self.train_xgboost(df)
        model_arima, error_arima = self.train_arima(df)
        
        self.results = {
            "prophet": {
                "name": "Prophet",
                "model": model_prophet,
                "error": error_prophet
            },
            "xgboost": {
                "name": "XGBoost",
                "model": model_xgboost,
                "error": error_xgboost,
                "features": xgb_features
            },
            "arima": {
                "name": "ARIMA",
                "model": model_arima,
                "error": error_arima
            }
        }
        return self.select_best_model()
    
    def select_best_model(self):
        best_model = min(self.results, key = lambda x: self.results[x]["error"])
        best_model_info = self.results[best_model]
        return {
            "model_name": best_model,
            "model": best_model_info["model"],
            "error": best_model_info["error"],
            "features": best_model_info.get("features", None)
        }
    
    def print_results(self):
        for model, info in self.results.items():
            print(f"\n{model.capitalize()} Model Results:")
            print(f"RMSE: {info['error']:.4f}")
            if model == "xgboost":
                print(f"Features Used: {info['features']}")
            print("\n")
    
    def get_all_models(self):
        return self.results