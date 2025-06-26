import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

class PredictorAgent:
    def __init__(self):
        pass

    def predict_next_90_days(self, model_name, model, hist_df, features = None):
        if model_name == "prophet":
            return self.predict_prophet(model, hist_df)
        elif model_name == "xgboost":
            return self.predict_xgboost(model, hist_df, features)
        elif model_name == "arima":
            return self.predict_arima(model, hist_df)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    
    def predict_prophet(self, model, df):
        future = model.make_future_dataframe(periods = 90)
        forecast = model.predict(future)
        forecast = forecast[["ds", "yhat"]].tail(90).rename(columns = {"ds": "Date", "yhat": "Predicted"})
        forecast["Date"] = pd.to_datetime(forecast["Date"])
        return forecast
    
    def predict_xgboost(self, model, df, features):
        steps = max(int(f.split("_")[1]) for f in features)
        history = df["Close"].values[-steps:].tolist()
        dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days = 1), periods = 90, freq = "B")
        predictions = []

        for _ in range(90):
            X_pred = np.array(history[-steps:]).reshape(1, -1)
            y_pred = model.predict(X_pred)[0]
            predictions.append(y_pred)
            history.append(y_pred)

        return pd.DataFrame({"Date": dates, "Predicted": predictions})
    
    def predict_arima(self, model, df):
        forecast = model.predict(n_periods = 90)
        last_date = df.index.max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days = 1), periods = 90, freq = "B")
        return pd.DataFrame({"Date": future_dates, "Predicted": forecast})
