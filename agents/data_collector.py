import yfinance as yf
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class DataCollectorAgent:
    def __init__(self):
        pass

    def get_stock_data(self, ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """
        Get stock data from Yahoo Finance
        Args:
            ticker: str
            period: str
            interval: str
        Returns:
            pd.DataFrame
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period = period, interval = interval)
            
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            df.reset_index(inplace = True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace = True)
            return df
        except Exception as e:
            print(f"Error getting stock data for {ticker}: {e}")
            return pd.DataFrame()
        
    