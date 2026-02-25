# data_loader.py

import yfinance as yf
import pandas as pd

def load_data(symbol: str = "TTE.PA",
              start: str = "2021-01-01", #par exemple
              end: str = None) -> pd.DataFrame:
    
    df = yf.download(symbol, start=start, end=end)
    
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.dropna(inplace=True)
        
    return df
