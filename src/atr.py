"""
atr.py

Calcul du average true range: 
The Current Period High minus (-) Current Period Low
The Absolute Value (abs) of the Current Period High minus (-) The Previous Period Close
The Absolute Value (abs) of the Current Period Low minus (-) The Previous Period Close

La période de rétrospective à utiliser pour l'ATR est subjective, cependant 14 jours est la valeur la plus commune

"""

import pandas as pd

def average_true_range(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    return atr