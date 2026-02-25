"""
fractal.py

Détection des points bas du cours de l'action suivant la méthode des fractales de Bill Williams (version 5 bars)

"""

import pandas as pd

def fractal_5bars(df: pd.DataFrame) -> pd.DataFrame:
    """
   Point i est fractal si :
        Low[i] < Low[i-1]
        Low[i] < Low[i-2]
        Low[i] < Low[i+1]
        Low[i] < Low[i+2]
    Méthode Bill Williams fractales 5bars
    """

    low = df['Low']

    cond = (
        (low < low.shift(1)) &
        (low < low.shift(2)) &
        (low < low.shift(-1)) &
        (low < low.shift(-2))
    )

    return df.loc[cond]