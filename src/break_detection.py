"""
break_detection.py

Détection d'une cassure par le bas du support projeté (prédit par RANSAC) sur le dataset de test.

3 conditions de confirmation du break:
    1. Cassure du support et d'au moins 1% en distance par rapport au support
    2. Cloture sous le support
    3. Conditions 1 et 2 sur plusieurs séances boursières consécutives

"""

import numpy as np
import pandas as pd


class SupportBreakDetector:
    """
    Parameters
    ----------
    penetration_threshold : float
        Distance minimale par rapport au support (e.g., 0.01 for 1%).
    confirmation_days : int
        Nombre de séances boursières consécutives pour confirmer le break de support.
    """

    def __init__(
        self,
        penetration_threshold: float = 0.01,
        confirmation_days: int = 3,
    ):
        self.penetration_threshold = penetration_threshold
        self.confirmation_days = confirmation_days

        self.break_series_ = None
        self.first_break_date_ = None

    def detect(self, df_test: pd.DataFrame, support_values: np.ndarray):
        """
        Détection des ruptures de support par le bas.

        Parameters
        ----------
        df_test : pd.DataFrame
            'Low' and 'Close' columns.
        support_values : np.ndarray
            les points du support oblique haussier prédit par RANSAC sur la période de test.

        Returns
        -------
        pd.Series
            Boolean series indicating confirmed breakdown days.
        """

        if len(df_test) != len(support_values):
            raise ValueError("pas la même longueur support projeté et dataset de test")

        support = pd.Series(support_values, index=df_test.index)

        # 1e condition: break du support et cours en deça du support (au moins 1%)
        penetration = (support - df_test["Low"]) / support
        penetration_condition = penetration > self.penetration_threshold

        # 2e condition: clôture sous le support
        close_condition = df_test["Close"] < support

        # clôture sous le support (et au moins 1% dessous)
        raw_break = penetration_condition & close_condition

        # 3e condition: reste sous le support 3 séances boursières consécutives
        confirmed_break = (
            raw_break.rolling(self.confirmation_days)
            .sum()
            .eq(self.confirmation_days)
        )

        self.break_series_ = confirmed_break

        # 1er break de support confirmé
        if confirmed_break.any():
            self.first_break_date_ = confirmed_break[confirmed_break].index[0]
        else:
            self.first_break_date_ = None

        return confirmed_break

    def get_first_break_date(self):
        """
        Retourne le 1er break de support qui confirmé.
        """

        return self.first_break_date_