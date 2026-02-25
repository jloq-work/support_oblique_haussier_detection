"""
ransac_support.py

Détection du support oblique haussier principal via l'algorithme RANSAC.

residual threshold : ajusté de la volatilité sur la période d'entrainement via le calcul de l'ATR (residual threshold=ATR choix fait pour ce pipeline).

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor, LinearRegression


class AscendingSupportRANSAC:
    """
    Utilisation de RANSAC pour déterminer le support oblique haussier dominant à partir des points bas estimés par fractales de Bill Williams.

    Parameters
    ----------
    min_samples_ratio : float
        Minimum fraction of samples required by RANSAC.
    min_inliers : int
        Minimum number of inliers required to validate support.
    min_inlier_ratio : float
        Minimum fraction of fractals required as inliers.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        min_samples_ratio: float = 0.2,
        min_inliers: int = 10,
        min_inlier_ratio: float = 0.15,
        random_state: int = 42,
    ):
        self.min_samples_ratio = min_samples_ratio
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.random_state = random_state

        self.model_ = None
        self.slope_ = None
        self.intercept_ = None
        self.inlier_mask_ = None
        self.valid_support_ = False

    def fit(self, df_fractals: pd.DataFrame, atr_mean: float):
        """
        Fits RANSAC on fractal lows.

        Parameters
        ----------
        df_fractals : pd.DataFrame
            table avec les points bas estimés par fractales ('Low' column).
        atr_mean : float
            ATR moyen calculé sur la période d'entraînement.
        """

        if df_fractals.empty:
            raise ValueError("table des points bas estimés par fractales vide")

        # regression inputs
        X = df_fractals.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y = df_fractals["Low"].values

        residual_threshold = atr_mean  

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=self.min_samples_ratio,
            residual_threshold=residual_threshold,
            random_state=self.random_state,
        )

        ransac.fit(X, y)

        self.model_ = ransac
        self.slope_ = ransac.estimator_.coef_[0]
        self.intercept_ = ransac.estimator_.intercept_
        self.inlier_mask_ = ransac.inlier_mask_

        self._validate_support(len(df_fractals))

        return self

    def _validate_support(self, total_points: int):
        """
        Validation du support via coefficient de la pente (positif) et ratio minimal de points bas signifiants pour le tracer.
        """

        if self.slope_ <= 0:
            self.valid_support_ = False
            return

        n_inliers = np.sum(self.inlier_mask_)
        inlier_ratio = n_inliers / total_points

        if n_inliers < self.min_inliers:
            self.valid_support_ = False
            return

        if inlier_ratio < self.min_inlier_ratio:
            self.valid_support_ = False
            return

        self.valid_support_ = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prédiction du support.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex.

        Returns
        -------
        np.ndarray
            Predicted support values.
        """

        if not self.valid_support_:
            raise RuntimeError("Pas de support oblique haussier détecté")

        X = df.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        return self.model_.predict(X)

    def get_inliers(self, df_fractals: pd.DataFrame) -> pd.DataFrame:
        """
        inlier fractal points.
        """

        if self.inlier_mask_ is None:
            raise RuntimeError("Modele non ajusté")

        return df_fractals[self.inlier_mask_]