"""
plot.py

Visualisation finale sous forme de graphique (points bas du support oblique haussier dominant, support projeté sur période de test, 1e cassure éventuelle du support par le bas)

"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_support_analysis(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    fractals_train: pd.DataFrame,
    inliers_train: pd.DataFrame,
    support_test: pd.Series,
    break_series: pd.Series = None,
    title: str = "Support oblique haussier et détection éventuelle de la 1e cassure par le bas du support",
):
    """
    Plots:
        - dataset action (train + test)
        - points bas estimés par fractales de Bill Williams
        - points bas significatifs estimés par RANSAC
        - Support oblique haussier projeté (période de test)
        - Break par le bas du support confirmé

    Parameters
    ----------
    df_train : pd.DataFrame
    df_test : pd.DataFrame
    fractals_train : pd.DataFrame
    inliers_train : pd.DataFrame
    support_test : pd.Series or np.ndarray
    break_series : pd.Series 
    title : str
    """

    plt.figure(figsize=(14, 7))

    # Combine price series
    df_all = pd.concat([df_train, df_test])

    # Plot price
    plt.plot(df_all.index, df_all["Close"], label="Cours de clôture action")

    # Plot fractals
    plt.scatter(
        fractals_train.index,
        fractals_train["Low"],
        marker="o",
        color="green",
        s=25,
        label="Points bas (estimation par Fractales de Bill Williams)",
    )

    # Plot inliers
    plt.scatter(
        inliers_train.index,
        inliers_train["Low"],
        marker="o",
        color="orange",
        s=50,
        label="Points bas signifiants (méthode RANSAC)",
    )

    # Plot projected support (test only)
    support_series = pd.Series(support_test, index=df_test.index)

    plt.plot(
        support_series.index,
        support_series.values,
        linestyle="--",
        label="Support oblique haussier projeté (Test)",
    )

    # Mark first confirmed breakdown
    if break_series is not None and break_series.any():
        first_break_date = break_series[break_series].index[0]
        first_break_price = df_test.loc[first_break_date, "Close"]

        plt.scatter(
            first_break_date,
            first_break_price,
            marker="X",
            color="red",
            s=160,
            label="1er break confirmé du support",
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cours")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()