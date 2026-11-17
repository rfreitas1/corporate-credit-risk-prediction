"""Reusable helpers for exploratory data analysis of the Polish dataset."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

from src.attributes import attribute_table
from src.data_processing import FEATURE_COLUMNS

# Colours (validated for colour-vision deficiency and contrast).
HEALTHY = "#2a78d6"   # blue
BANKRUPT = "#eb6834"  # orange
CLASS_COLORS = {0: HEALTHY, 1: BANKRUPT}
CLASS_LABELS = {0: "Healthy", 1: "Bankrupt"}
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#9c2a2a"]
)


def set_plot_style() -> None:
    """Apply a clean, consistent matplotlib style for all EDA figures."""
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlecolor": "#0b0b0b",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "xtick.labelcolor": "#52514e",
        "ytick.labelcolor": "#52514e",
        "legend.frameon": False,
        "font.size": 10,
        "figure.dpi": 110,
    })


def short(text: str, width: int) -> str:
    """Shorten a label to ``width`` characters at a word boundary, adding '…'."""
    if len(text) <= width:
        return text
    return text[: width - 1].rsplit(" ", 1)[0] + " …"


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of missing values per attribute, overall and by class.

    The ``ratio`` column (missing rate among bankrupt / among healthy) shows
    whether missingness itself carries information about the label.
    """
    is_missing = df[FEATURE_COLUMNS].isna()
    table = pd.DataFrame({
        "missing_%": 100 * is_missing.mean(),
        "missing_%_healthy": 100 * is_missing[df["class"] == 0].mean(),
        "missing_%_bankrupt": 100 * is_missing[df["class"] == 1].mean(),
    })
    table["ratio"] = table["missing_%_bankrupt"] / table["missing_%_healthy"]
    attrs = attribute_table()[["definition", "category"]]
    return table.join(attrs).sort_values("missing_%", ascending=False)


def univariate_auc(df: pd.DataFrame, feature: str) -> float:
    """ROC-AUC of a single attribute used directly as a score.

    AUC only depends on the *ranking* of values, so it is unaffected by the
    extreme outliers in these ratios. 0.5 = no signal; values below 0.5 mean
    *lower* values indicate bankruptcy. Missing rows are dropped.
    """
    mask = df[feature].notna()
    y = df.loc[mask, "class"]
    if y.nunique() < 2:
        return np.nan
    return roc_auc_score(y, df.loc[mask, feature])


def separation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Univariate AUC for every attribute, sorted by strength of signal.

    ``strength`` = |AUC - 0.5| * 2, from 0 (no signal) to 1 (perfect).
    ``direction`` says whether higher values point towards bankruptcy.
    """
    aucs = pd.Series({f: univariate_auc(df, f) for f in FEATURE_COLUMNS}, name="auc")
    table = aucs.to_frame()
    table["strength"] = (table["auc"] - 0.5).abs() * 2
    table["direction"] = np.where(table["auc"] > 0.5, "higher = riskier", "lower = riskier")
    attrs = attribute_table()[["definition", "category"]]
    return table.join(attrs).sort_values("strength", ascending=False)


def percentile_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Min, key percentiles and max per attribute, to expose extreme tails.

    ``tail_ratio`` = max |value| / |99th percentile|: how many times larger the
    most extreme value is than a "normal large" value.
    """
    q = df[FEATURE_COLUMNS].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).T
    q.columns = ["p1", "p25", "median", "p75", "p99"]
    q.insert(0, "min", df[FEATURE_COLUMNS].min())
    q["max"] = df[FEATURE_COLUMNS].max()
    extreme = q[["min", "max"]].abs().max(axis=1)
    typical = q[["p1", "p99"]].abs().max(axis=1).replace(0, np.nan)
    q["tail_ratio"] = extreme / typical
    return q.join(attribute_table()[["category"]])


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """List attribute pairs whose absolute correlation is at least ``threshold``."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("rho").reset_index()
    pairs.columns = ["attr_a", "attr_b", "rho"]
    pairs = pairs[pairs["rho"].abs() >= threshold]
    return pairs.reindex(pairs["rho"].abs().sort_values(ascending=False).index)
