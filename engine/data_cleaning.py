"""
Data cleaning pipeline for strategy return streams.
Detects and corrects level shifts, IQR-based outlier detection with graduated thresholds.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


def detect_and_correct_shift(
    returns: pd.DataFrame, threshold: float = 20.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect level-shifted strategies using cross-sectional MAD of medians.
    A strategy whose median deviates > threshold × cross-sectional MAD is corrected
    by subtracting its median.

    Parameters
    ----------
    returns : pd.DataFrame
        Strategy returns (columns = strategies, rows = observations).
    threshold : float
        Multiple of cross-sectional MAD beyond which a strategy is flagged.

    Returns
    -------
    corrected : pd.DataFrame
    diagnostics : dict with keys 'corrected' and 'skipped'
    """
    diagnostics = {"corrected": [], "skipped": []}
    if returns.shape[1] < 3:
        return returns.copy(), diagnostics

    medians = returns.median()
    median_of_medians = medians.median()
    cross_mad = (medians - median_of_medians).abs().median()

    if cross_mad < 1e-12:
        return returns.copy(), diagnostics

    corrected = returns.copy()
    for col in returns.columns:
        dev = abs(medians[col] - median_of_medians)
        ratio = dev / cross_mad
        if ratio > threshold:
            shift = medians[col]
            corrected[col] = returns[col] - shift
            diagnostics["corrected"].append(
                {"name": col, "shift": shift, "ratio": round(ratio, 1)}
            )

    return corrected, diagnostics


def clean_outliers(
    returns: pd.DataFrame, iqr_multiplier: float = 1.5, suppress_pct: float = 0.15
) -> Tuple[pd.DataFrame, Dict]:
    """
    IQR-based outlier detection with graduated thresholds.

    - > suppress_pct outliers: skip cleaning, flag as heavy-tailed
    - 5-15%: clean but flag as ambiguous
    - < 5%: clean silently

    Outliers are forward-filled from the previous clean value.

    Parameters
    ----------
    returns : pd.DataFrame
    iqr_multiplier : float
    suppress_pct : float

    Returns
    -------
    cleaned : pd.DataFrame
    diagnostics : dict[str, dict]
    """
    diagnostics = {}
    cleaned = returns.copy()

    for col in returns.columns:
        vals = returns[col].values
        n = len(vals)
        if n < 10:
            continue

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-15:
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        outlier_idx = np.where((vals < lower) | (vals > upper))[0]
        pct = len(outlier_idx) / n

        if pct > suppress_pct:
            diagnostics[col] = {
                "count": len(outlier_idx),
                "pct": pct,
                "suppressed": True,
            }
            continue

        if len(outlier_idx) == 0:
            continue

        # Forward-fill replacement
        col_cleaned = vals.copy()
        outlier_set = set(outlier_idx)
        for idx in outlier_idx:
            if idx == 0:
                next_clean = next(
                    (col_cleaned[j] for j in range(1, n) if j not in outlier_set), 0.0
                )
                col_cleaned[idx] = next_clean
            else:
                col_cleaned[idx] = col_cleaned[idx - 1]

        cleaned[col] = col_cleaned
        diagnostics[col] = {
            "count": len(outlier_idx),
            "pct": pct,
            "suppressed": False,
            "ambiguous": pct >= 0.05,
        }

    return cleaned, diagnostics
