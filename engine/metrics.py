"""
Strategy-level metric computation engine.
Computes Sharpe, Sortino, CVaR, drawdown, momentum, regime detection,
rolling statistics, autocorrelation, and directional consistency.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StrategyMetrics:
    name: str
    n: int
    mean: float
    std: float
    sharpe: float
    total_return: float
    max_drawdown: float
    calmar: float
    win_rate: float
    sortino: float
    cvar_5pct: float
    momentum_score: float
    autocorr_lag1: float
    skewness: float
    excess_kurtosis: float
    direction: str  # "LONG", "SHORT", "UNCLEAR"
    dir_consistent: bool
    recent_sharpe: float
    trend: str  # "up", "down", "flat"
    sufficient: bool
    cum_returns: np.ndarray = field(repr=False)
    drawdown_series: np.ndarray = field(repr=False)
    rolling_sharpe: np.ndarray = field(repr=False)
    rolling_vol: np.ndarray = field(repr=False)
    acf: np.ndarray = field(repr=False)
    returns: np.ndarray = field(repr=False)


def compute_metrics(returns: np.ndarray, name: str) -> Optional[StrategyMetrics]:
    """
    Compute full metric suite for a single strategy return series.

    Parameters
    ----------
    returns : np.ndarray
        1D array of period returns.
    name : str
        Strategy identifier.

    Returns
    -------
    StrategyMetrics or None if insufficient data.
    """
    clean = np.where(np.isfinite(returns), returns, 0.0)
    n = len(clean)
    if n < 3:
        return None

    mean = clean.mean()
    std = clean.std(ddof=1)
    sharpe = mean / std if std > 0 else 0.0
    total_return = clean.sum()

    # Cumulative returns and drawdown
    cum_ret = np.concatenate([[0.0], np.cumsum(clean)])
    peak = np.maximum.accumulate(cum_ret)
    dd_series = peak - cum_ret
    max_dd = dd_series.max()
    calmar = total_return / max_dd if max_dd > 0 else 0.0

    # Win rate
    win_rate = (clean > 0).sum() / n

    # Sortino
    neg = clean[clean < 0]
    down_dev = np.sqrt((neg ** 2).mean()) if len(neg) > 1 else std
    sortino = mean / down_dev if down_dev > 0 else 0.0

    # CVaR at 5%
    sorted_rets = np.sort(clean)
    cvar_n = max(1, int(n * 0.05))
    cvar_5pct = sorted_rets[:cvar_n].mean()

    # Momentum score (last 30% weighted 2x)
    cut = int(n * 0.7)
    early_mean = clean[:cut].mean() if cut > 0 else 0.0
    late_mean = clean[cut:].mean() if (n - cut) > 0 else 0.0
    momentum_score = early_mean * 0.33 + late_mean * 0.67

    # Autocorrelation lag-1
    demeaned = clean - mean
    den = (demeaned ** 2).sum()
    autocorr = (demeaned[1:] * demeaned[:-1]).sum() / den if den > 0 else 0.0

    # ACF lags 1-20
    max_lag = min(20, n // 3)
    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = (demeaned[lag:] * demeaned[:-lag]).sum() / den if den > 0 else 0.0

    # Directional consistency
    n_sections = min(6, max(3, n // 10))
    section_len = n // n_sections
    section_means = []
    for i in range(n_sections):
        start = i * section_len
        end = (i + 1) * section_len if i < n_sections - 1 else n
        section_means.append(clean[start:end].mean())
    section_means = np.array(section_means)
    all_pos = (section_means > 0).all()
    all_neg = (section_means < 0).all()
    direction = "LONG" if all_pos else ("SHORT" if all_neg else "UNCLEAR")
    dir_consistent = all_pos or all_neg

    # Rolling Sharpe
    roll_win = max(20, min(60, int(n * 0.3)))
    rolling_sharpe = np.zeros(n - roll_win + 1)
    rolling_vol = np.zeros(n - roll_win + 1)
    for i in range(roll_win - 1, n):
        window = clean[i - roll_win + 1 : i + 1]
        m = window.mean()
        s = window.std(ddof=1)
        rolling_sharpe[i - roll_win + 1] = m / s if s > 0 else 0.0
        rolling_vol[i - roll_win + 1] = s

    # Skewness and excess kurtosis
    if std > 0:
        skewness = ((clean - mean) / std).mean() ** 3
        # Proper skewness: E[(X-mu)^3] / sigma^3
        skewness = (((clean - mean) / std) ** 3).mean()
        excess_kurtosis = (((clean - mean) / std) ** 4).mean() - 3.0
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    # Regime detection: last 25% Sharpe vs full
    rc_start = int(n * 0.75)
    rc_slice = clean[rc_start:]
    rc_mean = rc_slice.mean()
    rc_std = rc_slice.std(ddof=1) if len(rc_slice) > 1 else 0.0
    recent_sharpe = rc_mean / rc_std if rc_std > 0 else 0.0
    sharpe_diff = recent_sharpe - sharpe
    sharpe_scale = max(abs(sharpe), 0.05)
    trend = (
        "up" if sharpe_diff > sharpe_scale * 0.3
        else ("down" if sharpe_diff < -sharpe_scale * 0.3 else "flat")
    )

    return StrategyMetrics(
        name=name, n=n, mean=mean, std=std, sharpe=sharpe,
        total_return=total_return, max_drawdown=max_dd, calmar=calmar,
        win_rate=win_rate, sortino=sortino, cvar_5pct=cvar_5pct,
        momentum_score=momentum_score, autocorr_lag1=autocorr,
        skewness=skewness, excess_kurtosis=excess_kurtosis,
        direction=direction, dir_consistent=dir_consistent,
        recent_sharpe=recent_sharpe, trend=trend, sufficient=n >= 30,
        cum_returns=cum_ret, drawdown_series=dd_series,
        rolling_sharpe=rolling_sharpe, rolling_vol=rolling_vol,
        acf=acf, returns=clean,
    )
