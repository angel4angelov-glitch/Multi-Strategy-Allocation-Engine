"""Multi-Strategy Allocation Engine."""

from .data_cleaning import detect_and_correct_shift, clean_outliers
from .metrics import compute_metrics, StrategyMetrics
from .allocator import allocate, ledoit_wolf_shrinkage, hrp_weights

__all__ = [
    "detect_and_correct_shift",
    "clean_outliers",
    "compute_metrics",
    "StrategyMetrics",
    "allocate",
    "ledoit_wolf_shrinkage",
    "hrp_weights",
]
