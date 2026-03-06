# Multi-Strategy Allocation Engine

An end-to-end portfolio allocation system that ingests a universe of strategy return streams, cleans the data, scores and ranks every strategy, selects a portfolio under correlation constraints, and optimises position weights using one of 7 allocation methods, all under a hard 10% single-position cap.

**[Live Dashboard →](https://angel4angelov-glitch.github.io/Multi-Strategy-Allocation-Engine)**

---

## Pipeline

```
CSV Upload → Data Cleaning → Metric Computation → Grading → Autopilot Selection → Weight Optimisation → Portfolio
```

### 1. Data Cleaning

- **Level-shift detection**: computes cross-sectional MAD of strategy medians; any strategy whose median deviates >20× MAD is corrected by subtracting the shift. Catches data artefacts (e.g. a strategy offset by +8.0).
- **IQR outlier detection**: 1.5× IQR fences with graduated thresholds — >15% outliers suppresses cleaning (flags as heavy-tailed), 5–15% cleans but flags as ambiguous, <5% cleans silently. Forward-fill replacement preserves time alignment.
- **NaN/Inf handling**: non-finite values replaced with 0 to keep all strategies temporally aligned.

### 2. Metric Computation

Per-strategy: Sharpe, Sortino, Calmar, CVaR (5%), max drawdown, win rate, momentum score (recency-weighted), skewness, excess kurtosis, lag-1 autocorrelation, full ACF (lags 1–20), rolling Sharpe/volatility/skewness, and regime detection (last-quarter Sharpe vs full-sample).

Directional consistency inferred from sub-period return decomposition — splits the return series into sections and checks whether all section means share the same sign. Strategies labelled LONG, SHORT, or UNCLEAR.

### 3. Grading & Autopilot Selection

- **Percentile grading**: strategies ranked by a configurable metric (Sharpe, Sortino, Calmar, or momentum). Top 20% = STRONG, top 50% = MODERATE, rest = WEAK. Absolute quality gates prevent garbage strategies from being graded STRONG regardless of relative rank (e.g. non-positive Sharpe can never be STRONG).
- **Adaptive selection gates**: starts with STRONG + MODERATE, relaxes progressively if fewer than 10 strategies qualify.
- **Correlation filter**: greedy selection with pairwise correlation cap (0.7 in Aggressive, 0.5 in Defensive mode) to limit redundant exposure.
- **Trend penalty**: strategies with decaying rolling Sharpe (trend = "down") are penalised 50% in rank score.
- **Direction-aware ranking**: SHORT strategies have their ranking metric negated so negative-return strategies are correctly signed rather than ranked as if they were LONG.

### 4. Allocation Methods

| Method | Description |
|---|---|
| **Equal Weight** | 1/N baseline |
| **Inverse Volatility** | Weight ∝ 1/σ (risk parity) |
| **Max Sharpe** | Gradient descent on Sharpe ratio with Ledoit-Wolf covariance |
| **Min Variance** | Gradient descent minimising portfolio variance |
| **Min CVaR** | Simulation-based gradient descent minimising Expected Shortfall (worst 5%) |
| **Max Sortino** | Simulation-based gradient descent maximising Sortino ratio |
| **HRP** | Hierarchical Risk Parity (López de Prado) — single-linkage clustering + recursive bisection |

All methods share:
- **Ledoit-Wolf shrinkage** for covariance estimation (scaled identity target)
- **10% position cap** enforced via iterative clamp-and-redistribute
- **Turnover penalty** (λ = 0.1) penalising deviation from previous weights
- **Fallback**: if the optimiser produces weights that violate constraints or degrade Sharpe vs equal-weight, it reverts to equal-weight automatically

### 5. Modes

- **Aggressive**: ranks by Sharpe (or selected metric), correlation cap at 0.7
- **Defensive**: ranks by inverse volatility, tighter correlation cap at 0.5

### 6. Dashboard Features

- **Strategy tearsheet**: cumulative return + rolling Sharpe overlay, rolling vol + rolling skewness, regime indicator, autocorrelation plot, top correlations
- **Risk flags**: automated flagging for deep drawdowns, fat tails (kurtosis), severe CVaR, negative skew, rolling Sharpe collapse, directional inconsistency, data insufficiency
- **Correlation heatmap**: full cross-strategy correlation matrix
- **Portfolio panel**: portfolio-level Sharpe, return, volatility, max drawdown, and CVaR
- **Veto mode**: manual override to exclude or force-short individual strategies
- **Submission tracker**: logs portfolio snapshots with timestamps for audit
- **Signals tab**: feature engineering (momentum, vol, Sharpe, z-score, skewness, autocorrelation at multiple windows) with Information Coefficient analysis against a selected target strategy
- **Session persistence**: crash recovery via state serialisation
- **Delta detection**: flags changes between data releases (new/missing strategies, row count changes)

---

## Test Scenarios

20 pre-loaded datasets stress-test the allocator:

| Category | Scenarios |
|---|---|
| **Base** | High Sharpe, Mixed Realistic |
| **Stress** | Extreme Vol, Fat Tails (Cauchy), Outlier Bomb, Negative Skew Crashes, Regime Break, GARCH Vol Clustering |
| **Edge** | 50-obs history, 200-obs history, Level-Shifted, Dead Zones / Late Activation, 70% Sparse, NaN/Inf Contaminated |
| **Traps** | All Negative Returns, Highly Correlated, Momentum Trap, Mean Reverting, Asymmetric Positive Skew |
| **Combined** | Kitchen Sink (all pathologies) |

---

## Repo Structure

```
├── index.html              # Self-contained dashboard (Plotly, no backend)
├── engine/                 # Python implementation of core logic
│   ├── __init__.py
│   ├── allocator.py        # 7 allocation methods + Ledoit-Wolf + constraints
│   ├── metrics.py          # Strategy-level metric computation
│   └── data_cleaning.py    # Shift detection + IQR outlier cleaning
└── data/                   # 20 test scenario CSVs
    ├── 01_easy_high_sharpe.csv
    ├── ...
    └── 20_kitchen_sink.csv
```

## Stack

- **Python** (NumPy, Pandas) — allocation engine, metrics, data cleaning
- **Plotly** — interactive charts
- Dashboard runs entirely in the browser, no backend required

## Usage

Upload any CSV with columns named `strat_0`, `strat_1`, etc. containing period returns, or select a test scenario from the dropdown.
