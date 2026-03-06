"""
Portfolio allocation methods for multi-strategy selection.
Implements 7 methods: Equal Weight, Inverse Vol (Risk Parity), Max Sharpe,
Min Variance, Min CVaR, Max Sortino, and Hierarchical Risk Parity (HRP).

All methods enforce a 10% single-position cap and normalise to sum-to-one.
"""

import numpy as np
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────
# COVARIANCE ESTIMATION
# ──────────────────────────────────────────────────────────────────────────

def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage estimator for covariance matrix.
    Target: scaled identity μI where μ = tr(S)/N.

    Parameters
    ----------
    returns : np.ndarray
        (N, T) matrix — N strategies, T observations.

    Returns
    -------
    np.ndarray : (N, N) shrunk covariance matrix.
    """
    N, T = returns.shape
    means = returns.mean(axis=1, keepdims=True)
    X = returns - means
    S = X @ X.T / T

    mu_target = np.trace(S) / N

    # δ² = ||S - μI||_F² / N²
    target = mu_target * np.eye(N)
    delta2 = np.sum((S - target) ** 2) / (N * N)

    # β̄²
    beta2 = 0.0
    for t in range(T):
        xt = X[:, t : t + 1]
        outer = xt @ xt.T
        beta2 += np.sum((outer - S) ** 2)
    beta2 /= T * T * N * N

    alpha = np.clip(beta2 / (delta2 + 1e-14), 0.0, 1.0)
    return (1 - alpha) * S + alpha * target


# ──────────────────────────────────────────────────────────────────────────
# CONSTRAINT ENFORCEMENT
# ──────────────────────────────────────────────────────────────────────────

def enforce_constraints(weights: np.ndarray, cap: float = 0.10) -> np.ndarray:
    """
    Enforce long-only, position-cap, and sum-to-one constraints.
    Iterative clamp-and-redistribute.
    """
    w = weights.copy()
    n = len(w)

    for _ in range(n + 5):
        w = np.maximum(w, 0.0)
        excess = np.sum(np.maximum(w - cap, 0.0))
        w = np.minimum(w, cap)

        if excess < 1e-14:
            break

        free_mask = w < cap
        free_sum = w[free_mask].sum()
        if free_sum < 1e-14:
            break
        w[free_mask] *= (free_sum + excess) / free_sum

    w = np.minimum(w, cap)
    s = w.sum()
    if s > 1e-10:
        w /= s
    w = np.minimum(w, cap)
    return w


# ──────────────────────────────────────────────────────────────────────────
# ALLOCATION METHODS
# ──────────────────────────────────────────────────────────────────────────

def equal_weight(n: int) -> np.ndarray:
    """Equal weight allocation."""
    return np.full(n, 1.0 / n)


def inverse_vol(volatilities: np.ndarray) -> np.ndarray:
    """Risk parity via inverse volatility."""
    inv = 1.0 / (volatilities + 1e-10)
    return enforce_constraints(inv / inv.sum())


def optimize_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray,
    objective: str = "maxsharpe",
    lr: float = 0.002,
    max_iter: int = 800,
    turnover_penalty: float = 0.1,
) -> np.ndarray:
    """
    Gradient-based optimiser for Max Sharpe or Min Variance.

    Parameters
    ----------
    mu : np.ndarray (N,)
    sigma : np.ndarray (N, N)
    w_prev : np.ndarray (N,) — previous weights for turnover penalty
    objective : 'maxsharpe' or 'minvar'
    """
    n = len(mu)
    w = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        Sw = sigma @ w
        wSw = w @ Sw
        sig_p = np.sqrt(max(wSw, 1e-14))

        grad = np.zeros(n)
        if objective == "maxsharpe":
            wmu = w @ mu
            grad = (mu * sig_p - wmu * Sw / sig_p) / (sig_p * sig_p)
        else:  # minvar
            grad = -2.0 * Sw

        # Turnover penalty
        grad -= 2.0 * turnover_penalty * (w - w_prev)

        w += lr * grad

        # Project onto feasible set
        for _ in range(15):
            w = np.clip(w, 0.0, 0.1)
            s = w.sum()
            if s > 1e-10:
                w /= s
            if np.all(w <= 0.1001):
                break

    return enforce_constraints(w)


def optimize_cvar_sortino(
    ret_matrix: np.ndarray,
    w_prev: np.ndarray,
    objective: str = "mincvar",
    lr: float = 0.001,
    max_iter: int = 600,
    turnover_penalty: float = 0.1,
) -> np.ndarray:
    """
    Simulation-based gradient descent for Min CVaR or Max Sortino.

    Parameters
    ----------
    ret_matrix : np.ndarray (N, T)
    w_prev : np.ndarray (N,)
    objective : 'mincvar' or 'maxsortino'
    """
    N, T = ret_matrix.shape
    cvar_n = max(1, int(T * 0.05))
    w = np.full(N, 1.0 / N)

    for _ in range(max_iter):
        port_rets = w @ ret_matrix

        grad = np.zeros(N)

        if objective == "mincvar":
            worst_idx = np.argsort(port_rets)[:cvar_n]
            for t in worst_idx:
                grad -= ret_matrix[:, t] / cvar_n
            grad = -grad
        else:  # maxsortino
            port_mean = port_rets.mean()
            neg_mask = port_rets < 0
            neg_rets = port_rets[neg_mask]
            n_neg = len(neg_rets)
            down_var = (neg_rets ** 2).mean() if n_neg > 0 else 1e-10
            down_dev = np.sqrt(max(down_var, 1e-14))

            for i in range(N):
                d_mean = ret_matrix[i].mean()
                d_down_var = 0.0
                if n_neg > 0:
                    neg_idx = np.where(neg_mask)[0]
                    d_down_var = 2.0 * (port_rets[neg_idx] * ret_matrix[i, neg_idx]).sum() / n_neg
                d_down_dev = d_down_var / (2.0 * down_dev)
                grad[i] = (d_mean * down_dev - port_mean * d_down_dev) / (down_dev ** 2)

        grad -= 2.0 * turnover_penalty * (w - w_prev)
        w += lr * grad

        for _ in range(15):
            w = np.clip(w, 0.0, 0.1)
            s = w.sum()
            if s > 1e-10:
                w /= s
            if np.all(w <= 0.1001):
                break

    return enforce_constraints(w)


# ──────────────────────────────────────────────────────────────────────────
# HIERARCHICAL RISK PARITY (López de Prado)
# ──────────────────────────────────────────────────────────────────────────

def _correlation_distance(ret_matrix: np.ndarray) -> np.ndarray:
    """Compute correlation distance matrix: d = sqrt(0.5 * (1 - corr))."""
    corr = np.corrcoef(ret_matrix)
    corr = np.clip(corr, -1.0, 1.0)
    return np.sqrt(0.5 * (1.0 - corr))


def _single_linkage_order(dist: np.ndarray) -> List[int]:
    """Single-linkage agglomerative clustering, returns leaf order."""
    n = dist.shape[0]
    active = set(range(n))
    members = {i: [i] for i in range(n)}
    children = {}
    next_node = n

    # Build cluster distance map
    cdist = {}
    for i in range(n):
        for j in range(i + 1, n):
            cdist[(i, j)] = dist[i, j]

    def key(a, b):
        return (min(a, b), max(a, b))

    while len(active) > 1:
        # Find closest pair
        min_d, min_i, min_j = float("inf"), -1, -1
        arr = sorted(active)
        for a_idx in range(len(arr)):
            for b_idx in range(a_idx + 1, len(arr)):
                d = cdist.get(key(arr[a_idx], arr[b_idx]), float("inf"))
                if d < min_d:
                    min_d, min_i, min_j = d, arr[a_idx], arr[b_idx]

        new_node = next_node
        next_node += 1
        children[new_node] = (min_i, min_j)
        members[new_node] = members[min_i] + members[min_j]

        for k in active:
            if k in (min_i, min_j):
                continue
            d = min(
                cdist.get(key(min_i, k), float("inf")),
                cdist.get(key(min_j, k), float("inf")),
            )
            cdist[key(new_node, k)] = d

        active.discard(min_i)
        active.discard(min_j)
        active.add(new_node)

    root = active.pop()

    def get_order(node):
        if node < n:
            return [node]
        left, right = children[node]
        return get_order(left) + get_order(right)

    return get_order(root)


def _hrp_bisect(
    w: np.ndarray, order: List[int], variances: np.ndarray
) -> None:
    """Recursive bisection: allocate inverse-variance between halves."""
    if len(order) <= 1:
        return

    mid = len(order) // 2
    left, right = order[:mid], order[mid:]

    left_var = np.mean([variances[i] for i in left])
    right_var = np.mean([variances[i] for i in right])
    alpha = 1.0 - left_var / (left_var + right_var)

    for i in left:
        w[i] *= alpha
    for i in right:
        w[i] *= (1.0 - alpha)

    _hrp_bisect(w, left, variances)
    _hrp_bisect(w, right, variances)


def hrp_weights(ret_matrix: np.ndarray) -> np.ndarray:
    """
    Hierarchical Risk Parity (López de Prado).

    Parameters
    ----------
    ret_matrix : np.ndarray (N, T)

    Returns
    -------
    np.ndarray (N,) — weights summing to 1, capped at 10%.
    """
    N, T = ret_matrix.shape
    if N <= 1:
        return np.array([1.0])

    dist = _correlation_distance(ret_matrix)
    order = _single_linkage_order(dist)
    variances = ret_matrix.var(axis=1, ddof=1)

    w = np.ones(N)
    _hrp_bisect(w, order, variances)
    return enforce_constraints(w / w.sum())


# ──────────────────────────────────────────────────────────────────────────
# UNIFIED ALLOCATOR
# ──────────────────────────────────────────────────────────────────────────

def allocate(
    ret_matrix: np.ndarray,
    method: str = "hrp",
    w_prev: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Unified allocation interface.

    Parameters
    ----------
    ret_matrix : np.ndarray (N, T) — strategy returns
    method : str — one of 'equal', 'invvol', 'maxsharpe', 'minvar',
                    'mincvar', 'maxsortino', 'hrp'
    w_prev : optional previous weights for turnover penalty

    Returns
    -------
    np.ndarray (N,) — portfolio weights
    """
    N, T = ret_matrix.shape
    if w_prev is None:
        w_prev = np.full(N, 1.0 / N)

    if method == "equal":
        return equal_weight(N)

    if method == "invvol":
        vols = ret_matrix.std(axis=1, ddof=1)
        return inverse_vol(vols)

    if method == "hrp":
        return hrp_weights(ret_matrix)

    if method in ("mincvar", "maxsortino"):
        return optimize_cvar_sortino(ret_matrix, w_prev, objective=method)

    # maxsharpe or minvar
    sigma = ledoit_wolf_shrinkage(ret_matrix)
    mu = ret_matrix.mean(axis=1)
    return optimize_weights(mu, sigma, w_prev, objective=method)
