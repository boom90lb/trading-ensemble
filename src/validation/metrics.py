# src/validation/metrics.py
"""Performance and overfitting metrics.

Implements the López de Prado / Bailey overfitting-adjusted metrics needed
for honest backtest reporting:

  * `probabilistic_sharpe_ratio(returns, sr_benchmark=0)` — Bailey & LdP 2012.
    Probability that the true Sharpe exceeds a benchmark given the observed
    sample, adjusting for skewness and kurtosis of returns.

  * `calmar_ratio(annualized_return, max_drawdown)` — annualized return over
    |max drawdown|. Uses the project convention that `max_drawdown` is stored
    as a positive number (matches `trading.py`'s `results['drawdown'].max()`).

  * `probability_backtest_overfitting(returns_matrix, n_splits=10)` — PBO
    via Combinatorially Symmetric Cross-Validation (Bailey/Borwein/LdP/Zhu
    2015). Returns the probability that the in-sample best strategy is
    out-of-sample below-median. Undefined for <2 strategies.

  * `deflated_sharpe_ratio(returns, trial_sharpes)` — Bailey & López de
    Prado (2014). PSR with the benchmark set to the *expected maximum*
    Sharpe under the null of no skill across N trials (`expected_max_sharpe`
    via the False Strategy Theorem). This is the half of M3 that needed a
    real hyperparameter-grid trial count (Phase 2.7) before it could be
    computed honestly — the trial Sharpes are the per-config OOS Sharpes
    from the sweep, and N is the (honestly counted) grid size.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, pd.Series]

# Euler–Mascheroni constant, used by the False Strategy Theorem expected-max.
EULER_GAMMA = 0.5772156649015329


def _to_array(returns: ArrayLike) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


_STD_FLOOR = 1e-15  # below this we treat the sample as degenerate; pandas/numpy
# variance is not exactly 0 on constant float arrays due to summation roundoff
# (e.g. np.full(100, 0.01).std(ddof=1) ≈ 1.7e-18).


def periodic_sharpe(returns: ArrayLike) -> float:
    """Sample Sharpe at the input frequency (no annualization).

    Returns NaN if there are <2 finite observations or the sample stdev is
    effectively zero.
    """
    arr = _to_array(returns)
    if arr.size < 2:
        return float("nan")
    std = arr.std(ddof=1)
    if not np.isfinite(std) or std < _STD_FLOOR:
        return float("nan")
    return float(arr.mean() / std)


def _sharpe_and_moments(returns: ArrayLike):
    """Return ``(sr, n, g3, g4)`` for a returns sample, or None if degenerate.

    ``sr`` is the periodic sample Sharpe; ``g3``/``g4`` are Pearson skewness
    and (non-excess) kurtosis using the population stdev. Degenerate means
    n<3 or effectively-zero stdev.
    """
    arr = _to_array(returns)
    n = arr.size
    if n < 3:
        return None

    std = arr.std(ddof=1)
    if not np.isfinite(std) or std < _STD_FLOOR:
        return None

    sr = arr.mean() / std
    # Pearson skewness and kurtosis (kurt = 3 for normal, not excess).
    mu3 = np.mean((arr - arr.mean()) ** 3)
    mu4 = np.mean((arr - arr.mean()) ** 4)
    sigma = arr.std(ddof=0)  # population stdev for moment ratios
    if sigma < _STD_FLOOR:
        return None
    g3 = mu3 / sigma**3
    g4 = mu4 / sigma**4
    return float(sr), int(n), float(g3), float(g4)


def _psr_from_stats(sr: float, n: int, sr_benchmark: float, g3: float, g4: float) -> float:
    """PSR core given the selected stats; shared by PSR and DSR.

    Returns NaN if the variance-of-Sharpe denominator is non-positive (can
    happen for very high SR with positive skew).
    """
    denom_sq = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if denom_sq <= 0.0 or not np.isfinite(denom_sq):
        logger.debug(
            f"PSR denominator non-positive (SR={sr:.3f}, skew={g3:.3f}, kurt={g4:.3f}); returning NaN"
        )
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def probabilistic_sharpe_ratio(
    returns: ArrayLike,
    sr_benchmark: float = 0.0,
) -> float:
    """Bailey & López de Prado (2012) Probabilistic Sharpe Ratio.

    Returns the probability that the true (population) Sharpe exceeds
    `sr_benchmark`, given the observed sample's mean, stdev, skewness, and
    kurtosis. `sr_benchmark` is a *periodic* Sharpe at the same frequency
    as `returns` (e.g. daily). The default 0 tests "does this strategy
    have any edge at all".

    Returns NaN if the sample is degenerate (n<3, std=0, or denominator
    non-positive — which can happen for very high SR with positive skew).
    """
    stats = _sharpe_and_moments(returns)
    if stats is None:
        return float("nan")
    sr, n, g3, g4 = stats
    return _psr_from_stats(sr, n, sr_benchmark, g3, g4)


def expected_max_sharpe(trial_sharpes: ArrayLike) -> float:
    """Expected maximum Sharpe under the null of no skill across N trials.

    Bailey & López de Prado's "False Strategy Theorem": when N independent
    strategies each have a true Sharpe of 0, the expected *maximum* sample
    Sharpe is

        E[max] ≈ σ_SR · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

    where σ_SR is the cross-trial dispersion of Sharpes and γ is the
    Euler–Mascheroni constant. This is the benchmark the selected strategy's
    Sharpe must beat to be credible — picking the best of N trials inflates
    the apparent Sharpe by roughly this much for free.

    `trial_sharpes` are *periodic* (e.g. daily) Sharpes, one per trial, on
    the same frequency as the returns later passed to `deflated_sharpe_ratio`.
    Returns NaN for <2 finite trials (a single trial has nothing to deflate);
    returns 0.0 when all trials are identical (no dispersion ⇒ no selection
    advantage to remove).
    """
    arr = _to_array(trial_sharpes)
    N = arr.size
    if N < 2:
        return float("nan")
    sr_std = arr.std(ddof=1)
    if not np.isfinite(sr_std) or sr_std < _STD_FLOOR:
        return 0.0
    z = (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / N) + EULER_GAMMA * norm.ppf(
        1.0 - 1.0 / (N * math.e)
    )
    return float(sr_std * z)


def deflated_sharpe_ratio(
    returns: ArrayLike,
    trial_sharpes: ArrayLike,
) -> float:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    DSR = PSR(returns; benchmark = expected_max_sharpe(trial_sharpes)) — the
    probability that the *selected* strategy's true Sharpe exceeds what the
    best of N skill-less trials would have produced by luck. It penalizes
    both the number of trials N and their dispersion, so it cannot be gamed
    by reporting only the single best run while hiding the search.

    `returns` is the selected strategy's periodic (e.g. daily) return series;
    `sr_obs`, `n`, skew and kurtosis are all derived from it (so DSR carries
    the same skew/kurtosis adjustment as PSR). `trial_sharpes` are the
    per-trial periodic Sharpes from the sweep, computed on the *same*
    frequency as `returns` — by construction the selected trial's Sharpe is
    the max of that array.

    Returns NaN if `returns` is degenerate (n<3, zero variance) or if there
    are <2 trials to deflate against.
    """
    sr_null = expected_max_sharpe(trial_sharpes)
    if not np.isfinite(sr_null):
        return float("nan")
    stats = _sharpe_and_moments(returns)
    if stats is None:
        return float("nan")
    sr, n, g3, g4 = stats
    return _psr_from_stats(sr, n, sr_null, g3, g4)


def calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Calmar = annualized_return / |max_drawdown|.

    Uses the project's positive-max_drawdown convention (drawdown is stored
    as `1 - portfolio/cummax`, always ≥ 0). Returns NaN if `max_drawdown`
    is non-positive or non-finite.
    """
    if not np.isfinite(annualized_return) or not np.isfinite(max_drawdown):
        return float("nan")
    if max_drawdown <= 1e-9:
        return float("nan")
    return float(annualized_return / max_drawdown)


def probability_backtest_overfitting(
    returns_matrix: Union[np.ndarray, pd.DataFrame],
    n_splits: int = 10,
) -> float:
    """PBO via Combinatorially Symmetric Cross-Validation (Bailey et al. 2015).

    Given a (T observations × N strategies) returns matrix, partition T into
    `n_splits` even slices, then over every (n_splits choose n_splits/2)
    split into IS / OOS halves:
      * pick the strategy with the highest IS Sharpe (n*)
      * compute n*'s OOS rank under the Bailey/LdP convention (rank 1 = worst
        OOS, rank N = best OOS); relative rank w = rank / (N + 1)
      * count the split as "overfit" if logit(w) < 0 i.e. w < 0.5 — meaning
        the IS-best strategy is OOS below-median.

    PBO is the mean of the overfit indicator across all splits.

    Returns NaN if there are <2 strategies, <2 splits, or T is too small
    to slice cleanly (`T < n_splits * 2`).
    """
    if isinstance(returns_matrix, pd.DataFrame):
        M = returns_matrix.to_numpy(dtype=float)
    else:
        M = np.asarray(returns_matrix, dtype=float)

    if M.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D, got shape {M.shape}")
    T, N = M.shape
    if N < 2:
        logger.warning("PBO undefined for <2 strategies; returning NaN")
        return float("nan")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even and ≥ 2, got {n_splits}")
    slice_len = T // n_splits
    if slice_len < 2:
        logger.warning(
            f"PBO: T={T} too small for n_splits={n_splits} (need slice_len ≥ 2); returning NaN"
        )
        return float("nan")

    # Pre-slice indices.
    slices = [np.arange(i * slice_len, (i + 1) * slice_len) for i in range(n_splits)]
    half = n_splits // 2

    overfit_count = 0
    total = 0
    for is_choice in combinations(range(n_splits), half):
        oos_choice = tuple(i for i in range(n_splits) if i not in is_choice)
        is_idx = np.concatenate([slices[i] for i in is_choice])
        oos_idx = np.concatenate([slices[i] for i in oos_choice])

        is_block = M[is_idx]
        oos_block = M[oos_idx]

        # Periodic Sharpe per strategy on each block. NaN-safe: if a column
        # has zero stdev, fall back to mean to break ties consistently.
        is_sharpe = _block_sharpe(is_block)
        oos_sharpe = _block_sharpe(oos_block)

        n_star = int(np.argmax(is_sharpe))
        # Bailey/LdP rank convention: 1 = worst OOS, N = best OOS. Ascending
        # argsort puts the lowest Sharpe at index 0 (rank 1).
        order = np.argsort(oos_sharpe, kind="stable")
        rank = int(np.where(order == n_star)[0][0]) + 1  # 1-indexed
        w = rank / (N + 1)
        if w >= 1.0:
            logit = float("inf")
        elif w <= 0.0:
            logit = float("-inf")
        else:
            logit = math.log(w / (1.0 - w))
        if logit < 0.0:
            overfit_count += 1
        total += 1

    return overfit_count / total if total > 0 else float("nan")


def _block_sharpe(block: np.ndarray) -> np.ndarray:
    """Per-column periodic Sharpe; columns with zero stdev get 0 to keep
    ranking deterministic without NaN propagation."""
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1)
    out = np.where(std > 0, mean / np.where(std > 0, std, 1.0), 0.0)
    return out
