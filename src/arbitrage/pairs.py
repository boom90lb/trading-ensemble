"""Cointegration-driven pair selection and causal spread signals.

The production-critical invariant is that discovery happens on a formation
sample only. The returned pair candidates are fixed objects: downstream signal
generation may update rolling spread statistics causally, but it must not rerun
cointegration tests on the evaluation period.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint  # type: ignore


_EPS = 1e-12


@dataclass(frozen=True)
class PairSelectionConfig:
    """Controls train-only pair discovery.

    `fdr_alpha` applies Benjamini-Hochberg to cointegration p-values before the
    economic filters. This is the minimum viable guard against "scan every pair,
    trade the luckiest p-value" selection bias.
    """

    min_obs: int = 252
    max_coint_pvalue: float = 0.05
    max_adf_pvalue: float = 0.05
    fdr_alpha: float = 0.10
    min_abs_return_corr: float = 0.30
    min_half_life: float = 1.0
    max_half_life: float = 60.0
    max_beta_drift: float = 0.50
    min_abs_beta: float = 0.05
    max_pairs: int = 10
    autolag: str = "aic"

    def __post_init__(self) -> None:
        if self.min_obs < 30:
            raise ValueError(f"min_obs must be >= 30, got {self.min_obs}")
        if not 0.0 < self.max_coint_pvalue < 1.0:
            raise ValueError(f"max_coint_pvalue must be in (0, 1), got {self.max_coint_pvalue}")
        if not 0.0 < self.max_adf_pvalue < 1.0:
            raise ValueError(f"max_adf_pvalue must be in (0, 1), got {self.max_adf_pvalue}")
        if not 0.0 < self.fdr_alpha < 1.0:
            raise ValueError(f"fdr_alpha must be in (0, 1), got {self.fdr_alpha}")
        if not 0.0 <= self.min_abs_return_corr <= 1.0:
            raise ValueError(f"min_abs_return_corr must be in [0, 1], got {self.min_abs_return_corr}")
        if self.min_half_life <= 0 or self.max_half_life <= self.min_half_life:
            raise ValueError("half-life bounds must satisfy 0 < min_half_life < max_half_life")
        if self.max_beta_drift < 0:
            raise ValueError(f"max_beta_drift must be >= 0, got {self.max_beta_drift}")
        if self.min_abs_beta < 0:
            raise ValueError(f"min_abs_beta must be >= 0, got {self.min_abs_beta}")
        if self.max_pairs < 1:
            raise ValueError(f"max_pairs must be >= 1, got {self.max_pairs}")


@dataclass(frozen=True)
class PairSignalConfig:
    """Controls causal spread-to-position conversion."""

    z_window: int = 60
    min_z_obs: int = 40
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    gross_per_pair: float = 1.0

    def __post_init__(self) -> None:
        if self.z_window < 2:
            raise ValueError(f"z_window must be >= 2, got {self.z_window}")
        if not 1 <= self.min_z_obs <= self.z_window:
            raise ValueError("min_z_obs must be in [1, z_window]")
        if self.entry_z <= 0:
            raise ValueError(f"entry_z must be > 0, got {self.entry_z}")
        if not 0 <= self.exit_z < self.entry_z:
            raise ValueError("exit_z must satisfy 0 <= exit_z < entry_z")
        if self.stop_z <= self.entry_z:
            raise ValueError("stop_z must be greater than entry_z")
        if self.gross_per_pair <= 0:
            raise ValueError(f"gross_per_pair must be > 0, got {self.gross_per_pair}")


@dataclass(frozen=True)
class PairCandidate:
    """A fixed train-sample cointegrated relationship."""

    asset_y: str
    asset_x: str
    alpha: float
    beta: float
    coint_pvalue: float
    coint_stat: float
    adf_pvalue: float
    adf_stat: float
    half_life: float
    beta_drift: float
    return_corr: float
    n_obs: int

    @property
    def symbols(self) -> tuple[str, str]:
        return self.asset_y, self.asset_x


@dataclass(frozen=True)
class PairSignalFrame:
    """Outputs from one pair's causal signal pass."""

    candidate: PairCandidate
    spread: pd.Series
    zscore: pd.Series
    target_weights: pd.DataFrame


@dataclass(frozen=True)
class PairSelectionReport:
    """Deterministic ledger for one train-only pair-selection pass."""

    n_symbols: int
    n_symbol_pairs: int
    n_raw_candidates: int
    fdr_alpha: float
    fdr_cutoff: float | None
    rejection_counts: dict[str, int]
    raw_candidates: tuple[PairCandidate, ...]
    selected_candidates: tuple[PairCandidate, ...]


def benjamini_hochberg_mask(pvalues: Iterable[float], alpha: float) -> np.ndarray:
    """Return the BH-FDR acceptance mask for a vector of p-values."""
    p = np.asarray(list(pvalues), dtype=float)
    mask = np.zeros(p.shape, dtype=bool)
    finite = np.isfinite(p)
    if not finite.any():
        return mask

    finite_idx = np.flatnonzero(finite)
    ordered_local = np.argsort(p[finite])
    ordered_idx = finite_idx[ordered_local]
    ordered_p = p[ordered_idx]
    m = float(len(ordered_p))
    thresholds = alpha * (np.arange(1, len(ordered_p) + 1) / m)
    passed = ordered_p <= thresholds
    if not passed.any():
        return mask

    k = int(np.flatnonzero(passed).max())
    cutoff = ordered_p[k]
    mask[finite] = p[finite] <= cutoff
    return mask


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS estimate of y = alpha + beta*x."""
    frame = pd.concat([y, x], axis=1).dropna()
    if len(frame) < 2:
        return float("nan"), float("nan")
    yy = frame.iloc[:, 0].to_numpy(dtype=float)
    xx = frame.iloc[:, 1].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(xx), xx])
    alpha, beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    return float(alpha), float(beta)


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate AR(1) mean-reversion half-life in bars.

    Regression is Δs_t = a + b*s_{t-1} + e_t. Mean reversion requires b < 0;
    otherwise the half-life is infinite and the pair is not tradeable.
    """
    s = pd.Series(spread, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3 or s.std(ddof=1) <= _EPS:
        return float("inf")

    lagged = s.shift(1)
    delta = s.diff()
    frame = pd.concat([delta.rename("delta"), lagged.rename("lagged")], axis=1).dropna()
    if len(frame) < 2:
        return float("inf")
    X = np.column_stack([np.ones(len(frame)), frame["lagged"].to_numpy(dtype=float)])
    slope = float(np.linalg.lstsq(X, frame["delta"].to_numpy(dtype=float), rcond=None)[0][1])
    if slope >= -_EPS:
        return float("inf")
    return float(max(1.0, -np.log(2.0) / slope))


def _log_prices(close_prices: pd.DataFrame) -> pd.DataFrame:
    numeric = close_prices.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.where(numeric > 0)
    return np.log(numeric).replace([np.inf, -np.inf], np.nan)


def _beta_drift(y: pd.Series, x: pd.Series, beta: float) -> float:
    frame = pd.concat([y, x], axis=1).dropna()
    if len(frame) < 6:
        return float("inf")
    half = len(frame) // 2
    _, b1 = estimate_hedge_ratio(frame.iloc[:half, 0], frame.iloc[:half, 1])
    _, b2 = estimate_hedge_ratio(frame.iloc[half:, 0], frame.iloc[half:, 1])
    if not np.isfinite(b1) or not np.isfinite(b2):
        return float("inf")
    return float(abs(b2 - b1) / max(abs(beta), _EPS))


def _candidate_from_pair(
    asset_y: str, asset_x: str, log_prices: pd.DataFrame, config: PairSelectionConfig
) -> Optional[PairCandidate]:
    pair = log_prices[[asset_y, asset_x]].dropna()
    if len(pair) < config.min_obs:
        return None

    y = pair[asset_y]
    x = pair[asset_x]
    try:
        coint_stat, coint_pvalue, _ = coint(y, x, autolag=config.autolag)
    except Exception:
        return None
    if not np.isfinite(coint_pvalue):
        return None

    alpha, beta = estimate_hedge_ratio(y, x)
    if not np.isfinite(beta) or abs(beta) < config.min_abs_beta:
        return None
    spread = y - alpha - beta * x
    if spread.std(ddof=1) <= _EPS:
        return None

    try:
        adf_stat, adf_pvalue, *_ = adfuller(spread, autolag=config.autolag)
    except Exception:
        return None
    half_life = estimate_half_life(spread)
    beta_drift = _beta_drift(y, x, beta)
    return_corr = float(pair.diff().dropna().corr().iloc[0, 1])

    return PairCandidate(
        asset_y=asset_y,
        asset_x=asset_x,
        alpha=float(alpha),
        beta=float(beta),
        coint_pvalue=float(coint_pvalue),
        coint_stat=float(coint_stat),
        adf_pvalue=float(adf_pvalue),
        adf_stat=float(adf_stat),
        half_life=float(half_life),
        beta_drift=float(beta_drift),
        return_corr=return_corr,
        n_obs=int(len(pair)),
    )


def _rejection_reason(candidate: PairCandidate, fdr_keep: bool, config: PairSelectionConfig) -> str | None:
    if not fdr_keep:
        return "fdr"
    if candidate.coint_pvalue > config.max_coint_pvalue:
        return "coint_pvalue"
    if candidate.adf_pvalue > config.max_adf_pvalue:
        return "adf_pvalue"
    if not np.isfinite(candidate.return_corr) or abs(candidate.return_corr) < config.min_abs_return_corr:
        return "return_corr"
    if not np.isfinite(candidate.half_life) or not config.min_half_life <= candidate.half_life <= config.max_half_life:
        return "half_life"
    if not np.isfinite(candidate.beta_drift) or candidate.beta_drift > config.max_beta_drift:
        return "beta_drift"
    return None


def _known_rejection_counts(counter: Counter[str]) -> dict[str, int]:
    reasons = [
        "candidate_unavailable",
        "fdr",
        "coint_pvalue",
        "adf_pvalue",
        "return_corr",
        "half_life",
        "beta_drift",
        "max_pairs",
    ]
    return {reason: int(counter.get(reason, 0)) for reason in reasons}


def scan_cointegrated_pairs_with_report(
    close_prices: pd.DataFrame, config: Optional[PairSelectionConfig] = None
) -> PairSelectionReport:
    """Scan a formation window and return selected pairs plus a selection ledger."""
    config = config or PairSelectionConfig()
    log_prices = _log_prices(close_prices)
    symbols = [c for c in log_prices.columns if log_prices[c].notna().sum() >= config.min_obs]
    n_symbol_pairs = len(symbols) * (len(symbols) - 1) // 2
    rejected: Counter[str] = Counter()
    raw: list[PairCandidate] = []
    for asset_y, asset_x in combinations(symbols, 2):
        candidates = [
            c
            for c in (
                _candidate_from_pair(asset_y, asset_x, log_prices, config),
                _candidate_from_pair(asset_x, asset_y, log_prices, config),
            )
            if c is not None
        ]
        if candidates:
            raw.append(min(candidates, key=lambda c: c.coint_pvalue))
        else:
            rejected["candidate_unavailable"] += 1

    if not raw:
        return PairSelectionReport(
            n_symbols=len(symbols),
            n_symbol_pairs=n_symbol_pairs,
            n_raw_candidates=0,
            fdr_alpha=float(config.fdr_alpha),
            fdr_cutoff=None,
            rejection_counts=_known_rejection_counts(rejected),
            raw_candidates=(),
            selected_candidates=(),
        )

    fdr_mask = benjamini_hochberg_mask((c.coint_pvalue for c in raw), config.fdr_alpha)
    accepted_pvalues = [c.coint_pvalue for keep, c in zip(fdr_mask, raw) if keep]
    fdr_cutoff = float(max(accepted_pvalues)) if accepted_pvalues else None
    filtered: list[PairCandidate] = []
    for keep, candidate in zip(fdr_mask, raw):
        reason = _rejection_reason(candidate, bool(keep), config)
        if reason is None:
            filtered.append(candidate)
        else:
            rejected[reason] += 1
    filtered.sort(key=lambda c: (c.coint_pvalue, c.half_life, c.beta_drift))
    selected = filtered[: config.max_pairs]
    rejected["max_pairs"] += max(0, len(filtered) - len(selected))
    return PairSelectionReport(
        n_symbols=len(symbols),
        n_symbol_pairs=n_symbol_pairs,
        n_raw_candidates=len(raw),
        fdr_alpha=float(config.fdr_alpha),
        fdr_cutoff=fdr_cutoff,
        rejection_counts=_known_rejection_counts(rejected),
        raw_candidates=tuple(raw),
        selected_candidates=tuple(selected),
    )


def scan_cointegrated_pairs(
    close_prices: pd.DataFrame, config: Optional[PairSelectionConfig] = None
) -> List[PairCandidate]:
    """Scan a formation close-price matrix for tradeable cointegrated pairs.

    The function only reads the supplied frame. To avoid lookahead, callers must
    pass a formation window, not the full backtest period.
    """
    report = scan_cointegrated_pairs_with_report(close_prices, config)
    return list(report.selected_candidates)


def generate_pair_positions(
    close_prices: pd.DataFrame,
    candidate: PairCandidate,
    config: Optional[PairSignalConfig] = None,
) -> PairSignalFrame:
    """Generate hedge-ratio spread weights from causal close-to-close z-scores.

    Returned weights are *targets at the close*. The portfolio backtester shifts
    them one bar before applying returns, so the signal bar never earns same-bar
    PnL.
    """
    config = config or PairSignalConfig()
    missing = [s for s in candidate.symbols if s not in close_prices.columns]
    if missing:
        raise KeyError(f"close_prices missing symbols: {missing}")

    log_prices = _log_prices(close_prices[list(candidate.symbols)])
    spread = (log_prices[candidate.asset_y] - candidate.alpha - candidate.beta * log_prices[candidate.asset_x]).rename(
        f"{candidate.asset_y}_{candidate.asset_x}_spread"
    )
    mean = spread.rolling(config.z_window, min_periods=config.min_z_obs).mean()
    std = spread.rolling(config.z_window, min_periods=config.min_z_obs).std(ddof=1)
    zscore = ((spread - mean) / std.where(std > _EPS)).rename("zscore")

    state = 0
    rows: list[dict[str, float]] = []
    gross = abs(1.0) + abs(candidate.beta)
    for z in zscore:
        if not np.isfinite(z):
            state = 0
        elif state != 0 and abs(z) >= config.stop_z:
            state = 0
        elif state != 0 and abs(z) <= config.exit_z:
            state = 0
        elif state == 0 and z >= config.entry_z:
            state = -1
        elif state == 0 and z <= -config.entry_z:
            state = 1

        if state == 0 or gross <= _EPS:
            rows.append({candidate.asset_y: 0.0, candidate.asset_x: 0.0})
            continue
        scale = config.gross_per_pair / gross
        rows.append(
            {
                candidate.asset_y: float(state * scale),
                candidate.asset_x: float(-state * candidate.beta * scale),
            }
        )

    weights = pd.DataFrame(rows, index=close_prices.index, columns=list(candidate.symbols)).fillna(0.0)
    return PairSignalFrame(candidate=candidate, spread=spread, zscore=zscore, target_weights=weights)
