#!/usr/bin/env python
"""Phase 2.7 — hyperparameter-grid sweep that unlocks the Deflated Sharpe Ratio.

DSR (Bailey & López de Prado 2014) deflates an observed Sharpe by the
expected maximum Sharpe of N skill-less trials. It is only honest when N is a
*real, fully-counted* trial set — which is exactly what this sweep produces.

Per the Phase 2.7 scope (confirmed with the user):

  * **Search space:** ensemble-layer knobs only — ``target_vol``,
    ``signal_threshold``, ``weighting_strategy``. Members are forecast-only
    (default ``xgboost,lstm``) so no RL agent is refit per trial (that would
    be infeasible at grid scale).
  * **Trials → DSR:** each trial's *net, cost-aware* daily WFO-OOS Sharpe is
    BOTH the selection criterion and ``sr_obs``. DSR deflates the selection
    over N — no separate holdout is burned (standard Bailey/LdP usage). All
    Sharpes are *periodic* (daily) so the PSR/DSR ``√(n-1)`` math is correct.
  * **Method:** full Cartesian grid → exact, reproducible N.

Each trial trains per-fold ensembles (reusing ``train_symbol_wfo``) and runs
a cost-aware WFO backtest (reusing ``run_symbol_wfo``); the trial's Sharpe is
the periodic Sharpe of the equal-weight-across-symbols daily return series.

Outputs under ``runs/sweep_{ts}/``:
  * ``sweep_results.json`` — every trial's config + Sharpe, plus the DSR
    summary (selected config, sr_obs, sr_null, PSR, DSR, N).
  * ``trial_sharpes.json`` — ``{"trial_sharpes": [...]}`` consumable by
    ``scripts/backtest.py --trial_sharpes_json`` to render DSR in the report.
  * ``selected_config.json`` — the winning hyperparameters.

One MLflow parent run ``sweep_{ts}`` with a nested run per trial (and per
symbol within a trial, matching the training loop's run structure).
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from scripts.backtest import load_fold_ensemble, run_symbol_wfo
from scripts.training import build_features, train_symbol_wfo
from src.config import (
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_TRAINING_CONFIG,
    ModelConfig,
    TrainingConfig,
)
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.sentiment_analysis import SentimentAnalyzer
from src.tracking.mlflow_utils import (
    init_mlflow,
    log_artifact_dir,
    log_metrics_safe,
    log_params_safe,
)
from src.validation.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    periodic_sharpe,
    probabilistic_sharpe_ratio,
)
from src.validation.walk_forward import PurgedWalkForward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default ensemble-layer grid (N = 3 * 3 * 2 = 18). All three axes are
# post-fit/backtest-time knobs that genuinely change positions, so the trials
# are real and independently counted. Override via --grid_json.
DEFAULT_GRID: Dict[str, List[Any]] = {
    "target_vol": [0.5, 1.0, 1.5],
    "signal_threshold": [0.05, 0.1, 0.2],
    "weighting_strategy": ["static", "dynamic"],
}


# -- Pure helpers (unit-tested) ---------------------------------------------


def expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Full Cartesian product of a {axis: [values]} grid → list of configs.

    Deterministic order (axes in dict order, values in list order) so the
    trial count N and trial indices are reproducible — DSR needs an exact N.
    """
    if not grid:
        return [{}]
    keys = list(grid.keys())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def combine_symbol_returns(
    per_symbol_returns: Dict[str, pd.Series],
) -> pd.Series:
    """Equal-weight-by-date portfolio of per-symbol daily returns.

    Aligns the per-symbol series on their date index and averages across
    symbols (skipping symbols absent on a given date). For a single symbol
    this is just that symbol's return series. Empty input → empty series.
    """
    series = {k: v for k, v in per_symbol_returns.items() if v is not None and len(v)}
    if not series:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(series)
    return frame.mean(axis=1, skipna=True).dropna()


@dataclass
class TrialResult:
    """One grid point's outcome."""

    trial_id: int
    config: Dict[str, Any]
    sharpe: float  # periodic (daily) net Sharpe of the combined return series
    combined_returns: pd.Series = field(repr=False)
    per_symbol_metrics: Dict[str, Dict[str, float]] = field(
        default_factory=dict, repr=False
    )


def summarize_sweep(trials: List[TrialResult]) -> Dict[str, Any]:
    """Select the best trial and deflate its Sharpe over the full trial set.

    Selection is argmax of the periodic Sharpe; ``sr_obs`` is that same
    selected trial's daily Sharpe (so DSR corrects the selection bias). DSR
    is NaN when <2 trials produced a finite Sharpe (nothing to deflate).
    """
    valid = [t for t in trials if np.isfinite(t.sharpe)]
    trial_table = [
        {"trial_id": t.trial_id, "config": t.config, "sharpe": float(t.sharpe)}
        for t in trials
    ]
    base: Dict[str, Any] = {
        "n_trials": len(trials),
        "n_valid": len(valid),
        "trials": trial_table,
        "trial_sharpes": [float(t.sharpe) for t in valid],
    }
    if not valid:
        base.update(
            selected_trial_id=None, selected_config=None,
            selected_sharpe=None, sr_null=None, psr=None, dsr=None,
        )
        return base

    best = max(valid, key=lambda t: t.sharpe)
    trial_sharpes = np.array([t.sharpe for t in valid], dtype=float)
    sr_null = expected_max_sharpe(trial_sharpes)
    psr = probabilistic_sharpe_ratio(best.combined_returns, sr_benchmark=0.0)
    dsr = deflated_sharpe_ratio(best.combined_returns, trial_sharpes)

    def _f(x: float) -> Optional[float]:
        return float(x) if np.isfinite(x) else None

    base.update(
        selected_trial_id=best.trial_id,
        selected_config=best.config,
        selected_sharpe=float(best.sharpe),
        sr_null=_f(sr_null),
        psr=_f(psr),
        dsr=_f(dsr),
    )
    return base


# -- Trial execution --------------------------------------------------------


def _backtest_args(args, signal_threshold: float) -> SimpleNamespace:
    """Build the args namespace ``run_symbol_wfo`` reads for cost config."""
    return SimpleNamespace(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        commission_bps=args.commission_bps,
        spread_bps=args.spread_bps,
        slippage_coeff=args.slippage_coeff,
        borrow_bps_annual=args.borrow_bps_annual,
        order_type=args.order_type,
        signal_threshold=signal_threshold,
    )


def run_trial(
    trial_id: int,
    config: Dict[str, Any],
    enhanced_data: Dict[str, pd.DataFrame],
    target_col: str,
    splitter: PurgedWalkForward,
    model_configs: List[ModelConfig],
    model_names: List[str],
    training_config: TrainingConfig,
    args,
    feature_engineer: FeatureEngineer,
    trial_dir: Path,
) -> TrialResult:
    """Train per-fold ensembles + cost-aware WFO backtest for one grid point.

    Reuses ``train_symbol_wfo`` (parametrized by the trial's ensemble knobs)
    and ``run_symbol_wfo`` (parametrized by the trial's signal threshold) so
    the sweep stays apples-to-apples with the production training/backtest.
    """
    horizon = training_config.prediction_horizon
    train_args = SimpleNamespace(
        rl_timesteps=getattr(args, "rl_timesteps", 0),
        checkpoint_freq=0, gpu=False,
    )
    bt_args = _backtest_args(args, float(config["signal_threshold"]))

    per_symbol_returns: Dict[str, pd.Series] = {}
    per_symbol_metrics: Dict[str, Dict[str, float]] = {}

    for symbol, full_df in enhanced_data.items():
        symbol_dir = trial_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=f"{symbol}", nested=True):
            train_symbol_wfo(
                symbol=symbol,
                full_df=full_df,
                target_col=target_col,
                splitter=splitter,
                model_configs=model_configs,
                model_names=model_names,
                training_config=training_config,
                args=train_args,
                gpu_available=False,
                feature_engineer=feature_engineer,
                symbol_dir=symbol_dir,
                weighting_strategy=str(config["weighting_strategy"]),
                target_vol=float(config["target_vol"]),
                position_cap=float(config.get("position_cap", 1.0)),
            )

        fold_dirs = sorted(d for d in symbol_dir.glob("fold_*") if d.is_dir())
        if not fold_dirs:
            logger.warning(f"trial {trial_id} {symbol}: no folds fit; skipping")
            continue

        usable = full_df.dropna(subset=[target_col])
        first_model = load_fold_ensemble(fold_dirs[0], horizon)
        res = run_symbol_wfo(
            symbol=symbol,
            model=first_model,
            fold_dirs=fold_dirs,
            full_df=full_df,
            usable=usable,
            args=bt_args,
            use_sentiment=False,
            sentiment_analyzer=None,
            load_fold_model=load_fold_ensemble,
            horizon=horizon,
        )
        eq = res["equity_curve"]
        if eq.empty or "daily_return" not in eq:
            continue
        per_symbol_returns[symbol] = eq["daily_return"].dropna()
        per_symbol_metrics[symbol] = res["metrics"]

    combined = combine_symbol_returns(per_symbol_returns)
    sharpe = periodic_sharpe(combined)
    logger.info(
        f"trial {trial_id} {config} → daily Sharpe {sharpe:.4f} "
        f"({len(per_symbol_returns)} symbols)"
    )
    return TrialResult(
        trial_id=trial_id, config=config, sharpe=float(sharpe),
        combined_returns=combined, per_symbol_metrics=per_symbol_metrics,
    )


# -- Orchestration ----------------------------------------------------------


def run_sweep(
    args,
    enhanced_data: Dict[str, pd.DataFrame],
    grid: Dict[str, List[Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the full grid, deflate the winner, persist + log results.

    ``enhanced_data`` is the per-symbol feature frame (so this is testable
    with synthetic data); ``main`` builds it via the data loader.
    """
    trials_spec = expand_grid(grid)
    horizon = args.horizon
    target_col = f"target_{horizon}"

    model_names = [n.strip() for n in args.models.split(",") if n.strip()]
    model_configs = [
        ModelConfig(name=n, enabled=True, weight=DEFAULT_MODEL_WEIGHTS.get(n, 1.0))
        for n in model_names
    ]

    training_config = TrainingConfig(
        symbols=list(enhanced_data.keys()),
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        prediction_horizon=horizon,
        n_splits=args.n_splits,
        embargo_pct=args.embargo_pct,
        expanding=not args.rolling,
    )
    splitter = PurgedWalkForward(
        n_splits=training_config.n_splits,
        purge_horizon=training_config.effective_purge_horizon,
        embargo_pct=training_config.embargo_pct,
        expanding=training_config.expanding,
    )
    feature_engineer = FeatureEngineer()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = init_mlflow(args.experiment)

    logger.info(
        f"Sweep: {len(trials_spec)} trials over grid {grid} "
        f"on {len(enhanced_data)} symbols, models={model_names}"
    )

    trial_results: List[TrialResult] = []
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"sweep_{timestamp}",
    ):
        log_params_safe({
            "grid": json.dumps(grid),
            "n_trials": len(trials_spec),
            "symbols": ",".join(enhanced_data.keys()),
            "models": ",".join(model_names),
            "horizon": horizon,
            "n_splits": training_config.n_splits,
            "embargo_pct": training_config.embargo_pct,
            "expanding": training_config.expanding,
        })

        for i, config in enumerate(trials_spec):
            trial_dir = output_dir / f"trial_{i}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            with mlflow.start_run(run_name=f"trial_{i}", nested=True):
                log_params_safe({f"cfg_{k}": v for k, v in config.items()})
                tr = run_trial(
                    trial_id=i,
                    config=config,
                    enhanced_data=enhanced_data,
                    target_col=target_col,
                    splitter=splitter,
                    model_configs=model_configs,
                    model_names=model_names,
                    training_config=training_config,
                    args=args,
                    feature_engineer=feature_engineer,
                    trial_dir=trial_dir,
                )
                trial_results.append(tr)
                log_metrics_safe({"trial_daily_sharpe": tr.sharpe})

        summary = summarize_sweep(trial_results)
        log_metrics_safe({
            k: v for k, v in {
                "dsr": summary.get("dsr"),
                "psr": summary.get("psr"),
                "sr_null": summary.get("sr_null"),
                "selected_sharpe": summary.get("selected_sharpe"),
                "n_valid_trials": summary.get("n_valid"),
            }.items() if v is not None
        })

        # Persist artifacts.
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(output_dir / "trial_sharpes.json", "w") as f:
            json.dump({"trial_sharpes": summary["trial_sharpes"]}, f, indent=2)
        if summary.get("selected_config") is not None:
            with open(output_dir / "selected_config.json", "w") as f:
                json.dump(summary["selected_config"], f, indent=2, default=str)
        log_artifact_dir(output_dir)

    _print_summary(summary)
    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    lines = ["", "=" * 60, "Hyperparameter sweep — DSR summary", "-" * 60]
    lines.append(f"trials run:        {summary['n_trials']} "
                 f"({summary['n_valid']} with a finite Sharpe)")
    if summary.get("selected_config") is None:
        lines.append("no valid trial — nothing selected")
    else:
        lines.append(f"selected config:   {summary['selected_config']}")
        lines.append(f"selected Sharpe:   {summary['selected_sharpe']:.4f} (daily)")
        sr_null = summary.get("sr_null")
        lines.append(
            "expected max (H0): "
            + (f"{sr_null:.4f}" if sr_null is not None else "n/a (<2 trials)")
        )
        psr = summary.get("psr")
        lines.append("PSR (vs 0):        "
                     + (f"{psr:.3f}" if psr is not None else "n/a"))
        dsr = summary.get("dsr")
        lines.append("DSR (deflated):    "
                     + (f"{dsr:.3f}" if dsr is not None else "n/a (<2 trials)"))
    lines.extend(["=" * 60, ""])
    print("\n".join(lines))


def prepare_sweep_data(args) -> Dict[str, pd.DataFrame]:
    """Fetch + causally build the per-symbol feature frames for the sweep."""
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    training_config = TrainingConfig(
        symbols=symbols, timeframe=args.timeframe,
        start_date=args.start_date, end_date=args.end_date,
        prediction_horizon=args.horizon, use_sentiment=args.use_sentiment,
        n_splits=args.n_splits, embargo_pct=args.embargo_pct,
        expanding=not args.rolling,
    )
    data_loader = DataLoader()
    all_data = data_loader.fetch_training_data(training_config)
    feature_engineer = FeatureEngineer()
    sentiment_analyzer = SentimentAnalyzer() if args.use_sentiment else None

    enhanced: Dict[str, pd.DataFrame] = {}
    for symbol, raw_df in all_data.items():
        enhanced[symbol] = build_features(
            raw_df=raw_df, symbol=symbol,
            feature_engineer=feature_engineer,
            sentiment_analyzer=sentiment_analyzer, horizon=args.horizon,
        )
    return enhanced


def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter-grid sweep with DSR (Phase 2.7).",
    )
    p.add_argument("--symbols", type=str,
                   default=",".join(DEFAULT_TRAINING_CONFIG.symbols))
    p.add_argument("--timeframe", type=str, default="1d")
    p.add_argument("--start_date", type=str,
                   default=DEFAULT_TRAINING_CONFIG.start_date)
    p.add_argument("--end_date", type=str, default=None)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--use_sentiment", action="store_true")
    p.add_argument(
        "--models", type=str, default="xgboost,lstm",
        help="Forecast-only members. RL members make the grid infeasible.",
    )
    p.add_argument("--rl_timesteps", type=int, default=0)

    # WFO knobs.
    p.add_argument("--n_splits", type=int, default=DEFAULT_TRAINING_CONFIG.n_splits)
    p.add_argument("--embargo_pct", type=float,
                   default=DEFAULT_TRAINING_CONFIG.embargo_pct)
    p.add_argument("--rolling", action="store_true")

    # Execution / trading config.
    p.add_argument("--initial_capital", type=float, default=10000.0)
    p.add_argument("--position_size", type=float, default=0.1)
    p.add_argument("--stop_loss", type=float, default=0.02)
    p.add_argument("--take_profit", type=float, default=0.05)
    p.add_argument("--commission_bps", type=float, default=1.0)
    p.add_argument("--spread_bps", type=float, default=1.0)
    p.add_argument("--slippage_coeff", type=float, default=10.0)
    p.add_argument("--borrow_bps_annual", type=float, default=50.0)
    p.add_argument("--order_type", type=str, default="MOO",
                   choices=["MOO", "MOC"])

    # Grid override + MLflow.
    p.add_argument(
        "--grid_json", type=str, default=None,
        help='JSON {"axis": [values]} overriding the default 18-point grid.',
    )
    p.add_argument("--experiment", type=str, default="trading_ensemble_sweep")
    return p.parse_args()


def main():
    args = parse_args()
    grid = DEFAULT_GRID
    if args.grid_json:
        with open(args.grid_json) as f:
            grid = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Sweep artifacts → {output_dir.resolve()}")

    enhanced_data = prepare_sweep_data(args)
    if not enhanced_data:
        logger.error("No data fetched; aborting sweep")
        return
    run_sweep(args, enhanced_data, grid, output_dir)


if __name__ == "__main__":
    main()
