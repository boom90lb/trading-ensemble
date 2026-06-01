#!/usr/bin/env python
"""WFO backtest the ensemble against a training run's per-fold models.

Phase 2.2: replaces the one-shot backtest with a per-symbol per-fold
outer loop that iterates the exact PurgedWalkForward fold structure
saved by ``scripts/training.py``. Within each symbol:

  * One TradingStrategy instance, ``_reset_state()`` called once before
    fold 0 — cash, positions, transaction log persist across folds.
  * Per fold: load ``fold_{k}/ensemble_model/`` into ``strategy.model``,
    slice the features frame to that fold's test_idx-derived date range,
    call ``strategy.run_segment()``, then ``execution_model.drop_pending()``
    so orders queued at the last bar of fold k cannot fill in fold k+1.
  * Per-fold segment DataFrames are concatenated and finalized into one
    continuous equity curve; Sharpe / PSR / MaxDD / Calmar are computed
    on that curve.
  * Baselines (BAH / MA-X / TSMOM) run inside the SAME fold loop with the
    same state-persistence story so PBO across {ensemble, baselines} is
    fold-aligned by date.

PBO is computed per symbol across that symbol's strategy set. Each symbol's
artifacts land in ``RESULTS_DIR/wfo_backtest_{timestamp}/{symbol}/``.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from scripts.training import build_features
from src.baselines import BuyAndHold, MACrossover, TSMOM
from src.config import (
    ExecutionConfig,
    RESULTS_DIR,
    TradingConfig,
)
from src.data_loader import DataLoader
from src.execution import ExecutionModel
from src.features import FeatureEngineer
from src.models.base import BaseModel
from src.models.ensemble import EnsembleModel
from src.sentiment_analysis import SentimentAnalyzer
from src.tracking.mlflow_utils import (
    init_mlflow,
    log_artifact_dir,
    log_metrics_safe,
    log_params_safe,
)
from src.trading import TradingStrategy
from src.validation.metrics import (
    deflated_sharpe_ratio,
    periodic_sharpe,
    probability_backtest_overfitting,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "WFO backtest the ensemble using per-fold models from a "
            "scripts/training.py run."
        ),
    )

    parser.add_argument(
        "--training_run", type=str, required=True,
        help="Path to runs/{run_name}/ produced by scripts/training.py",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Optional subset of training-run symbols (comma-separated)",
    )

    # Execution config (mirrors training-side flags).
    parser.add_argument("--initial_capital", type=float, default=10000.0)
    parser.add_argument("--position_size", type=float, default=0.1)
    parser.add_argument("--stop_loss", type=float, default=0.02)
    parser.add_argument("--take_profit", type=float, default=0.05)
    parser.add_argument("--commission_bps", type=float, default=1.0)
    parser.add_argument("--spread_bps", type=float, default=1.0)
    parser.add_argument("--slippage_coeff", type=float, default=10.0)
    parser.add_argument("--borrow_bps_annual", type=float, default=50.0)
    parser.add_argument(
        "--order_type", type=str, default="MOO", choices=["MOO", "MOC"],
    )

    # Sentiment override (training run dictates default).
    parser.add_argument(
        "--no_sentiment", action="store_true",
        help="Skip sentiment join even if the training run used sentiment",
    )

    # Baselines.
    parser.add_argument("--no_baselines", action="store_true")
    parser.add_argument(
        "--no_dividends", action="store_true",
        help="Skip dividend cash credit (Phase 4 §4.1); avoids /dividends API calls.",
    )
    parser.add_argument("--ma_fast", type=int, default=20)
    parser.add_argument("--ma_slow", type=int, default=50)
    parser.add_argument("--tsmom_lookback", type=int, default=60)

    # Signal threshold (sweep-selected configs override the 0.1 default).
    parser.add_argument("--signal_threshold", type=float, default=0.1)

    # DSR: path to a sweep's trial_sharpes JSON ({"trial_sharpes": [...]}).
    # When given, the report deflates the ensemble's Sharpe against the
    # honestly-counted trial set (Phase 2.7). Absent → DSR shown as n/a.
    parser.add_argument("--trial_sharpes_json", type=str, default=None)

    # MLflow.
    parser.add_argument(
        "--experiment", type=str, default="trading_ensemble_backtest",
        help="MLflow experiment name",
    )

    return parser.parse_args()


def load_training_run(
    training_run_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, List[Path]]]:
    """Read training_config.json + enumerate per-symbol fold directories.

    Returns ``(training_config_dict, {symbol: [fold_0_path, fold_1_path, …]})``.
    Symbols whose directory exists but has zero ``fold_*`` subdirs are
    omitted (a training run that crashed mid-symbol would land here).
    """
    config_path = training_run_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"training_config.json not found in {training_run_dir}; "
            "is this a Phase 2.2 training run?"
        )
    with open(config_path) as f:
        training_config = json.load(f)

    fold_dirs_per_symbol: Dict[str, List[Path]] = {}
    for symbol_dir in sorted(training_run_dir.iterdir()):
        if not symbol_dir.is_dir():
            continue
        fold_dirs = sorted(
            d for d in symbol_dir.glob("fold_*") if d.is_dir()
        )
        if fold_dirs:
            fold_dirs_per_symbol[symbol_dir.name] = fold_dirs
    return training_config, fold_dirs_per_symbol


def load_fold_ensemble(fold_dir: Path, horizon: int) -> EnsembleModel:
    """Load the EnsembleModel pickled by scripts/training.py for one fold."""
    state_path = fold_dir / "ensemble_model" / f"ensemble_state_h{horizon}.pkl"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing ensemble state at {state_path}")
    ensemble = EnsembleModel(target_column="close", horizon=horizon)
    ensemble.load(state_path)
    return ensemble


def _build_execution_and_trading_configs(args) -> Tuple[ExecutionConfig, TradingConfig]:
    execution_config = ExecutionConfig(
        spread_bps=args.spread_bps,
        slippage_coeff=args.slippage_coeff,
        commission_bps=args.commission_bps,
        borrow_rate_bps_annual=args.borrow_bps_annual,
        default_order_type=args.order_type,
    )
    trading_config = TradingConfig(
        initial_capital=args.initial_capital,
        position_size=args.position_size,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        # Sweep (Phase 2.7) varies this; absent on a plain backtest → 0.1.
        signal_threshold=getattr(args, "signal_threshold", 0.1),
        execution=execution_config,
    )
    return execution_config, trading_config


def _fold_date_range(
    fold_dir: Path, usable: pd.DataFrame,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Resolve the (start, end) bar dates for one fold's test slice.

    Uses the ``test_idx`` array saved at training time to look up dates
    in ``usable`` (the dropna-on-target frame). Returns None if the slice
    is empty.
    """
    sidx_path = fold_dir / "split_idx.npz"
    if not sidx_path.exists():
        logger.warning(f"Missing split_idx.npz in {fold_dir}; skipping fold")
        return None
    sidx = np.load(sidx_path)
    test_idx = sidx["test_idx"]
    if test_idx.size == 0:
        return None
    test_dates = usable.index[test_idx]
    return test_dates[0], test_dates[-1]


def run_symbol_wfo(
    symbol: str,
    model: BaseModel,
    fold_dirs: List[Path],
    full_df: pd.DataFrame,
    usable: pd.DataFrame,
    args,
    use_sentiment: bool,
    sentiment_analyzer: Optional[SentimentAnalyzer],
    load_fold_model: Optional[callable] = None,
    horizon: Optional[int] = None,
    dividends: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Run the per-fold WFO loop for one symbol with one model family.

    For the ensemble path, ``model`` is the fold-0 ensemble and
    ``load_fold_model(fold_dir)`` is supplied so the strategy swaps in
    the appropriate per-fold ensemble at every iteration.

    For baselines, ``model`` is the single baseline instance that already
    ``prepare()``-d on the full close series; ``load_fold_model`` is None
    and the same model is reused across folds (baselines are stateless
    parameters, so no per-fold refit is needed).

    State persistence (Q3 Recommended): ``_reset_state()`` once before
    fold 0; ``execution_model.drop_pending()`` after every fold.
    """
    execution_config, trading_config = _build_execution_and_trading_configs(args)
    strategy = TradingStrategy(
        model=model,
        config=trading_config,
        sentiment_analyzer=sentiment_analyzer,
        use_sentiment=use_sentiment,
        execution_model=ExecutionModel(execution_config),
    )
    strategy._reset_state()

    segments: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, fold_dir in enumerate(fold_dirs):
        if fold_idx > 0 and load_fold_model is not None:
            assert horizon is not None
            strategy.model = load_fold_model(fold_dir, horizon)

        date_range = _fold_date_range(fold_dir, usable)
        if date_range is None:
            fold_metrics.append({"fold": fold_idx, "n_bars": 0})
            continue
        start_d, end_d = date_range

        seg = strategy.run_segment(
            data={symbol: full_df},
            start_date=str(start_d),
            end_date=str(end_d),
            use_sentiment=use_sentiment,
            dividends={symbol: dividends} if dividends is not None else None,
        )
        # Fold-boundary drop: any orders queued at fold k's last bar are
        # discarded so they cannot fill in fold k+1's first bar.
        strategy.execution_model.drop_pending()

        if seg.empty:
            fold_metrics.append({"fold": fold_idx, "n_bars": 0})
            continue

        finalized_seg = TradingStrategy._finalize_results(seg)
        seg_metrics = strategy.calculate_performance_metrics(finalized_seg)
        seg_metrics["fold"] = fold_idx
        seg_metrics["n_bars"] = len(seg)
        fold_metrics.append(seg_metrics)
        segments.append(seg)

    if not segments:
        return {
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "fold_metrics": fold_metrics,
            "transaction_log": [],
        }

    concatenated = pd.concat(segments)
    finalized = TradingStrategy._finalize_results(concatenated)
    metrics = strategy.calculate_performance_metrics(finalized)

    return {
        "equity_curve": finalized,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "transaction_log": list(strategy.transaction_log),
    }


def _build_baselines(args) -> List[BaseModel]:
    """Fresh baseline instances per symbol (each binds via prepare())."""
    return [
        BuyAndHold(),
        MACrossover(fast=args.ma_fast, slow=args.ma_slow),
        TSMOM(lookback=args.tsmom_lookback),
    ]


def run_symbol_baselines(
    symbol: str,
    fold_dirs: List[Path],
    full_df: pd.DataFrame,
    usable: pd.DataFrame,
    args,
    dividends: Optional[pd.Series] = None,
) -> List[Dict[str, Any]]:
    """Backtest each baseline over the same fold structure as the ensemble.

    Baselines don't have per-fold weights so the same baseline instance
    runs across all folds; state persistence + drop_pending semantics
    match the ensemble path exactly so PBO comparisons are apples-to-apples.
    Dividends are credited on the same total-return basis as the ensemble so a
    BuyAndHold over a dividend payer isn't unfairly penalized (Phase 4 §4.1).
    """
    records: List[Dict[str, Any]] = []
    for baseline in _build_baselines(args):
        baseline.fit(full_df, full_df["close"])
        baseline.prepare(full_df["close"])  # type: ignore[attr-defined]
        result = run_symbol_wfo(
            symbol=symbol,
            model=baseline,
            fold_dirs=fold_dirs,
            full_df=full_df,
            usable=usable,
            args=args,
            use_sentiment=False,
            sentiment_analyzer=None,
            load_fold_model=None,
            horizon=None,
            dividends=dividends,
        )
        if result["equity_curve"].empty:
            continue
        records.append({
            "name": baseline.name,
            "symbol": symbol,
            "metrics": result["metrics"],
            "fold_metrics": result["fold_metrics"],
            "daily_return": result["equity_curve"]["daily_return"],
        })
        logger.info(
            f"Baseline {baseline.name} on {symbol}: "
            f"total_return={_fmt_metric(result['metrics'], 'total_return')}"
        )
    return records


# -- Reporting helpers (mostly preserved from the pre-WFO version) ----------


_METRIC_KEYS = (
    "total_return", "annualized_return", "sharpe_ratio", "psr",
    "max_drawdown", "calmar_ratio",
)


def _fmt_metric(metrics: Dict[str, float], key: str) -> str:
    if key not in metrics or not np.isfinite(metrics[key]):
        return "n/a"
    val = metrics[key]
    if key in ("total_return", "annualized_return", "max_drawdown"):
        return f"{val * 100:+.2f}%"
    if key == "psr":
        return f"{val:.3f}"
    return f"{val:+.3f}"


def _build_returns_matrix(
    ensemble_returns: pd.Series,
    baseline_records: List[Dict[str, Any]],
) -> pd.DataFrame:
    cols: Dict[str, pd.Series] = {"ensemble": ensemble_returns}
    for r in baseline_records:
        key = f"{r['name']}[{r['symbol']}]"
        cols[key] = r["daily_return"]
    df = pd.DataFrame(cols).dropna(how="any")
    return df


def _load_trial_sharpes(path: Optional[str]) -> Optional[np.ndarray]:
    """Load the sweep's trial Sharpes for DSR; None when not supplied."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"--trial_sharpes_json {p} not found; DSR skipped")
        return None
    with open(p) as f:
        payload = json.load(f)
    arr = np.asarray(payload.get("trial_sharpes", []), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        logger.warning(
            f"trial_sharpes has <2 finite entries ({arr.size}); DSR skipped"
        )
        return None
    return arr


def _format_comparison_table(
    ensemble_metrics: Dict[str, float],
    baseline_records: List[Dict[str, Any]],
    pbo: float = float("nan"),
    dsr: float = float("nan"),
) -> str:
    header_cols = (
        "strategy", "total_return", "annualized_return",
        "sharpe_ratio", "psr", "max_drawdown", "calmar_ratio",
    )
    pretty = {
        "total_return": "TotalRet", "annualized_return": "AnnRet",
        "sharpe_ratio": "Sharpe", "psr": "PSR(>0)",
        "max_drawdown": "MaxDD", "calmar_ratio": "Calmar",
    }
    name_width = max(
        28,
        max((len(r["name"]) + len(r["symbol"]) + 3 for r in baseline_records),
            default=28),
    )
    metric_width = 12

    header = (
        f"{'strategy':<{name_width}}"
        + "".join(
            f"{pretty[k]:>{metric_width}}"
            for k in header_cols if k != "strategy"
        )
    )
    lines = ["", "=" * len(header),
             "WFO Comparison vs baselines", header, "-" * len(header)]

    def row(label: str, metrics: Dict[str, float]) -> str:
        return (
            f"{label:<{name_width}}"
            + "".join(
                f"{_fmt_metric(metrics, k):>{metric_width}}"
                for k in header_cols if k != "strategy"
            )
        )

    lines.append(row("ensemble", ensemble_metrics))
    for r in baseline_records:
        label = f"{r['name']} [{r['symbol']}]"
        lines.append(row(label, r["metrics"]))
    lines.append("-" * len(header))
    if np.isfinite(pbo):
        lines.append(
            f"PBO (CSCV, P[IS-best is OOS below-median]): {pbo:.3f}"
        )
    else:
        lines.append("PBO (CSCV): n/a (insufficient aligned observations)")
    if np.isfinite(dsr):
        lines.append(
            f"DSR (deflated vs sweep trial count): {dsr:.3f}"
        )
    else:
        lines.append("DSR: n/a (no --trial_sharpes_json supplied)")
    lines.append("=" * len(header))
    lines.append("")
    return "\n".join(lines)


def _plot_equity(
    equity: pd.DataFrame,
    transaction_log: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    if equity.empty:
        return
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    equity["portfolio_value"].plot(ax=ax1, label="Portfolio", color="blue")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title("WFO Backtest Equity")
    ax1.legend(); ax1.grid(True)
    equity["drawdown"].plot(
        ax=ax2, label="Drawdown", color="red",
    )
    ax2.fill_between(equity.index, 0, equity["drawdown"], alpha=0.3, color="red")
    ax2.set_ylabel("Drawdown")
    ax2.set_ylim(0, max(0.01, equity["drawdown"].max() * 1.1))
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# -- Main entrypoint --------------------------------------------------------


def main():
    args = parse_args()

    training_run_dir = Path(args.training_run)
    if not training_run_dir.is_dir():
        raise FileNotFoundError(
            f"Training run directory not found: {training_run_dir}"
        )

    training_config, fold_dirs_per_symbol = load_training_run(training_run_dir)
    horizon = int(training_config["prediction_horizon"])
    use_sentiment = (
        bool(training_config.get("use_sentiment", False)) and not args.no_sentiment
    )

    if args.symbols:
        requested = {s.strip() for s in args.symbols.split(",") if s.strip()}
        fold_dirs_per_symbol = {
            s: d for s, d in fold_dirs_per_symbol.items() if s in requested
        }

    if not fold_dirs_per_symbol:
        logger.error("No symbols to backtest after filter; aborting")
        return

    experiment_id = init_mlflow(args.experiment)

    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    sentiment_analyzer = SentimentAnalyzer() if use_sentiment else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = RESULTS_DIR / f"wfo_backtest_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"backtest_{timestamp}",
    ):
        log_params_safe({
            "training_run": str(training_run_dir.resolve()),
            "training_timestamp": training_config.get("timestamp"),
            "symbols": ",".join(fold_dirs_per_symbol.keys()),
            "n_splits": training_config.get("n_splits"),
            "purge_horizon": training_config.get("purge_horizon"),
            "embargo_pct": training_config.get("embargo_pct"),
            "expanding": training_config.get("expanding"),
            "horizon": horizon,
            "use_sentiment": use_sentiment,
            "initial_capital": args.initial_capital,
            "position_size": args.position_size,
            "commission_bps": args.commission_bps,
            "spread_bps": args.spread_bps,
            "slippage_coeff": args.slippage_coeff,
            "borrow_bps_annual": args.borrow_bps_annual,
            "order_type": args.order_type,
        })

        all_symbol_results: List[Dict[str, Any]] = []
        for symbol, fold_dirs in fold_dirs_per_symbol.items():
            logger.info(
                f"WFO backtest for {symbol} across {len(fold_dirs)} folds"
            )

            raw_df = data_loader.fetch_historical_data(
                symbol=symbol,
                interval=training_config["timeframe"],
                start_date=training_config["start_date"],
                end_date=training_config.get("end_date"),
            )
            if raw_df.empty:
                logger.warning(f"No data for {symbol}; skipping")
                continue

            # Phase 4 §4.1: dividends credited as cash (prices are split- but
            # not dividend-adjusted). Empty/None when disabled or unavailable.
            symbol_dividends = None
            if not args.no_dividends:
                symbol_dividends = data_loader.fetch_dividends(
                    symbol=symbol,
                    start_date=training_config["start_date"],
                    end_date=training_config.get("end_date"),
                )

            full_df = build_features(
                raw_df=raw_df,
                symbol=symbol,
                feature_engineer=feature_engineer,
                sentiment_analyzer=sentiment_analyzer,
                horizon=horizon,
            )
            target_col = f"target_{horizon}"
            if target_col not in full_df.columns:
                logger.error(
                    f"target_{horizon} missing from features for {symbol}; "
                    "training config mismatch"
                )
                continue
            usable = full_df.dropna(subset=[target_col])

            symbol_out = output_root / symbol
            symbol_out.mkdir(parents=True, exist_ok=True)

            with mlflow.start_run(run_name=symbol, nested=True):
                log_params_safe({"symbol": symbol, "n_folds": len(fold_dirs)})

                first_model = load_fold_ensemble(fold_dirs[0], horizon)
                ensemble_result = run_symbol_wfo(
                    symbol=symbol,
                    model=first_model,
                    fold_dirs=fold_dirs,
                    full_df=full_df,
                    usable=usable,
                    args=args,
                    use_sentiment=use_sentiment,
                    sentiment_analyzer=sentiment_analyzer,
                    load_fold_model=load_fold_ensemble,
                    horizon=horizon,
                    dividends=symbol_dividends,
                )

                log_metrics_safe(
                    ensemble_result["metrics"], prefix="ensemble_concat/"
                )
                for fm in ensemble_result["fold_metrics"]:
                    step = fm.pop("fold")
                    log_metrics_safe(fm, step=step, prefix="ensemble_fold/")

                baseline_records: List[Dict[str, Any]] = []
                pbo_value = float("nan")
                if not args.no_baselines:
                    baseline_records = run_symbol_baselines(
                        symbol=symbol,
                        fold_dirs=fold_dirs,
                        full_df=full_df,
                        usable=usable,
                        args=args,
                        dividends=symbol_dividends,
                    )
                    for r in baseline_records:
                        log_metrics_safe(
                            r["metrics"], prefix=f"{r['name']}_concat/",
                        )

                    returns_matrix = _build_returns_matrix(
                        ensemble_result["equity_curve"]["daily_return"]
                        if "daily_return" in ensemble_result["equity_curve"]
                        else pd.Series(dtype=float),
                        baseline_records,
                    )
                    if (
                        returns_matrix.shape[1] >= 2
                        and returns_matrix.shape[0] >= 20
                    ):
                        pbo_value = probability_backtest_overfitting(
                            returns_matrix, n_splits=10,
                        )
                        log_metrics_safe({"pbo": float(pbo_value)})

                # DSR: deflate the ensemble's daily Sharpe against the sweep's
                # honestly-counted trial set, when one was supplied.
                dsr_value = float("nan")
                trial_sharpes = _load_trial_sharpes(args.trial_sharpes_json)
                eq = ensemble_result["equity_curve"]
                if trial_sharpes is not None and "daily_return" in eq:
                    daily = eq["daily_return"].dropna()
                    dsr_value = deflated_sharpe_ratio(daily, trial_sharpes)
                    log_metrics_safe({
                        "dsr": float(dsr_value),
                        "selected_daily_sharpe": float(periodic_sharpe(daily)),
                        "n_trials": float(trial_sharpes.size),
                    })

                # Render + persist comparison table.
                comparison_text = _format_comparison_table(
                    ensemble_result["metrics"], baseline_records,
                    pbo=pbo_value, dsr=dsr_value,
                )
                print(comparison_text)
                with open(symbol_out / "report.txt", "w") as f:
                    f.write(comparison_text)

                # Save equity curve + transaction log + per-fold metrics.
                if not ensemble_result["equity_curve"].empty:
                    ensemble_result["equity_curve"].to_csv(
                        symbol_out / "equity_curve.csv"
                    )
                    _plot_equity(
                        ensemble_result["equity_curve"],
                        ensemble_result["transaction_log"],
                        symbol_out / "equity_plot.png",
                    )
                if ensemble_result["transaction_log"]:
                    pd.DataFrame(ensemble_result["transaction_log"]).to_csv(
                        symbol_out / "transaction_log.csv", index=False,
                    )

                metrics_payload = {
                    "ensemble": {
                        k: float(v)
                        for k, v in ensemble_result["metrics"].items()
                        if isinstance(v, (int, float))
                    },
                    "ensemble_fold_metrics": ensemble_result["fold_metrics"],
                    "baselines": [
                        {
                            "name": r["name"],
                            "symbol": r["symbol"],
                            "metrics": {
                                k: float(v) for k, v in r["metrics"].items()
                                if isinstance(v, (int, float))
                            },
                            "fold_metrics": r["fold_metrics"],
                        }
                        for r in baseline_records
                    ],
                    "pbo": float(pbo_value) if np.isfinite(pbo_value) else None,
                    "dsr": float(dsr_value) if np.isfinite(dsr_value) else None,
                }
                with open(symbol_out / "metrics.json", "w") as f:
                    json.dump(metrics_payload, f, indent=2, default=str)

                log_artifact_dir(symbol_out)

                all_symbol_results.append({
                    "symbol": symbol,
                    "metrics": metrics_payload,
                })

        # Aggregated payload.
        with open(output_root / "summary.json", "w") as f:
            json.dump(
                {
                    "training_run": str(training_run_dir.resolve()),
                    "timestamp": timestamp,
                    "symbols": all_symbol_results,
                },
                f, indent=2, default=str,
            )
        log_artifact_dir(output_root, artifact_path="summary")

    logger.info(f"WFO backtest complete; artifacts under {output_root}")


if __name__ == "__main__":
    main()
