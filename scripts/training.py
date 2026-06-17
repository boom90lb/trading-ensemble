#!/usr/bin/env python
"""Train the time series ensemble model with Purged Walk-Forward CV.

Phase 2.2: replaces the previous one-shot 80/20 split with an outer
PurgedWalkForward loop per symbol. Each fold:

  1. Slices the per-symbol enhanced frame to (train_idx, test_idx).
  2. Refits FeatureEngineer scalers + clip_bounds on the fold's TRAIN
     slice only (the leakage fix from Phase 2.8 audit b only protects
     against transform-time leaks if fit_scalers is called per fold).
  3. Builds a fresh EnsembleModel; policy members' ``*_X_train_with_close``
     kwargs are the fold's unscaled TRAIN slice.
  4. Fits, evaluates on the fold's TEST slice, persists the fold's model
     under ``runs/{run_name}/{symbol}/fold_{k}/ensemble_model``.
  5. Logs per-fold metrics to MLflow under nested-run ``{symbol}`` with
     ``step=fold_idx``; aggregated mean/std metrics logged on the same run.

The outer MLflow run captures top-level params (n_splits, purge_horizon,
embargo_pct, …) so a downstream backtest can replay the same fold structure.
"""

import argparse
import hashlib
import json
import logging
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.config import (
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_TRAINING_CONFIG,
    EnsembleConfig,
    ModelConfig,
    TrainingConfig,
)
from src.data_loader import DataLoader, _tz_aware_filter
from src.features import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.sentiment_analysis import SentimentAnalyzer
from src.logging_utils import configure_logging, get_symbol_logger
from src.tracking.mlflow_utils import (
    init_mlflow,
    log_artifact_dir,
    log_metrics_safe,
    log_params_safe,
)
from src.validation.walk_forward import PurgedWalkForward

# Suppress specific Pandas warnings (like DataFrame fragmentation)
try:
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
except AttributeError:
    warnings.warn(
        "Could not specifically filter PerformanceWarning. Filtering FutureWarning instead.",
        FutureWarning,
    )
    warnings.simplefilter(action="ignore", category=FutureWarning)

# Logging is configured in main() via configure_logging() so --verbose and the
# per-run log file are honored; this module-level logger is the fallback name.
logger = logging.getLogger(__name__)

R0_QUARANTINED_MODELS = frozenset({"lstm"})
DEFAULT_MODEL_SELECTION = "arima,prophet,xgboost,lstm_ppo,xlstm_ppo,xlstm_grpo"
FOLD_METADATA_SCHEMA_VERSION = 2


def parse_model_names(raw: str) -> List[str]:
    """Parse and validate a comma-separated production model list."""
    model_names = [n.strip() for n in raw.split(",") if n.strip()]
    quarantined = sorted(set(model_names) & R0_QUARANTINED_MODELS)
    if quarantined:
        raise ValueError(
            "R0 quarantine: forecast LSTM is disabled until the R1 rebuild "
            f"removes label-in-window leakage and inference zero-fill. Got: {quarantined}"
        )
    return model_names


def reject_sentiment_flag(enabled: bool, *, surface: str) -> None:
    """Fail loud on production sentiment paths quarantined by R0."""
    if enabled:
        raise ValueError(
            f"R0 quarantine: sentiment is disabled for {surface} until the "
            "paginated point-in-time FinBERT path replaces the legacy keyword scorer."
        )


def index_sha256(index: pd.Index) -> str:
    """Stable SHA-256 over a DatetimeIndex's exact replay order."""
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is not None:
        normalized = dt_index.tz_convert("UTC")
    else:
        normalized = dt_index
    values = np.asarray(normalized.asi8, dtype="<i8")
    h = hashlib.sha256()
    h.update(str(dt_index.tz).encode("utf-8"))
    h.update(len(values).to_bytes(8, byteorder="little", signed=False))
    h.update(values.tobytes())
    return h.hexdigest()


def index_utc_ns(index: pd.Index) -> List[int]:
    """Exact DatetimeIndex membership as UTC nanoseconds, preserving order."""
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is not None:
        normalized = dt_index.tz_convert("UTC")
    else:
        normalized = dt_index
    return [int(v) for v in np.asarray(normalized.asi8, dtype="<i8")]


def index_date_span(index: pd.Index) -> Optional[Dict[str, str]]:
    """Inclusive start/end span for fold metadata."""
    if len(index) == 0:
        return None
    dt_index = pd.DatetimeIndex(index)
    return {
        "start": pd.Timestamp(dt_index[0]).isoformat(),
        "end": pd.Timestamp(dt_index[-1]).isoformat(),
    }


def build_fold_metadata(
    *,
    symbol: str,
    horizon: int,
    target_col: str,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    model_feature_columns: List[str],
) -> Dict[str, Any]:
    """Create the replay contract persisted beside a fold model."""
    return {
        "schema_version": FOLD_METADATA_SCHEMA_VERSION,
        "symbol": symbol,
        "horizon": int(horizon),
        "target_column": target_col,
        "train_date_span": index_date_span(train_raw.index),
        "test_date_span": index_date_span(test_raw.index),
        "train_index_utc_ns": index_utc_ns(train_raw.index),
        "test_index_utc_ns": index_utc_ns(test_raw.index),
        "train_index_sha256": index_sha256(train_raw.index),
        "test_index_sha256": index_sha256(test_raw.index),
        "model_feature_columns": list(model_feature_columns),
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the time series ensemble model with Purged Walk-Forward CV.",
    )

    parser.add_argument(
        "--symbols", type=str,
        default=",".join(DEFAULT_TRAINING_CONFIG.symbols),
        help="Comma-separated list of symbols to train on",
    )
    parser.add_argument(
        "--universe", type=str, default=None,
        help="Path to a universe file (one symbol per line, '#' comments); "
             "overrides --symbols when given (Phase 4 §4.3).",
    )
    parser.add_argument(
        "--universe_asof", type=str, default=None,
        help="As-of date YYYY-MM-DD: drop symbols with no data at/before this "
             "date (point-in-time universe guard; best-effort survivorship).",
    )
    parser.add_argument(
        "--timeframe", type=str, default=DEFAULT_TRAINING_CONFIG.timeframe,
        help="Time interval (e.g., 1d, 1h)",
    )
    parser.add_argument(
        "--start_date", type=str, default=DEFAULT_TRAINING_CONFIG.start_date,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument("--end_date", type=str, default=None,
                        help="End date in YYYY-MM-DD format")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forecast horizon (bars)")
    parser.add_argument("--use_sentiment", action="store_true",
                        help="Use sentiment analysis features")
    parser.add_argument(
        "--models", type=str,
        default=DEFAULT_MODEL_SELECTION,
        help="Comma-separated list of ensemble member names",
    )
    parser.add_argument("--optimize", action="store_true",
                        help="Run hyperparameter optimization")
    parser.add_argument("--rl_timesteps", type=int, default=100000,
                        help="Total timesteps for each PPO member's training")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration if available")
    parser.add_argument("--checkpoint_freq", type=int, default=50000,
                        help="Checkpoint frequency for RL training")
    parser.add_argument("--extended_features", action="store_true",
                        help="Use extended feature set")

    # WFO knobs
    parser.add_argument("--n_splits", type=int, default=DEFAULT_TRAINING_CONFIG.n_splits,
                        help="Number of PurgedWalkForward outer folds (>=2)")
    parser.add_argument("--purge_horizon", type=int, default=None,
                        help="Bars to drop between train/test (default: --horizon)")
    parser.add_argument("--embargo_pct", type=float, default=DEFAULT_TRAINING_CONFIG.embargo_pct,
                        help="Embargo as a fraction of total bars (AFML §7.4)")
    parser.add_argument("--rolling", action="store_true",
                        help="Use rolling-window train (default: expanding)")

    # MLflow
    parser.add_argument("--experiment", type=str, default="trading_ensemble",
                        help="MLflow experiment name")

    parser.add_argument("--verbose", action="store_true",
                        help="Console DEBUG logging (the log file is always DEBUG)")

    return parser.parse_args()


def select_rl_features(X_train: pd.DataFrame) -> List[str]:
    """Pick a compact feature subset for the RL agents.

    Priority features focused on price, momentum, vol, and volume. If the
    pipeline's column set is sparse, we top up with the most recent lagged
    versions of the available base columns. Capped at 30 features to keep
    the policy network manageable.
    """
    priority_features = [
        "open", "high", "low", "close", "volume",
        "ma5", "ma20", "ema5", "ema20",
        "macd", "macd_hist", "rsi_14",
        "price_change", "price_change_5", "price_change_10",
        "volatility_5", "volatility_20", "bb_width", "atr_14",
        "volume_change", "volume_ma5", "mfi_14",
        "daily_range", "gap", "roc_5", "roc_10",
    ]
    available = [f for f in priority_features if f in X_train.columns]

    if len(available) < 10:
        for col in [c for c in X_train.columns
                    if any(c.startswith(f) for f in ["close", "rsi", "macd", "price"])]:
            lag_features = [f for f in X_train.columns if f.startswith(f"{col}_lag")]
            recent = [f for f in lag_features if f.endswith("lag1") or f.endswith("lag2")]
            available.extend(recent)

    if len(available) > 30:
        essential = [f for f in available
                     if f in ["close", "price_change", "rsi_14",
                              "macd", "bb_width", "volatility_20"]]
        volume = [f for f in available if "volume" in f][:3]
        mas = [f for f in available
               if f.startswith("ma") or f.startswith("ema")][:4]
        additional = [f for f in available
                      if f not in essential + volume + mas]
        additional = sorted(
            additional,
            key=lambda x: priority_features.index(x) if x in priority_features else 999,
        )[: 30 - len(essential) - len(volume) - len(mas)]
        available = essential + volume + mas + additional

    logger.info(f"Selected {len(available)} features for RL agents")
    return available


def check_gpu_availability() -> bool:
    """Check whether a CUDA GPU is visible via PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"PyTorch GPU available: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass
    logger.warning("No PyTorch GPU available, using CPU")
    return False


def clean_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative cleanup applied causally on the full enhanced frame.

    * Inf -> NaN (row-wise, causal).
    * Drop columns with >30% NaN (column-set decision; barely informative).
    * Forward-fill remaining NaNs (past -> future, causal).
    * Final fill-with-0 to handle warmup-period leading NaNs.

    Outlier clipping is intentionally NOT done here — that lives in
    ``FeatureEngineer.transform_features`` which uses train-only bounds
    (Phase 2.8 audit b). Doing 5×IQR clipping on the full frame here
    would re-introduce a row-level future-leak via the IQR computation.
    """
    # Phase 4 §4.2: the whole pipeline (point-in-time sentiment join, date
    # filtering) assumes an ET-localized index. Fail loud if feature building
    # dropped the tz rather than letting naive timestamps leak downstream.
    if isinstance(df.index, pd.DatetimeIndex):
        assert df.index.tz is not None, "training frame index lost its timezone"
    result = df.replace([np.inf, -np.inf], np.nan)
    nan_pct = result.isna().mean()
    high_nan_cols = nan_pct[nan_pct > 0.3].index.tolist()
    if high_nan_cols:
        logger.warning(f"Dropping {len(high_nan_cols)} cols with >30% NaN")
        result = result.drop(columns=high_nan_cols)
    result = result.ffill().fillna(0)
    return result


def build_features(
    raw_df: pd.DataFrame,
    symbol: str,
    feature_engineer: FeatureEngineer,
    sentiment_analyzer: Optional[SentimentAnalyzer],
    horizon: int,
) -> pd.DataFrame:
    """Causally build the full enhanced frame for one symbol.

    Technical indicators (rolling/ewm/pct_change) are causal so they may
    be computed once on the full series. Sentiment join is point-in-time
    (Phase 2.8 fix). FE scaling + outlier clipping happens per fold inside
    ``train_symbol_wfo`` so this frame is intentionally UNSCALED.
    """
    df = feature_engineer.create_features(raw_df)
    df = feature_engineer.create_lagged_features(df, [1, 2, 5, 10])
    df = feature_engineer.create_target_variable(df, "close", horizon)

    if sentiment_analyzer is not None:
        sentiment = sentiment_analyzer.create_sentiment_features(
            symbol, pd.DatetimeIndex(df.index)  # type: ignore
        )
        if not sentiment.empty:
            df = df.join(sentiment)

    df = clean_data_for_training(df)
    return df


def build_policy_fit_kwargs(
    model_names: List[str],
    X_train: pd.DataFrame,
    train_raw: pd.DataFrame,
    args,
    gpu_available: bool,
    fold_dir: Path,
    symbol: str,
) -> Dict[str, Any]:
    """Assemble per-policy fit kwargs for one fold.

    ``train_raw`` is the unscaled fold-TRAIN slice. Each policy member's
    ``*_X_train_with_close`` kwarg points to this same frame so the
    member's RL environment sees the actual price series with no
    scaler-induced distortion. The ensemble's ``_oof_policy`` further
    subdivides this per inner-fold for meta-learner OOF.
    """
    rl_features: Optional[List[str]] = None
    fit_kwargs: Dict[str, Any] = {}

    def _device() -> str:
        return "cuda" if (args.gpu and gpu_available) else "cpu"

    def _features() -> List[str]:
        nonlocal rl_features
        if rl_features is None:
            rl_features = select_rl_features(X_train)
        return rl_features

    for name in model_names:
        if name == "lstm_ppo":
            fit_kwargs["lstm_ppo_timesteps"] = args.rl_timesteps
            fit_kwargs["lstm_ppo_checkpoint_freq"] = args.checkpoint_freq
            fit_kwargs["lstm_ppo_device"] = _device()
            fit_kwargs["lstm_ppo_features"] = _features()
            fit_kwargs["lstm_ppo_X_train_with_close"] = train_raw
            fit_kwargs["tensorboard_log_path"] = str(
                fold_dir / f"tensorboard_logs_{symbol}_lstm_ppo"
            )
        elif name == "xlstm_ppo":
            fit_kwargs["xlstm_ppo_timesteps"] = args.rl_timesteps
            fit_kwargs["xlstm_ppo_checkpoint_freq"] = args.checkpoint_freq
            fit_kwargs["xlstm_ppo_device"] = _device()
            fit_kwargs["xlstm_ppo_features"] = _features()
            fit_kwargs["xlstm_ppo_X_train_with_close"] = train_raw
            fit_kwargs["xlstm_ppo_tensorboard_log_path"] = str(
                fold_dir / f"tensorboard_logs_{symbol}_xlstm_ppo"
            )
        elif name == "xlstm_grpo":
            fit_kwargs["xlstm_grpo_updates"] = max(1, args.rl_timesteps // 100)
            fit_kwargs["xlstm_grpo_device"] = _device()
            fit_kwargs["xlstm_grpo_features"] = _features()
            fit_kwargs["xlstm_grpo_X_train_with_close"] = train_raw
            fit_kwargs["xlstm_grpo_log_path"] = str(
                fold_dir / f"xlstm_grpo_logs_{symbol}"
            )

    return fit_kwargs


def aggregate_fold_metrics(per_fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean (+ ddof=1 std when ≥2 folds) over the per-fold metric dicts.

    Non-finite values are dropped before aggregation. A metric absent
    from any fold simply won't appear in the output.
    """
    all_keys: set = set()
    for m in per_fold_metrics:
        all_keys.update(m.keys())
    out: Dict[str, float] = {}
    for k in all_keys:
        vals = [m[k] for m in per_fold_metrics
                if k in m and np.isfinite(m[k])]
        if not vals:
            continue
        out[f"mean_{k}"] = float(np.mean(vals))
        if len(vals) > 1:
            out[f"std_{k}"] = float(np.std(vals, ddof=1))
    return out


def train_symbol_wfo(
    symbol: str,
    full_df: pd.DataFrame,
    target_col: str,
    splitter: PurgedWalkForward,
    model_configs: List[ModelConfig],
    model_names: List[str],
    training_config: TrainingConfig,
    args,
    gpu_available: bool,
    feature_engineer: FeatureEngineer,
    symbol_dir: Path,
    weighting_strategy: str = "dynamic",
    target_vol: float = 1.0,
    position_cap: float = 1.0,
) -> List[Dict[str, float]]:
    """Run the WFO outer loop for one symbol.

    Returns the list of per-fold metric dicts (one per successfully fit
    fold). Per-fold ensemble pickles land in
    ``{symbol_dir}/fold_{k}/ensemble_model/``; a ``fold_metrics.json``
    summarizing per-fold + aggregated metrics is written to
    ``symbol_dir``.

    ``weighting_strategy``/``target_vol``/``position_cap`` parametrize the
    EnsembleConfig built per fold so a hyperparameter sweep (Phase 2.7) can
    vary them; defaults reproduce the pre-sweep behavior (dynamic weighting,
    unit vol target, unit cap).
    """
    usable = full_df.dropna(subset=[target_col]).copy()
    feature_cols = [
        c for c in usable.columns
        if "target_" not in c and "direction_" not in c
    ]

    log_params_safe({
        "symbol": symbol,
        "n_usable_rows": len(usable),
        "n_features": len(feature_cols),
    })

    per_fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(usable)):
        logger.info(
            f"[{symbol}] fold {fold_idx}: train={len(train_idx)} rows "
            f"({train_idx[0]}->{train_idx[-1]}), "
            f"test={len(test_idx)} rows ({test_idx[0]}->{test_idx[-1]})"
        )
        fold_dir = symbol_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_raw = usable.iloc[train_idx]
        test_raw = usable.iloc[test_idx]

        # Fit scalers on TRAIN ONLY for this fold; clip_bounds[symbol]
        # is overwritten on each call so per-fold leakage is avoided.
        feature_engineer.fit_scalers(train_raw, symbol)
        train_scaled = feature_engineer.transform_features(
            train_raw, symbol, is_train=True,
        )
        test_scaled = feature_engineer.transform_features(
            test_raw, symbol, is_train=False,
        )

        X_train = train_scaled.drop(
            columns=[c for c in train_scaled.columns
                     if "target_" in c or "direction_" in c]
        )
        y_train = train_scaled[target_col]
        X_test = test_scaled.drop(
            columns=[c for c in test_scaled.columns
                     if "target_" in c or "direction_" in c]
        )
        y_test = test_scaled[target_col]
        # Defensive: re-align if scalers dropped any columns on test.
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        model_feature_columns = X_train.columns.tolist()

        ensemble = EnsembleModel(
            target_column="close",
            horizon=training_config.prediction_horizon,
            config=EnsembleConfig(
                models=model_configs,
                weighting_strategy=weighting_strategy,
                target_vol=target_vol,
                position_cap=position_cap,
            ),
        )

        fit_kwargs = build_policy_fit_kwargs(
            model_names=model_names,
            X_train=X_train,
            train_raw=train_raw,
            args=args,
            gpu_available=gpu_available,
            fold_dir=fold_dir,
            symbol=symbol,
        )

        try:
            ensemble.fit(X_train, y_train, **fit_kwargs)
        except Exception as e:
            logger.error(
                f"Error fitting ensemble for {symbol} fold {fold_idx}: {e}",
                exc_info=True,
            )
            continue

        metrics = ensemble.evaluate(X_test, y_test)
        per_fold_metrics.append(metrics)
        log_metrics_safe(metrics, step=fold_idx, prefix="fold/")
        log_metrics_safe(
            {f"{name}_weight": float(w)
             for name, w in ensemble.weights.items()},
            step=fold_idx,
            prefix="fold_weight/",
        )

        ensemble.save(directory=fold_dir / "ensemble_model")
        feature_engineer.save_state(
            fold_dir / "feature_engineer_state.pkl",
            symbol=symbol,
            feature_columns=model_feature_columns,
        )
        with open(fold_dir / "fold_metadata.json", "w") as f:
            json.dump(
                build_fold_metadata(
                    symbol=symbol,
                    horizon=training_config.prediction_horizon,
                    target_col=target_col,
                    train_raw=train_raw,
                    test_raw=test_raw,
                    model_feature_columns=model_feature_columns,
                ),
                f,
                indent=2,
            )
        # Persist the fold's split indices so the backtest WFO loop can
        # iterate the same fold structure deterministically.
        np.savez(
            fold_dir / "split_idx.npz",
            train_idx=train_idx,
            test_idx=test_idx,
        )

    if per_fold_metrics:
        aggregated = aggregate_fold_metrics(per_fold_metrics)
        log_metrics_safe(aggregated)
        out_payload = {
            "per_fold": [
                {k: float(v) for k, v in m.items()}
                for m in per_fold_metrics
            ],
            "aggregated": {k: float(v) for k, v in aggregated.items()},
        }
        with open(symbol_dir / "fold_metrics.json", "w") as f:
            json.dump(out_payload, f, indent=2)

    return per_fold_metrics


def load_universe(path: str) -> List[str]:
    """Read a tracked universe file: one symbol per line, '#' comments and
    blank lines ignored (Phase 4 §4.3).

    Order is preserved and duplicates dropped (first occurrence wins) so the
    file reads as the canonical investable set for a given as-of date.
    """
    symbols: List[str] = []
    for line in Path(path).read_text().splitlines():
        token = line.split("#", 1)[0].strip()
        if token and token not in symbols:
            symbols.append(token)
    return symbols


def filter_universe_asof(
    all_data: Dict[str, pd.DataFrame], asof: Optional[str]
) -> Dict[str, pd.DataFrame]:
    """Drop symbols with no data at or before the as-of date.

    A symbol whose first bar is after ``asof`` was not investable then, so
    including it is a forward-looking universe bias. This is a best-effort
    point-in-time guard on the *included* set; it does NOT recover delisted
    names (true survivorship-free construction needs a delisting database —
    documented as out of scope in the README). ``asof=None`` is a passthrough.
    """
    if not asof:
        return all_data
    asof_ts = _tz_aware_filter(asof)
    kept: Dict[str, pd.DataFrame] = {}
    for symbol, df in all_data.items():
        has_early_data = (
            isinstance(df.index, pd.DatetimeIndex)
            and len(df) > 0
            and df.index.min() <= asof_ts
        )
        if has_early_data:
            kept[symbol] = df
        else:
            logger.warning(
                f"Dropping {symbol} from universe: no data at/before as-of {asof}"
            )
    return kept


def main():
    """Main function."""
    args = parse_args()
    reject_sentiment_flag(args.use_sentiment, surface="training")
    model_names = parse_model_names(args.models)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configure_logging(verbose=args.verbose, run_name="training")

    gpu_available = False
    if args.gpu:
        gpu_available = check_gpu_availability()
        if not gpu_available:
            logger.warning("GPU acceleration requested but no GPU available")

    if args.universe:
        symbols = load_universe(args.universe)
        logger.info(f"Loaded {len(symbols)} symbols from universe {args.universe}")
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("Empty symbol universe; aborting")
        return
    # Keep run-dir names bounded for large universes.
    symbols_str = "_".join(symbols) if len(symbols) <= 8 else f"{len(symbols)}symbols"
    run_name = f"run_{symbols_str}_{timestamp}"
    output_dir = Path("runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving run artifacts to: {output_dir.resolve()}")

    training_config = TrainingConfig(
        symbols=symbols,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        prediction_horizon=args.horizon,
        use_sentiment=args.use_sentiment,
        optimize=args.optimize,
        n_splits=args.n_splits,
        purge_horizon=args.purge_horizon,
        embargo_pct=args.embargo_pct,
        expanding=not args.rolling,
    )
    logger.info(f"Training configuration: {training_config}")

    experiment_id = init_mlflow(args.experiment)

    data_loader = DataLoader()
    all_data = data_loader.fetch_training_data(training_config)
    all_data = filter_universe_asof(all_data, args.universe_asof)
    if not all_data:
        logger.error("No data fetched for any symbol; aborting")
        return

    feature_engineer = FeatureEngineer()
    sentiment_analyzer = None

    enhanced_data: Dict[str, pd.DataFrame] = {}
    for symbol, raw_df in all_data.items():
        get_symbol_logger(logger, symbol).info("Building features")
        enhanced_data[symbol] = build_features(
            raw_df=raw_df,
            symbol=symbol,
            feature_engineer=feature_engineer,
            sentiment_analyzer=sentiment_analyzer,
            horizon=training_config.prediction_horizon,
        )

    model_configs = [
        ModelConfig(
            name=name,
            enabled=True,
            weight=DEFAULT_MODEL_WEIGHTS.get(name, 1.0),
        )
        for name in model_names
    ]

    splitter = PurgedWalkForward(
        n_splits=training_config.n_splits,
        purge_horizon=training_config.effective_purge_horizon,
        embargo_pct=training_config.embargo_pct,
        expanding=training_config.expanding,
    )

    target_col = f"target_{training_config.prediction_horizon}"

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"training_{timestamp}",
    ):
        log_params_safe({
            "symbols": ",".join(symbols),
            "timeframe": args.timeframe,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "horizon": args.horizon,
            "use_sentiment": args.use_sentiment,
            "models": ",".join(model_names),
            "n_splits": training_config.n_splits,
            "purge_horizon": training_config.effective_purge_horizon,
            "embargo_pct": training_config.embargo_pct,
            "expanding": training_config.expanding,
            "rl_timesteps": args.rl_timesteps,
            "gpu": args.gpu,
            "output_dir": str(output_dir.resolve()),
        })

        run_summary: Dict[str, Any] = {}
        for symbol, full_df in enhanced_data.items():
            symbol_dir = output_dir / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            get_symbol_logger(logger, symbol).info(
                "Training WFO folds (%d bars)", len(full_df)
            )

            with mlflow.start_run(run_name=symbol, nested=True):
                per_fold_metrics = train_symbol_wfo(
                    symbol=symbol,
                    full_df=full_df,
                    target_col=target_col,
                    splitter=splitter,
                    model_configs=model_configs,
                    model_names=model_names,
                    training_config=training_config,
                    args=args,
                    gpu_available=gpu_available,
                    feature_engineer=feature_engineer,
                    symbol_dir=symbol_dir,
                )
                log_artifact_dir(symbol_dir)
                run_summary[symbol] = {
                    "n_folds_fit": len(per_fold_metrics),
                }

        # Save the training config + run summary for the backtest WFO loop.
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    **asdict(training_config),
                    "models": model_names,
                    "rl_timesteps": args.rl_timesteps,
                    "gpu": args.gpu,
                    "timestamp": timestamp,
                    "experiment": args.experiment,
                    "run_summary": run_summary,
                },
                f, indent=2, default=str,
            )

    logger.info("Training complete")


if __name__ == "__main__":
    main()
