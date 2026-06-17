#!/usr/bin/env python
"""Multi-seed RL overfitting study (Phase 3.2 / M10).

Single-seed RL Sharpes are noise-dominated. This opt-in driver re-fits each
RL agent over ``seed ∈ {0,1,2,3,4}`` and reports the **mean ± std of net
WFO-OOS periodic Sharpe** across seeds as the primary metric (DSR over the
seed set is a secondary line). It closes M10 part 1; parts 2 and 3 (held-out
eval on the same WFO test folds; reward = price[t+1]-price[t] only) are
already structural and are pinned by tests rather than rebuilt here.

Design (scoped via AskUserQuestion, Phase 3.2):
  * **Bare agents, no meta-learner.** Each (member, symbol, seed) fits a
    1-member *static-weight* EnsembleModel whose sole member is the seeded RL
    agent. Static weighting skips ``_fit_meta_learner_weights`` → no
    ``_oof_policy`` ``×meta_cv_folds`` RL-refit blowup: exactly one RL fit per
    fold per seed.
  * **Same WFO test folds as forecast members.** Symbols and per-fold split
    indices are read from an existing ``--training_run`` (the contract
    ``scripts/backtest.py`` already uses), so RL ``predict()`` is evaluated on
    the same held-out test slices — never on a training env.
  * **Net periodic Sharpe via the canonical path.** Each seed's positions are
    routed through ``scripts.backtest.run_symbol_wfo`` →
    ``TradingStrategy.backtest`` → ``ExecutionModel`` (commission, half-spread,
    linear impact, borrow, dividends, fill-at-next-open). No re-implemented
    cost model.

Cost ≈ ``|members| × |symbols| × |folds| × |seeds|`` bare RL fits. Opt-in;
never on the main training path (which stays single-seed via the agent
default of 42).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from scripts.backtest import (
    load_training_run,
    run_symbol_wfo,
)
from scripts.sweep import combine_symbol_returns
from scripts.training import build_features, select_rl_features
from src.config import EnsembleConfig, ModelConfig
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.models.ensemble import EnsembleModel
from src.models.registry import POLICY_MODELS
from src.sentiment_analysis import SentimentAnalyzer
from src.tracking.mlflow_utils import init_mlflow, log_metrics_safe, log_params_safe
from src.validation.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    periodic_sharpe,
    probabilistic_sharpe_ratio,
)
from src.validation.trials import (
    current_git_commit,
    emit_research_claim_packet,
    summary_claim_fields,
    validate_claim_packet_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MEMBERS = ["lstm_ppo", "xlstm_ppo", "xlstm_grpo"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed RL overfitting study (Phase 3.2 / M10).",
    )
    parser.add_argument(
        "--training_run", type=str, required=True,
        help="Path to runs/{run_name}/ produced by scripts/training.py "
             "(supplies the symbol list + per-fold split indices).",
    )
    parser.add_argument(
        "--members", type=str, default=",".join(DEFAULT_MEMBERS),
        help="Comma-separated RL members to study (subset of "
             f"{sorted(POLICY_MODELS)}).",
    )
    parser.add_argument(
        "--seeds", type=str, default="0,1,2,3,4",
        help="Comma-separated PRNG seeds (one trial per seed).",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Optional subset of the training-run symbols.",
    )
    parser.add_argument(
        "--rl_timesteps", type=int, default=100000,
        help="PPO timesteps per fit (xlstm_grpo uses //100 updates, "
             "mirroring scripts/training.py).",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--no_dividends", action="store_true",
        help="Skip dividend cash credit (avoids /dividends API calls).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write rl_seed_eval.json (default: the training_run dir).",
    )

    # Execution config — consumed by run_symbol_wfo's TradingStrategy. Defaults
    # match scripts/backtest.py so the net-Sharpe convention is identical.
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
    parser.add_argument("--signal_threshold", type=float, default=0.1)

    parser.add_argument(
        "--experiment", type=str, default="trading_ensemble_rl_seed_eval",
    )
    return parser.parse_args()


def make_seeded_fold_model_factory(
    member: str,
    seed: int,
    usable: pd.DataFrame,
    target_col: str,
    horizon: int,
    feature_engineer: FeatureEngineer,
    symbol: str,
    rl_timesteps: int,
    device: str,
) -> Callable[[Path, int], Optional[EnsembleModel]]:
    """Return a ``(fold_dir, horizon) -> EnsembleModel`` factory.

    The returned closure fits a fresh 1-member *static* EnsembleModel — whose
    sole member is ``member`` constructed at ``seed`` — on the fold's TRAIN
    slice (read from ``fold_dir/split_idx.npz``). Scalers/clip-bounds are fit
    on TRAIN only per fold, mirroring ``train_symbol_wfo`` so the Phase 2.8
    leak guard applies. Static weighting → no meta-learner → one RL fit/fold.
    """

    def _make(fold_dir: Path, _horizon: int) -> Optional[EnsembleModel]:
        sidx_path = fold_dir / "split_idx.npz"
        if not sidx_path.exists():
            logger.warning(f"Missing split_idx.npz in {fold_dir}; skipping fold")
            return None
        train_idx = np.load(sidx_path)["train_idx"]
        if train_idx.size == 0:
            return None

        train_raw = usable.iloc[train_idx]

        # Per-fold TRAIN-only scaling (overwrites symbol clip_bounds → the
        # Phase 2.8 audit-b leak guard takes effect per fold).
        feature_engineer.fit_scalers(train_raw, symbol)
        train_scaled = feature_engineer.transform_features(
            train_raw, symbol, is_train=True,
        )
        X_train = train_scaled.drop(
            columns=[c for c in train_scaled.columns
                     if "target_" in c or "direction_" in c]
        )
        y_train = train_scaled[target_col]

        ensemble = EnsembleModel(
            target_column="close",
            horizon=horizon,
            config=EnsembleConfig(
                models=[ModelConfig(name=member, weight=1.0)],
                weighting_strategy="static",
            ),
        )
        # Override the member instance with a seeded one (the ctor built it at
        # the default seed). _create_model threads seed → jax.random.key(seed).
        seeded = ensemble._create_model(member, seed=seed)
        if seeded is None:
            logger.error(f"Could not construct seeded {member}; skipping fold")
            return None
        ensemble.models[member] = seeded

        rl_features = select_rl_features(X_train)
        fit_kwargs: Dict[str, Any] = {
            f"{member}_features": rl_features,
            f"{member}_X_train_with_close": train_raw,
            f"{member}_device": device,
        }
        if member == "xlstm_grpo":
            fit_kwargs["xlstm_grpo_updates"] = max(1, rl_timesteps // 100)
        else:
            fit_kwargs[f"{member}_timesteps"] = rl_timesteps

        ensemble.fit(X_train, y_train, **fit_kwargs)
        return ensemble

    return _make


def eval_member_seed(
    member: str,
    seed: int,
    symbol_to_folds: Dict[str, List[Path]],
    symbol_to_frames: Dict[str, pd.DataFrame],
    target_col: str,
    horizon: int,
    feature_engineer: FeatureEngineer,
    args,
    sentiment_analyzer: Optional[SentimentAnalyzer],
    use_sentiment: bool,
    symbol_to_dividends: Dict[str, Optional[pd.Series]],
) -> pd.Series:
    """Net daily-return series (equal-weight across symbols) for one seed.

    For each symbol, builds a per-fold seeded-model factory and drives it
    through ``run_symbol_wfo``. fold 0 uses the ``model=`` arg; folds ≥1 use
    the ``load_fold_model`` callback — so we pass the SAME factory to both,
    making every fold (including 0) a freshly-fit seeded agent on its own
    TRAIN slice and evaluated on its own held-out TEST slice.
    """
    device = "cuda" if args.gpu else "cpu"
    per_symbol_returns: Dict[str, pd.Series] = {}

    for symbol, fold_dirs in symbol_to_folds.items():
        usable = symbol_to_frames[symbol]
        full_df = usable  # run_symbol_wfo slices by date; usable carries the index
        factory = make_seeded_fold_model_factory(
            member=member, seed=seed, usable=usable, target_col=target_col,
            horizon=horizon, feature_engineer=feature_engineer, symbol=symbol,
            rl_timesteps=args.rl_timesteps, device=device,
        )
        fold0_model = factory(fold_dirs[0], horizon)
        if fold0_model is None:
            logger.warning(f"[{member} seed={seed} {symbol}] fold 0 unbuildable; skipping symbol")
            continue

        result = run_symbol_wfo(
            symbol=symbol,
            model=fold0_model,
            fold_dirs=fold_dirs,
            full_df=full_df,
            usable=usable,
            args=args,
            use_sentiment=use_sentiment,
            sentiment_analyzer=sentiment_analyzer,
            load_fold_model=factory,
            horizon=horizon,
            dividends=symbol_to_dividends.get(symbol),
        )
        eq = result.get("equity_curve")
        if eq is None or eq.empty or "daily_return" not in eq:
            continue
        per_symbol_returns[symbol] = eq["daily_return"].dropna()

    return combine_symbol_returns(per_symbol_returns)


def summarize_member(
    member: str, seed_returns: Dict[int, pd.Series],
) -> Dict[str, Any]:
    """Aggregate one member's per-seed return series into the metrics of record.

    Primary: mean ± std of per-seed periodic (daily) net Sharpe. Secondary:
    DSR over the seed set (the best seed's return series deflated against the
    seed Sharpes as the trial set) + PSR + expected-max — mirrors
    ``summarize_sweep`` but with seeds as the trials.
    """
    seed_sharpe = {
        s: periodic_sharpe(r) for s, r in seed_returns.items() if len(r)
    }
    valid = {s: v for s, v in seed_sharpe.items() if np.isfinite(v)}

    def _f(x: float) -> Optional[float]:
        return float(x) if np.isfinite(x) else None

    out: Dict[str, Any] = {
        "member": member,
        "n_seeds": len(seed_returns),
        "n_valid": len(valid),
        "seed_sharpe": {str(s): float(v) for s, v in seed_sharpe.items()},
        "mean": None, "std": None, "sr_null": None, "psr": None, "dsr": None,
    }
    if not valid:
        return out

    sharpes = np.array(list(valid.values()), dtype=float)
    out["mean"] = float(sharpes.mean())
    # Population-style spread across seeds; ddof=1 (NaN for a single valid seed
    # is correct — no dispersion to estimate from one trial).
    out["std"] = _f(sharpes.std(ddof=1)) if len(sharpes) > 1 else None

    best_seed = max(valid, key=lambda s: valid[s])
    best_returns = seed_returns[best_seed]
    out["sr_null"] = _f(expected_max_sharpe(sharpes))
    out["psr"] = _f(probabilistic_sharpe_ratio(best_returns, sr_benchmark=0.0))
    out["dsr"] = _f(deflated_sharpe_ratio(best_returns, sharpes))
    return out


def main():
    args = parse_args()
    training_run_dir = Path(args.training_run)
    training_config, fold_dirs_per_symbol = load_training_run(training_run_dir)

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    bad = [m for m in members if m not in POLICY_MODELS]
    if bad:
        raise ValueError(f"--members contains non-RL members {bad}; choose from {sorted(POLICY_MODELS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    if args.symbols:
        requested = {s.strip() for s in args.symbols.split(",")}
        fold_dirs_per_symbol = {
            s: d for s, d in fold_dirs_per_symbol.items() if s in requested
        }
    if not fold_dirs_per_symbol:
        raise ValueError("No symbols to evaluate after filtering.")

    horizon = training_config["prediction_horizon"]
    target_col = f"target_{horizon}"
    # Sentiment join mirrors the training run; keyword analyzer is the wired one.
    use_sentiment = bool(training_config.get("use_sentiment", False))
    sentiment_analyzer = SentimentAnalyzer() if use_sentiment else None

    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()

    # Pre-build each symbol's unscaled feature frame + dividends once (reused
    # across every member × seed). build_features is byte-identical to the
    # training/backtest column set.
    symbol_to_frames: Dict[str, pd.DataFrame] = {}
    symbol_to_dividends: Dict[str, Optional[pd.Series]] = {}
    for symbol in list(fold_dirs_per_symbol):
        raw_df = data_loader.fetch_historical_data(
            symbol=symbol,
            interval=training_config["timeframe"],
            start_date=training_config["start_date"],
            end_date=training_config.get("end_date"),
        )
        if raw_df.empty:
            logger.warning(f"No data for {symbol}; dropping from study")
            fold_dirs_per_symbol.pop(symbol)
            continue
        full_df = build_features(
            raw_df=raw_df, symbol=symbol, feature_engineer=feature_engineer,
            sentiment_analyzer=sentiment_analyzer, horizon=horizon,
        )
        if target_col not in full_df.columns:
            logger.error(f"{target_col} missing for {symbol}; training config mismatch")
            fold_dirs_per_symbol.pop(symbol)
            continue
        symbol_to_frames[symbol] = full_df.dropna(subset=[target_col])
        symbol_to_dividends[symbol] = (
            None if args.no_dividends else data_loader.fetch_dividends(
                symbol=symbol,
                start_date=training_config["start_date"],
                end_date=training_config.get("end_date"),
            )
        )

    if not symbol_to_frames:
        raise ValueError("No usable symbol frames; nothing to evaluate.")

    init_mlflow(args.experiment)
    per_member_summary: Dict[str, Any] = {}
    member_seed_returns: Dict[str, Dict[int, pd.Series]] = {}

    with mlflow.start_run(run_name=f"rl_seed_eval_{training_run_dir.name}"):
        log_params_safe({
            "training_run": str(training_run_dir),
            "members": ",".join(members),
            "seeds": ",".join(map(str, seeds)),
            "symbols": ",".join(symbol_to_frames),
            "rl_timesteps": args.rl_timesteps,
            "horizon": horizon,
        })

        for member in members:
            logger.info(f"=== {member}: {len(seeds)} seeds × {len(symbol_to_frames)} symbols ===")
            with mlflow.start_run(run_name=member, nested=True):
                seed_returns: Dict[int, pd.Series] = {}
                for seed in seeds:
                    logger.info(f"[{member}] seed={seed}")
                    combined = eval_member_seed(
                        member=member, seed=seed,
                        symbol_to_folds=fold_dirs_per_symbol,
                        symbol_to_frames=symbol_to_frames,
                        target_col=target_col, horizon=horizon,
                        feature_engineer=feature_engineer, args=args,
                        sentiment_analyzer=sentiment_analyzer,
                        use_sentiment=use_sentiment,
                        symbol_to_dividends=symbol_to_dividends,
                    )
                    seed_returns[seed] = combined
                    sr = periodic_sharpe(combined) if len(combined) else float("nan")
                    log_metrics_safe({f"{member}_seed_sharpe": sr}, step=seed)

                summary = summarize_member(member, seed_returns)
                per_member_summary[member] = summary
                member_seed_returns[member] = seed_returns
                # Aggregated mean±std is the PRIMARY metric of record.
                log_metrics_safe({
                    f"{member}_seed_sharpe_mean": summary["mean"],
                    f"{member}_seed_sharpe_std": summary["std"],
                    f"{member}_seed_dsr": summary["dsr"],
                    f"{member}_seed_psr": summary["psr"],
                })

    out_dir = Path(args.output_dir) if args.output_dir else training_run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-member canonical research-claim packet. RL member robustness is a
    # claim about seed stability, not a single P&L curve, so each member is
    # represented by a real single-seed return series (the seed whose Sharpe is
    # closest to the seed-mean -- never a variance-suppressed seed average) and
    # the cross-seed DSR over n_seeds trials rides along in the metrics.
    for member, summary in list(per_member_summary.items()):
        mean = summary.get("mean")
        seed_sharpe = summary.get("seed_sharpe") or {}
        finite_seed_sharpe = {
            s: v for s, v in seed_sharpe.items() if v is not None and np.isfinite(v)
        }
        if mean is None or not finite_seed_sharpe:
            continue  # no valid seed -> nothing to claim
        rep_seed = min(finite_seed_sharpe, key=lambda s: abs(finite_seed_sharpe[s] - mean))
        rep_returns = member_seed_returns[member][int(rep_seed)]
        member_dir = out_dir / member
        member_dir.mkdir(parents=True, exist_ok=True)
        rep_returns.to_csv(member_dir / "representative_returns.csv")
        packet = emit_research_claim_packet(
            member_dir,
            strategy="rl_member_seed_robustness",
            config={
                "member": member,
                "representative_seed": int(rep_seed),
                "seeds": seeds,
                "seed_sharpe_mean": mean,
                "seed_sharpe_std": summary.get("std"),
                "n_seeds": summary.get("n_seeds"),
                "n_valid_seeds": summary.get("n_valid"),
                "rl_timesteps": args.rl_timesteps,
                "horizon": horizon,
                "representation": (
                    "single representative seed closest to the seed-mean Sharpe; "
                    "cross-seed robustness is carried in metrics.dsr over n_seeds trials"
                ),
            },
            data={
                "symbols": list(symbol_to_frames),
                "timeframe": training_config["timeframe"],
                "start_date": training_config["start_date"],
                "end_date": training_config.get("end_date"),
                "source": "Twelvedata",
                "data_convention": (
                    "split_adjusted_price_return_no_dividends"
                    if args.no_dividends
                    else "split_adjusted_price_return_dividends_as_cash"
                ),
                "training_run": str(training_run_dir),
                "universe_policy": "training_run_symbol_list",
            },
            returns=rep_returns,
            summary={"dsr": summary.get("dsr"), "n_trials": summary.get("n_seeds")},
            artifacts={
                "representative_returns": "representative_returns.csv",
                "claim_packet": "claim_packet.json",
            },
            code_commit=current_git_commit(Path(__file__).resolve().parents[1]),
        )
        per_member_summary[member] = {**summary, **summary_claim_fields(packet)}
        validate_claim_packet_dir(member_dir)

    out_path = out_dir / "rl_seed_eval.json"
    with open(out_path, "w") as f:
        json.dump({
            "training_run": str(training_run_dir),
            "seeds": seeds,
            "symbols": list(symbol_to_frames),
            "rl_timesteps": args.rl_timesteps,
            "members": per_member_summary,
        }, f, indent=2)

    logger.info(f"Wrote {out_path}")
    for member, s in per_member_summary.items():
        logger.info(
            f"{member}: mean={s['mean']} std={s['std']} "
            f"dsr={s['dsr']} (n_valid={s['n_valid']}/{s['n_seeds']})"
        )


if __name__ == "__main__":
    main()
