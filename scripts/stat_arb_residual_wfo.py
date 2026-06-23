"""Rolling walk-forward residual (Avellaneda-Lee) stat-arb research CLI.

Flag defaults ARE the v1 frozen config. Every completed run appends one entry
(config hash, OOS Sharpe) to a persistent JSONL trial ledger, and
``residual_set_dsr`` is the deflated Sharpe of *this* run's OOS returns against
the ledger's full search history — so the first run reports NaN (one trial has
nothing to deflate against) and re-running with tweaked knobs automatically
deflates against every previous attempt. Deleting the ledger and re-running is
self-deception, not a fresh start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.arbitrage.factors import ResidualStatArbConfig
from src.arbitrage.residual_walk_forward import residual_fold_to_dict, run_residual_stat_arb_walk_forward
from src.arbitrage.walk_forward import StatArbWalkForwardConfig
from src.config import ExecutionConfig, RESULTS_DIR
from src.data_loader import DataLoader
from src.logging_utils import configure_logging, get_symbol_logger
from src.validation.metrics import deflated_sharpe_ratio, deflated_sharpe_ratio_with_n
from src.validation.trials import (
    current_git_commit,
    emit_research_claim_packet,
    summary_claim_fields,
    validate_claim_packet_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_TRIAL_LEDGER = RESULTS_DIR / "stat_arb_residual_trials.jsonl"
STRATEGY_TAG = "residual_wfo_v1"
# The eleven SPDR Select Sector ETFs span the GICS sectors over the 2020+ sample
# (XLRE since 2015, XLC since 2018), so each is present for the whole backtest.
DEFAULT_SECTOR_ETFS = "XLK,XLF,XLV,XLY,XLP,XLE,XLI,XLB,XLU,XLRE,XLC"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rolling walk-forward residual (Avellaneda-Lee) stat-arb research.")
    p.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (alternative to --universe).")
    p.add_argument("--universe", type=str, default=None, help="Universe file: one symbol per line, '#' comments.")
    p.add_argument("--universe_asof", type=str, default=None, help="Drop symbols with no data at/before this date.")
    p.add_argument(
        "--membership",
        type=str,
        default=None,
        help="Parquet of point-in-time membership intervals [ticker,start,end] from "
        "scripts.build_sp500_universe. Gates eligibility so a name trades only while it was "
        "actually an index member; changing it is a new ledger trial (new config hash).",
    )
    p.add_argument(
        "--coverage",
        type=str,
        default=None,
        help="Coverage JSON from the universe build; recorded as provenance so the "
        "survivorship skip-list (missing delisted prices) travels with the deflated-Sharpe claim.",
    )
    p.add_argument(
        "--top_liquid",
        type=int,
        default=None,
        help="Keep only the N most liquid names by trailing median dollar volume at the first "
        "formation end (a causal pre-test screen).",
    )
    p.add_argument("--start_date", type=str, default="2020-01-01")
    p.add_argument("--end_date", type=str, default=None)
    # Signal config — defaults are the v1 frozen configuration.
    p.add_argument("--corr_window", type=int, default=252)
    p.add_argument("--regr_window", type=int, default=60)
    p.add_argument("--n_factors", type=int, default=15)
    p.add_argument("--rebalance_every", type=int, default=5)
    p.add_argument("--min_price", type=float, default=5.0)
    p.add_argument("--min_median_dollar_volume", type=float, default=1_000_000.0)
    p.add_argument("--dollar_volume_window", type=int, default=20)
    p.add_argument("--s_entry_long", type=float, default=-1.25)
    p.add_argument("--s_exit_long", type=float, default=-0.50)
    p.add_argument("--s_entry_short", type=float, default=1.25)
    p.add_argument("--s_exit_short", type=float, default=0.75)
    p.add_argument("--max_half_life", type=float, default=30.0)
    p.add_argument("--position_unit", type=float, default=0.02)
    p.add_argument("--sizing_mode", type=str, default="unit", choices=["unit", "strength"])
    p.add_argument(
        "--volume_time",
        action="store_true",
        help="A-L §6 trading-time ablation: weight residual returns by typical/actual volume. "
        "OFF by default (frozen v1); enabling it is a new ledger trial.",
    )
    p.add_argument("--volume_time_window", type=int, default=60)
    p.add_argument("--volume_time_clip", type=float, default=4.0)
    p.add_argument(
        "--factor_mode",
        type=str,
        default="pca",
        choices=("pca", "etf"),
        help="Factor construction: 'pca' eigenportfolios (frozen v1) or 'etf' sector-ETF regression. "
        "'etf' is a new ledger trial.",
    )
    p.add_argument(
        "--etf_symbols",
        type=str,
        default=DEFAULT_SECTOR_ETFS,
        help="Comma-separated sector-ETF factor symbols, used only when --factor_mode etf.",
    )
    # Fold geometry and portfolio caps.
    p.add_argument(
        "--formation_bars", type=int, default=312, help="Must be >= corr_window + regr_window (estimator warmup)."
    )
    p.add_argument("--test_bars", type=int, default=63)
    p.add_argument("--step_bars", type=int, default=None)
    p.add_argument("--min_test_bars", type=int, default=20)
    p.add_argument("--max_gross", type=float, default=1.0)
    p.add_argument("--max_symbol_abs_weight", type=float, default=0.35)
    p.add_argument("--no_trade_band", type=float, default=0.0,
                   help="Proportional no-trade band on book weights (Davis-Norman/Garleanu-Pedersen): hold a "
                        "name until its target moves > band, cutting turnover. 0 = off (frozen-v1). A new ledger trial.")
    p.add_argument("--band_mode", type=str, default="fixed", choices=["fixed", "cost_aware"],
                   help="'fixed' uses the scalar --no_trade_band (frozen-v1); 'cost_aware' sizes a per-name band "
                        "from each name's OU half-life and round-trip cost (gamma=1.0, not fitted). A new ledger trial.")
    # Execution costs.
    p.add_argument("--commission_bps", type=float, default=1.0)
    p.add_argument("--spread_bps", type=float, default=1.0)
    p.add_argument("--slippage_coeff", type=float, default=10.0)
    p.add_argument("--adv_impact_coeff", type=float, default=0.0)
    p.add_argument("--adv_impact_model", type=str, default="linear", choices=["linear", "sqrt"])
    p.add_argument("--adv_floor_dollars", type=float, default=100000.0)
    p.add_argument("--borrow_bps_annual", type=float, default=50.0)
    p.add_argument(
        "--initial_capital",
        type=float,
        default=1.0,
        help="Portfolio notional used for equity and ADV participation. Default 1.0 preserves residual returns.",
    )
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--trial_ledger", type=str, default=str(DEFAULT_TRIAL_LEDGER))
    p.add_argument("--design_trials", type=int, default=0,
                   help="Pre-registered floor on researcher degrees of freedom: count the design grid you "
                        "searched (configs tried AND discarded), not just runs logged. >0 deflates "
                        "residual_set_dsr against max(N, ledger rows). 0 = ledger-only (optimistic).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if bool(args.symbols) == bool(args.universe):
        raise SystemExit("Pass exactly one of --symbols or --universe.")
    if args.universe:
        # Local import: scripts.training pulls the full ML stack (jax/torch)
        # at module level, which this numpy/pandas-only CLI must not pay for
        # unless a universe file is actually used.
        from scripts.training import load_universe

        return [s.upper() for s in load_universe(args.universe)]
    return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]


def _fetch_frames(symbols: list[str], start_date: str, end_date: str | None) -> dict[str, pd.DataFrame]:
    loader = DataLoader()
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        log = get_symbol_logger(logger, symbol)
        df = loader.fetch_historical_data(symbol, "1d", start_date, end_date)
        if df.empty:
            log.warning("no bars returned; skipping")
            continue
        missing = {"open", "close", "volume"} - set(df.columns)
        if missing:
            log.warning("missing required columns %s; skipping", sorted(missing))
            continue
        # Vendors occasionally return overlapping/duplicate bars; a duplicate
        # DatetimeIndex breaks wide-panel construction. Keep the last bar per day.
        if not df.index.is_unique:
            log.warning("duplicate bars dropped (%d)", int(df.index.duplicated().sum()))
            df = df[~df.index.duplicated(keep="last")]
        frames[symbol] = df.sort_index()
    return frames


def _panel_matrices(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes = pd.DataFrame({s: df["close"].astype(float) for s, df in frames.items()}).sort_index()
    closes = closes.dropna(how="all")
    opens = pd.DataFrame({s: df["open"].astype(float) for s, df in frames.items()}).reindex(closes.index)
    volumes = pd.DataFrame({s: df["volume"].astype(float) for s, df in frames.items()}).reindex(closes.index)
    return closes, opens, volumes


def _with_etf_factors(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    volumes: pd.DataFrame,
    etf_symbols: list[str],
    start_date: str,
    end_date: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append the sector-ETF factor columns to an already-screened stock panel.

    ETFs are fetched *after* the stock universe screens so they never compete in
    the liquidity ranking and are guaranteed present as factors; they are then
    aligned to the stock trading calendar. The signal panel marks them
    ineligible as stock legs (factors/hedges only).
    """
    collisions = sorted(set(etf_symbols) & {str(c) for c in closes.columns})
    if collisions:
        raise RuntimeError(f"ETF factor symbols collide with the stock universe: {collisions}")
    etf_frames = _fetch_frames(etf_symbols, start_date, end_date)
    missing = [s for s in etf_symbols if s not in etf_frames]
    if missing:
        raise RuntimeError(f"Could not fetch ETF factor bars for: {missing}")
    e_close, e_open, e_vol = _panel_matrices(etf_frames)
    closes = pd.concat([closes, e_close.reindex(closes.index)], axis=1)
    opens = pd.concat([opens, e_open.reindex(opens.index)], axis=1)
    volumes = pd.concat([volumes, e_vol.reindex(volumes.index)], axis=1)
    return closes, opens, volumes


def _top_liquid_symbols(
    closes: pd.DataFrame, volumes: pd.DataFrame, n: int, formation_bars: int, window: int
) -> list[str]:
    """Rank by trailing median dollar volume at the end of the FIRST formation window.

    The ranking bar precedes every test bar, so this screen is causal for the
    whole walk-forward; it is still a one-shot liquidity snapshot, not a
    point-in-time liquidity universe.
    """
    if len(closes) < formation_bars:
        raise RuntimeError(f"Need at least formation_bars={formation_bars} rows for the liquidity screen")
    asof_pos = formation_bars - 1
    dollar = (closes * volumes).iloc[max(0, asof_pos - window + 1) : asof_pos + 1]
    ranked = dollar.median(axis=0, skipna=True).dropna().sort_values(ascending=False)
    return sorted(str(s) for s in ranked.head(n).index)


def _config_payload(
    signal_cfg: ResidualStatArbConfig,
    walk_cfg: StatArbWalkForwardConfig,
    execution: ExecutionConfig,
    args: argparse.Namespace,
    symbols: list[str],
    universe_meta: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "strategy": STRATEGY_TAG,
        "signal": asdict(signal_cfg),
        "walk": asdict(walk_cfg),
        "execution": asdict(execution),
        "initial_capital": float(args.initial_capital),
        "design_trials": int(args.design_trials),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "symbols": sorted(symbols),
        "universe": universe_meta or {},
    }


def _config_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _file_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _membership_provenance(args: argparse.Namespace, closes: pd.DataFrame) -> tuple[object, dict[str, object], list[str]]:
    """Build the day x symbol membership mask and a compact provenance block.

    Returns ``(mask_or_None, universe_meta, coverage_skipped)``. ``universe_meta``
    is hashed into the config (so a different universe is a different trial);
    the verbose skip-list goes only to the human-readable summary.
    """
    universe_meta: dict[str, object] = {
        "universe_file": args.universe,
        "universe_asof": args.universe_asof,
        "membership_file": args.membership,
        "coverage_file": args.coverage,
    }
    mask = None
    if args.membership:
        from src.universe_sp500 import build_membership_mask

        membership = pd.read_parquet(args.membership)
        mask = build_membership_mask(membership, closes.index, list(closes.columns))
        universe_meta["membership_sha"] = _file_sha(args.membership)
        universe_meta["n_membership_intervals"] = int(len(membership))
    coverage_skipped: list[str] = []
    if args.coverage:
        cov = json.loads(Path(args.coverage).read_text())
        universe_meta["coverage"] = {
            k: cov.get(k) for k in ("asof", "n_ever_members", "n_resolved", "n_skipped", "coverage_fraction")
        }
        coverage_skipped = list(cov.get("skipped", []))
    return mask, universe_meta, coverage_skipped


def _append_trial(ledger_path: Path, entry: dict[str, object]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True, allow_nan=True) + "\n")


def _load_trial_sharpes(ledger_path: Path) -> list[float]:
    """All finite periodic OOS Sharpes recorded in the ledger.

    A corrupt line is logged loudly: silently dropping trials understates the
    search history and inflates the deflated Sharpe.
    """
    if not ledger_path.exists():
        return []
    sharpes: list[float] = []
    for lineno, line in enumerate(ledger_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Trial ledger {ledger_path} line {lineno} is corrupt; DSR will understate trials")
            continue
        sharpe = entry.get("oos_periodic_sharpe")
        if isinstance(sharpe, (int, float)) and np.isfinite(sharpe):
            sharpes.append(float(sharpe))
    return sharpes


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    symbols = _resolve_symbols(args)
    frames = _fetch_frames(symbols, args.start_date, args.end_date)
    if args.universe_asof:
        from scripts.training import filter_universe_asof  # heavy transitive imports; see _resolve_symbols

        frames = filter_universe_asof(frames, args.universe_asof)
    min_stocks = 2 if args.factor_mode == "etf" else args.n_factors + 2
    if len(frames) < min_stocks:
        raise RuntimeError(f"Only {len(frames)} symbols with usable bars; need at least {min_stocks}")

    closes, opens, volumes = _panel_matrices(frames)
    if args.top_liquid is not None and args.top_liquid < len(closes.columns):
        keep = _top_liquid_symbols(closes, volumes, args.top_liquid, args.formation_bars, args.dollar_volume_window)
        logger.info(f"Liquidity screen kept {len(keep)}/{len(closes.columns)} symbols")
        closes, opens, volumes = closes[keep], opens[keep], volumes[keep]

    etf_syms: list[str] = []
    if args.factor_mode == "etf":
        etf_syms = list(dict.fromkeys(s.strip().upper() for s in args.etf_symbols.split(",") if s.strip()))
        if not etf_syms:
            raise SystemExit("--factor_mode etf requires a non-empty --etf_symbols list")
        closes, opens, volumes = _with_etf_factors(closes, opens, volumes, etf_syms, args.start_date, args.end_date)
        logger.info(f"ETF-factor mode: {len(etf_syms)} sector ETFs over {len(closes.columns) - len(etf_syms)} stocks")

    membership_mask, universe_meta, coverage_skipped = _membership_provenance(args, closes)
    if membership_mask is not None:
        logger.info("Point-in-time membership gate active over %d panel columns", closes.shape[1])

    signal_cfg = ResidualStatArbConfig(
        corr_window=args.corr_window,
        regr_window=args.regr_window,
        n_factors=args.n_factors,
        rebalance_every=args.rebalance_every,
        min_price=args.min_price,
        min_median_dollar_volume=args.min_median_dollar_volume,
        dollar_volume_window=args.dollar_volume_window,
        s_entry_long=args.s_entry_long,
        s_exit_long=args.s_exit_long,
        s_entry_short=args.s_entry_short,
        s_exit_short=args.s_exit_short,
        max_half_life_bars=args.max_half_life,
        position_unit=args.position_unit,
        volume_time=args.volume_time,
        volume_time_window=args.volume_time_window,
        volume_time_clip=args.volume_time_clip,
        factor_mode=args.factor_mode,
        etf_symbols=tuple(etf_syms),
        sizing_mode=args.sizing_mode,
    )
    walk_cfg = StatArbWalkForwardConfig(
        formation_bars=args.formation_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        min_test_bars=args.min_test_bars,
        max_gross=args.max_gross,
        max_symbol_abs_weight=args.max_symbol_abs_weight,
        no_trade_band=args.no_trade_band,
        band_mode=args.band_mode,
    )
    execution = ExecutionConfig(
        spread_bps=args.spread_bps,
        slippage_coeff=args.slippage_coeff,
        commission_bps=args.commission_bps,
        borrow_rate_bps_annual=args.borrow_bps_annual,
        adv_impact_coeff=args.adv_impact_coeff,
        adv_impact_model=args.adv_impact_model,
        adv_floor_dollars=args.adv_floor_dollars,
    )

    result = run_residual_stat_arb_walk_forward(
        closes,
        opens,
        volumes,
        signal_config=signal_cfg,
        walk_config=walk_cfg,
        execution=execution,
        membership_mask=membership_mask,
        initial_capital=args.initial_capital,
    )

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else RESULTS_DIR / f"stat_arb_residual_wfo_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "folds": "folds.json",
        "returns": "returns.csv",
        "equity": "equity.csv",
        "target_weights": "target_weights.csv",
        "costs": "costs.csv",
        "config": "config.json",
        "summary": "summary.json",
        "claim_packet": "claim_packet.json",
    }

    payload = _config_payload(signal_cfg, walk_cfg, execution, args, list(closes.columns), universe_meta)
    config_hash = _config_hash(payload)
    ledger_path = Path(args.trial_ledger)
    _append_trial(
        ledger_path,
        {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "strategy": STRATEGY_TAG,
            "config_hash": config_hash,
            "oos_periodic_sharpe": result.summary["oos_periodic_sharpe"],
            "oos_annualized_sharpe": result.summary["sharpe"],
            "n_obs": int(len(result.portfolio.returns)),
            "n_folds": int(result.summary["n_folds"]),
            "n_symbols": int(result.summary["n_symbols"]),
            "design_trials": int(args.design_trials),
            "output_dir": str(out_dir),
            "config": payload,
        },
    )
    trial_sharpes = _load_trial_sharpes(ledger_path)
    if args.design_trials > 0:
        residual_set_dsr = deflated_sharpe_ratio_with_n(
            result.portfolio.returns, np.asarray(trial_sharpes), args.design_trials
        )
    else:
        residual_set_dsr = deflated_sharpe_ratio(result.portfolio.returns, np.asarray(trial_sharpes))

    summary: dict[str, object] = {
        "output_dir": str(out_dir),
        "config_hash": config_hash,
        "trial_ledger": str(ledger_path),
        "ledger_trials": len(trial_sharpes),
        "design_trials": int(args.design_trials),
        "initial_capital": float(args.initial_capital),
        **result.summary,
        "residual_set_dsr": float(residual_set_dsr),
        "universe": universe_meta,
        "coverage_skipped": coverage_skipped,
    }
    (out_dir / artifacts["folds"]).write_text(
        json.dumps([residual_fold_to_dict(f) for f in result.folds], indent=2, allow_nan=True)
    )
    (out_dir / artifacts["config"]).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    result.portfolio.returns.to_csv(out_dir / artifacts["returns"])
    result.portfolio.equity.to_csv(out_dir / artifacts["equity"])
    result.portfolio.target_weights.to_csv(out_dir / artifacts["target_weights"])
    result.portfolio.costs.to_csv(out_dir / artifacts["costs"])

    data_payload = {
        "symbols": list(closes.columns),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "source": "Twelvedata",
        "bar_interval": "1d",
        "data_convention": "split_adjusted_open_close_price_return_no_dividends",
        "universe_policy": "universe_file" if args.universe else "explicit_symbol_list",
        "membership": universe_meta.get("membership_file"),
        "coverage": universe_meta.get("coverage_file"),
        "universe": universe_meta,
        "coverage_skipped_count": len(coverage_skipped),
    }
    packet = emit_research_claim_packet(
        out_dir,
        filename=artifacts["claim_packet"],
        strategy="residual_stat_arb_wfo",
        config=payload,
        data=data_payload,
        returns=result.portfolio.returns,
        costs=result.portfolio.costs,
        target_weights=result.portfolio.target_weights,
        summary=summary,
        artifacts=artifacts,
        code_commit=current_git_commit(RESULTS_DIR.parent),
    )
    summary_payload = {**summary, **summary_claim_fields(packet, packet_filename=artifacts["claim_packet"])}
    (out_dir / artifacts["summary"]).write_text(json.dumps(summary_payload, indent=2, allow_nan=True))
    validate_claim_packet_dir(out_dir)
    print(json.dumps(summary_payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
