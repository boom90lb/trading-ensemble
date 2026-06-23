"""Rolling walk-forward statistical-arbitrage research CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from scripts.stat_arb_data import fetch_price_matrices as _fetch_matrices
from src.arbitrage.pairs import (
    PairSelectionConfig,
    PairSignalConfig,
)
from src.arbitrage.walk_forward import (
    StatArbWalkForwardConfig,
    fold_result_to_dict,
    run_stat_arb_walk_forward,
)
from src.config import ExecutionConfig, RESULTS_DIR
from src.logging_utils import configure_logging
from src.validation.trials import (
    current_git_commit,
    emit_research_claim_packet,
    summary_claim_fields,
    validate_claim_packet_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rolling walk-forward pairs stat-arb research.")
    p.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols.")
    p.add_argument("--start_date", type=str, default="2020-01-01")
    p.add_argument("--end_date", type=str, default=None)
    p.add_argument("--formation_bars", type=int, default=504)
    p.add_argument("--test_bars", type=int, default=63)
    p.add_argument("--step_bars", type=int, default=None)
    p.add_argument("--min_test_bars", type=int, default=20)
    p.add_argument("--max_pairs", type=int, default=10)
    p.add_argument("--fdr_alpha", type=float, default=0.10)
    p.add_argument("--entry_z", type=float, default=2.0)
    p.add_argument("--exit_z", type=float, default=0.5)
    p.add_argument("--stop_z", type=float, default=4.0)
    p.add_argument("--z_window", type=int, default=60)
    p.add_argument("--max_gross", type=float, default=1.0)
    p.add_argument("--max_symbol_abs_weight", type=float, default=0.35)
    p.add_argument("--commission_bps", type=float, default=1.0)
    p.add_argument("--spread_bps", type=float, default=1.0)
    p.add_argument("--slippage_coeff", type=float, default=10.0)
    p.add_argument("--borrow_bps_annual", type=float, default=50.0)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    close_matrix, open_matrix = _fetch_matrices(symbols, args.start_date, args.end_date)

    selection_cfg = PairSelectionConfig(
        min_obs=min(args.formation_bars, 252),
        max_pairs=args.max_pairs,
        fdr_alpha=args.fdr_alpha,
    )
    signal_cfg = PairSignalConfig(
        z_window=args.z_window,
        min_z_obs=max(2, min(args.z_window, args.z_window // 2)),
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
    )
    walk_cfg = StatArbWalkForwardConfig(
        formation_bars=args.formation_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        min_test_bars=args.min_test_bars,
        max_gross=args.max_gross,
        max_symbol_abs_weight=args.max_symbol_abs_weight,
    )
    execution = ExecutionConfig(
        spread_bps=args.spread_bps,
        slippage_coeff=args.slippage_coeff,
        commission_bps=args.commission_bps,
        borrow_rate_bps_annual=args.borrow_bps_annual,
    )
    result = run_stat_arb_walk_forward(
        close_matrix,
        open_matrix,
        selection_config=selection_cfg,
        signal_config=signal_cfg,
        walk_config=walk_cfg,
        execution=execution,
    )

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else RESULTS_DIR / f"stat_arb_wfo_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "pairs": "pairs.json",
        "folds": "folds.json",
        "returns": "returns.csv",
        "equity": "equity.csv",
        "target_weights": "target_weights.csv",
        "costs": "costs.csv",
        "pair_trial_sharpes": "pair_trial_sharpes.csv",
        "summary": "summary.json",
        "claim_packet": "claim_packet.json",
    }
    folds_payload = [fold_result_to_dict(fold) for fold in result.folds]
    pairs_payload = [
        {
            "fold": fold["fold"],
            "formation_start": fold["formation_start"],
            "formation_end": fold["formation_end"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "selected_pairs": fold["selected_pairs"],
        }
        for fold in folds_payload
    ]
    (out_dir / artifacts["pairs"]).write_text(json.dumps(pairs_payload, indent=2, allow_nan=True))
    (out_dir / artifacts["folds"]).write_text(json.dumps(folds_payload, indent=2, allow_nan=True))
    result.portfolio.returns.to_csv(out_dir / artifacts["returns"])
    result.portfolio.equity.to_csv(out_dir / artifacts["equity"])
    result.portfolio.target_weights.to_csv(out_dir / artifacts["target_weights"])
    result.portfolio.costs.to_csv(out_dir / artifacts["costs"])
    pd.Series(result.pair_trial_sharpes, name="daily_sharpe").to_csv(
        out_dir / artifacts["pair_trial_sharpes"],
        index_label="trial",
    )

    config_payload = {
        "selection": asdict(selection_cfg),
        "signal": asdict(signal_cfg),
        "walk": asdict(walk_cfg),
        "execution": asdict(execution),
    }
    data_payload = {
        "symbols": symbols,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "source": "Twelvedata",
        "bar_interval": "1d",
        "data_convention": "split_adjusted_open_close_price_return_no_dividends",
        "universe_policy": "explicit_symbol_list",
    }
    packet = emit_research_claim_packet(
        out_dir,
        filename=artifacts["claim_packet"],
        strategy="pairs_stat_arb_wfo",
        config=config_payload,
        data=data_payload,
        returns=result.portfolio.returns,
        costs=result.portfolio.costs,
        target_weights=result.portfolio.target_weights,
        summary=result.summary,
        artifacts=artifacts,
        code_commit=current_git_commit(RESULTS_DIR.parent),
    )

    summary_payload = {**result.summary, **summary_claim_fields(packet, packet_filename=artifacts["claim_packet"])}
    (out_dir / artifacts["summary"]).write_text(json.dumps(summary_payload, indent=2, allow_nan=True))
    validate_claim_packet_dir(out_dir)
    print(
        json.dumps(
            {"output_dir": str(out_dir), **summary_payload},
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
