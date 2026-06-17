# Statistical Arbitrage

This path is separate from the directional ensemble. It scans a formation
sample for cointegrated pairs, converts stationary spread dislocations into
hedge-ratio target weights, combines pair books into a capped portfolio, and
accounts for next-open fills, spread, commission, impact, and borrow.

It is still a research harness, not a production execution stack. The important
upgrade is structural: the system can now test market-neutral hypotheses instead
of only ranking single-symbol directional forecasts.

## Design Invariants

- Pair discovery is train-only. `scan_cointegrated_pairs` should receive a
  formation window, never the full evaluation period.
- Multiple-testing risk is explicit. Cointegration p-values pass a
  Benjamini-Hochberg FDR gate before economic filters.
- Candidate pairs must clear both Engle-Granger cointegration and residual ADF
  stationarity, plus half-life, beta-drift, and return-correlation filters.
- Signals are close-time targets. `backtest_target_weights` shifts targets by
  one bar before returns, so the signal bar earns no same-bar PnL.
- The portfolio layer caps both gross exposure and per-symbol absolute weight.
- Rolling WFO folds close positions before fold boundaries. The current code
  rejects carry rules until they are explicitly designed and tested.

## CLI

```bash
# Rolling walk-forward evaluation. Prefer this for any reported result.
python -m scripts.stat_arb_wfo --symbols AAPL,MSFT,GOOG,AMZN,META,NVDA \
    --start_date 2020-01-01 --formation_bars 504 --test_bars 63 --max_pairs 5

# Single formation/test smoke test.
python -m scripts.stat_arb --symbols AAPL,MSFT,GOOG,AMZN,META,NVDA \
    --start_date 2020-01-01 --formation_bars 504 --max_pairs 5
```

Walk-forward outputs go to `results/stat_arb_wfo_<ts>/`:

- `pairs.json` - selected fixed relationships by fold.
- `folds.json` - formation/test dates, raw candidates, FDR cutoff, rejection
  counts, selected pairs, pair turnover, test-window diagnostics, and fold
  metrics.
- `target_weights.csv` - close-time target weights emitted only on test
  windows; fold tails are forced flat because carry rules are not implemented.
- `returns.csv`, `equity.csv`, `costs.csv` - next-open costed accounting.
- `pair_trial_sharpes.csv` - selected pair-book Sharpes used for the
  `pair_set_dsr` overfit diagnostic.
- `summary.json` - return, Sharpe, PSR, drawdown, exposure, turnover, cost,
  search-trial counts, pair turnover, `pair_set_dsr`, the config hash, and the
  claim tier.
- `claim_packet.json` - canonical research-trial packet with data convention,
  config hash, code commit, artifact manifest, gross/net/cost metrics, DSR, and
  the strongest supported claim tier.

The single-split command writes the same core portfolio artifacts under
`results/stat_arb_<ts>/`, but it is only a smoke test. Do not treat one
formation window as a serious arbitrage result.

## Known Gaps

- Cointegration is pairwise only. There is no sparse basket/Johansen layer yet.
- The hedge ratio is static over the evaluation period. Rolling/Kalman hedge
  updates need their own train-only state and drift controls before use.
- The backtester uses open-to-open portfolio returns. It is appropriate for
  daily research, but not a broker-grade intraday simulator.
- Borrow availability, locate constraints, hard-to-borrow fees, and exchange
  halts are not modeled.
- Factor, beta, sector, and crowding neutrality are not modeled yet; the pair
  hedge ratio is not sufficient evidence of market neutrality.
