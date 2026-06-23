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

## Residual (Avellaneda-Lee) mode

The second research path replaces pair spreads with cross-sectional factor
residuals, after Avellaneda & Lee (2010), "Statistical arbitrage in the US
equities market". It is a deliberately thin vertical slice: factor residuals +
OU s-scores + target mapping. Two factor constructions are implemented — PCA
eigenportfolios (frozen v1 default) and **sector-ETF regression**
(`--factor_mode etf`, A-L's ETF model) — plus volume "trading time" as an
**opt-in ablation** (`--volume_time`, off by default; A-L §6 eq. 20).
Parameter sweeps, graphical matching, sparse baskets, and Kalman hedges remain
deferred.

Pipeline (`src/arbitrage/factors.py`, `residual.py`, `residual_walk_forward.py`):

0. **Consensus calendar** (`consensus_trading_days`): the wide panel is built
   on the *union* of every symbol's dates, which inherits phantom non-trading
   days — a vendor sometimes carries a single stray bar on an exchange holiday
   (Memorial Day, Christmas), so the union gains a date on which ~every name is
   NaN. Under the strict full-history rule below, one such row would bench the
   entire cross-section for a full `corr_window`. Dates where fewer than 60% of
   the panel's peak name count are present are dropped before any estimator runs
   (`n_dropped_calendar_days` in the summary). Real sessions keep ~all names;
   phantoms keep ~1, so the cut is unambiguous.
1. **Eligibility** (causal, per day): a full trailing `corr_window` of finite
   returns, close >= $5, and a trailing 20-day median dollar-volume floor.
2. **Eigenportfolios**: at each weekly rebalance, standardized-return
   correlation PCA on the trailing 252 bars of the eligible cross-section;
   top-15 eigenvectors, sign-fixed to positive weight sums; A-L weights
   `Q[j,i] = v[j,i] / sigma_i`; the cross-section freezes until the next
   rebalance.
3. **Betas/residuals** (daily): one batched OLS per day of every active
   stock's trailing 60 returns on the shared factor-return window. With
   `--volume_time` (A-L §6, off by default), the OLS *dependent* returns are
   first scaled by `<dV>/dV_t` (typical trailing share volume over the bar's
   own volume, clipped to `[1/4, 4]`) — measuring reversion in "trading time".
   Eligibility, PCA, and the factor returns stay on raw returns; with the flag
   off, `Rw is R` and the path is bit-for-bit the frozen one.
4. **OU s-score** (daily): AR(1) on the cumulative residual; map to
   `(m, sigma_eq, half-life)`; s-score with the A-L drift adjustment
   (`m` shifted by `alpha/kappa`). `b` outside (0,1) or a degenerate variance
   is *no signal* and is counted — the invalid-OU rate is a headline
   diagnostic. Half-lives above 30 bars are filtered the same way.
5. **State machine** (frozen A-L bands): open long s < -1.25, close long
   s > -0.50, open short s > +1.25, close short s < +0.75; eligibility loss
   or an invalid fit forces flat; an exit may flip directly when the s-score
   has crossed the far entry band.
6. **Book**: each position is `position_unit` of stock hedged with
   `-beta_j` eigenportfolio units mapped back to constituents through `Q`,
   netted per symbol *before* caps, then gross-scaled (ratio-preserving)
   and symbol-clipped. Same close-time targets, next-open fills, costs, and
   fold-flattening as the pairs path.

**Factor modes.** Step 2 above is the default `--factor_mode pca`. Under
`--factor_mode etf` the PCA eigenportfolios are replaced by the returns of the
11 SPDR sector ETFs (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC):
the factor "portfolio" that replicates factor *j* is a fixed unit position in
ETF *j*, so the same `window_R @ Q.T` yields the ETF factor returns and
`build_book_row` hedges each stock's betas with the ETFs themselves (carried in
the panel as non-tradeable factor columns, priced for the hedge legs but never
a stock leg). Each stock is regressed on the *full* ETF panel — the multivariate
ETF spec; A-L's headline model uses each stock's single matched sector ETF,
which needs a point-in-time GICS map. The selector is fixed, so there is no
weekly re-estimation: `n_rebalances=1` and `avg_explained_variance` is NaN in
ETF runs.

Unlike frozen pair candidates, the estimators re-roll causally through test
bars; folds exist for reporting and forced flattening. Only hyperparameters
are fixed — and in v1 they are frozen at the config defaults.

### CLI

```bash
# Mechanics smoke on a small cached panel (plumbing only, not a result).
python -m scripts.stat_arb_residual_wfo --symbols AAPL,MSFT,GOOG,AMZN \
    --n_factors 2 --start_date 2020-01-01

# Broad liquid subset first: top ~100 names by trailing dollar volume.
python -m scripts.stat_arb_residual_wfo --universe data/universe/sp500-2026-06-11.txt \
    --top_liquid 100 --start_date 2020-01-01

# Full current-S&P run (survivorship-biased universe; see Limitations).
python -m scripts.stat_arb_residual_wfo --universe data/universe/sp500-2026-06-11.txt \
    --start_date 2020-01-01

# ETF-factor mode: regress on the 11 SPDR sector ETFs instead of PCA factors.
python -m scripts.stat_arb_residual_wfo --universe data/universe/sp500-2026-06-11.txt \
    --factor_mode etf --start_date 2020-01-01 --end_date 2026-06-10
```

Outputs go to `results/stat_arb_residual_wfo_<ts>/`: `folds.json` (per-fold
names traded, gross, turnover, cost share, invalid-OU rate, metrics),
`summary.json`, `config.json`, and the usual returns/equity/weights/costs CSVs.

### Trial ledger and `residual_set_dsr`

Every run appends one entry (config hash, OOS periodic Sharpe) to
`results/stat_arb_residual_trials.jsonl`. `residual_set_dsr` is the deflated
Sharpe of the run's OOS returns against the ledger's full search history, so:

- the first run reports NaN (a single trial has nothing to deflate against);
- tweaking any frozen parameter after seeing a summary is a **new trial** —
  rerun and let the ledger deflate it. Deleting the ledger does not undo the
  search; it just hides it.

`residual_set_dsr` without `--design_trials` counts only ledger rows (runs
actually executed), so it **undercounts** the real researcher degrees of freedom
— `n_factors`, the windows, the four s-score bands, `position_unit`,
`max_half_life`, rebalance frequency, and the band/sizing modes are all searched
choices. It is therefore **optimistic and not comparable** to the pairs path's
`pair_set_dsr`, which counts one trial per candidate pair and so deflates much
harder. `--design_trials N` (plan 006) deflates against `max(N, ledger rows)`:
pass a pre-registered floor on the grid you searched (configs tried *and*
discarded), recorded as `design_trials` in the summary and ledger. The complete
fix — a global MLflow-mined trial ledger (audit V-2 / R-3) — is deferred; this
explicit-N floor is the honest interim.

### v1 limitations (documented, not fixed)

- **Dividends.** Loader prices are split-adjusted only
  (`DataLoader.fetch_historical_data`, `adjust=splits`) and the arbitrage
  backtester has no dividend cash leg (`backtest_target_weights` earns pure
  open-to-open price returns), so residuals and PnL are distorted around
  ex-dates — shorts are flattered, longs penalized. Acceptable for a first
  screen; a blocker for claiming real edge. The dividend leg is the first
  follow-up if results warrant it.
- **Survivorship.** `data/universe/sp500-<date>.txt` is the *current*
  constituent list. `filter_universe_asof` only drops names with no pre-as-of
  data; it cannot recover delisted or removed names. Results are biased up;
  do not call them point-in-time.
- **Mid-week data loss.** A cross-section stock whose return goes missing
  between rebalances contributes zero to factor returns for those bars.
- **One-shot liquidity screen.** `--top_liquid` ranks once, at the end of the
  first formation window — causal, but not a rolling liquidity universe.

### v1 result (frozen config, 2020-2026)

First broad run of the frozen config (`config_hash 14df1b6865c3`), full current
S&P 500, 2020-01-01 to 2026-06-10, 21 folds, ~493 names/bar:

| run | OOS Sharpe (ann.) | `residual_set_dsr` | avg gross | turnover/day | invalid-OU |
|---|---|---|---|---|---|
| top-100 liquid | +0.49 | NaN (1st trial) | 0.93 | 0.34 | 5e-5 |
| full S&P 500 (PCA) | **-0.63** | 0.009 (vs 2 trials) | 0.97 | 0.30 | 6e-5 |
| full S&P + volume time | **-0.81** | 0.001 (vs 3 trials) | 0.97 | 0.30 | 2e-6 |
| full S&P, ETF factors | **-0.23** | 0.029 (vs 4 trials) | 0.97 | 0.26 | 1e-4 |

**The naive baseline has no edge, and volume time does not rescue it.** The
full run loses money after costs (Sharpe -0.63, 6/21 folds positive) and is
roughly flat *before* costs (gross PnL ~+2% over five years vs. ~9% costs), so
realistic spread/impact/commission turn a near-zero gross signal negative.
`residual_set_dsr` ~0.009 says it is statistically indistinguishable from luck.
The volume "trading time" ablation makes it slightly *worse* (-0.81) — which
matches A-L's own finding that trading time helps ETF-factor strategies
"unequivocally" but gives "no significant improvement" for PCA-factor
strategies (§6). This is the expected, honest outcome of a clean baseline:
A-L report the daily residual-reversal Sharpe decayed sharply after ~2002 as
the trade crowded. The deliverable is a credible harness that *falsifies* the
naive trade at scale and reproduces the paper's PCA/volume result, not a
profitable strategy. Mechanics are validated (broad book, full gross
deployment, near-zero invalid-OU, clean calendar); the PCA alpha is not there.

**ETF factors are the first lever that moves the needle.** Holding the frozen
config fixed and switching only `--factor_mode etf` (same 503-name S&P, same
window, same costs; `config_hash a2c5538e70b1`) lifts the annualized OOS Sharpe
from -0.63 to **-0.23**. More tellingly, the ETF residual signal is *positive
before costs* — gross annualized Sharpe **+0.54** (gross cumulative +6.1% vs
8.6% in costs) — where the PCA signal was flat. That is exactly A-L's claim
that ETF factors beat PCA factors, now visible as a real gross reversion edge
rather than noise. It is still **net-negative**: ~0.26/day turnover at 2bp
round-trip + impact + 50bp borrow eats the whole gross edge, and
`residual_set_dsr` ~0.029 (deflated against all four trials) keeps it
statistically indistinguishable from zero. So ETF factors refine the verdict
from "no signal" to "a real but cost-bound signal" — the harness still
falsifies the naive trade *net of costs* on this universe, but the alpha is no
longer absent, it is being eaten by frictions. That redirects the next levers
toward turnover/holding-period and cost reduction rather than the factor model.
Remaining step-5 layers (parameter sweeps with full ledger discipline, the
dividend leg, and turnover reduction to chase the now-visible gross ETF edge)
would each be new ledger trials.

## Known Gaps

- Cointegration is pairwise only. There is no sparse basket/Johansen layer yet.
- The pairs hedge ratio is static over the evaluation period. Rolling/Kalman
  hedge updates need their own train-only state and drift controls before use.
- The backtester uses open-to-open portfolio returns. It is appropriate for
  daily research, but not a broker-grade intraday simulator.
- Borrow availability, locate constraints, hard-to-borrow fees, and exchange
  halts are not modeled.
- Pair books have no factor/beta/sector neutrality beyond the hedge ratio;
  residual books are factor-hedged only up to the estimated betas, with no
  explicit dollar- or beta-neutrality constraint on the netted book.
