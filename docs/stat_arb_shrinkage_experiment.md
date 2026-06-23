# Stat-Arb Experiment: FDR-Hard Gate vs Portfolio-Level Shrinkage

Status: **implemented, opt-in, first run negative after deflation.** The default
construction path remains the hard FDR gate (`fdr_hard`). The
`shrunk_candidates` arm is a falsifiable claim-packet experiment, not a default
strategy change.

## Motivation

Da, Nagel & Xiu (2024), *The Statistical Limit of Arbitrage*, show that the
optimal *feasible* arbitrage portfolio does **not** hard-threshold signals on
discovery significance. When alphas are weak and rare — empirically the regime
of the equity cross-section (their stocks: cross-sectional R² ≈ 8%, only 7.58%
with |t| > 2) — a Benjamini–Hochberg / FDR gate is "too conservative, with too
few or no discoveries," and the optimal portfolio instead assigns *continuous*
weights proportional to signal strength while *cross-sectionally smoothing*
(shrinking) the weak signals toward zero rather than dropping them.

The default stat-arb path uses BH-FDR on cointegration p-values as the final
portfolio on/off switch in `scan_cointegrated_pairs_with_report`. That is the
correct tool for *detecting* whether a pair is cointegrated, but Da–Nagel–Xiu's
result is a caution that detection significance is the wrong final switch for
*portfolio construction*: it sits on the conservative side of the bias–variance
trade-off for feasible net Sharpe.

This connects to three existing audit findings:
- **§10 (breadth ceiling, `IR ≈ IC·√BR`)** — the binding constraint is signal
  density; throwing away weak-but-real pairs is the opposite of what a
  breadth-limited book wants.
- **D-3 (the simplex meta-learner cannot say "no")** — same theme on the
  directional path: hard allocation vs shrinkage.
- **S-1 (min-of-orderings poisons the FDR input)** — the FDR gate is also
  anti-conservative in a different direction; both motivate demoting it from a
  gate to a diagnostic.

## Research question

> Does replacing the hard BH/FDR pair-inclusion gate with portfolio-level
> shrinkage improve feasible **net** OOS performance after costs, while
> preserving the selection ledger and claim-packet discipline?

## What `fdr_hard` does today (baseline)

`scan_cointegrated_pairs_with_report` in `src/arbitrage/pairs.py`:
1. Builds raw candidates per symbol pair (`_candidate_from_pair`).
2. Applies `benjamini_hochberg_mask` over cointegration p-values; BH failure is
   rejection reason `"fdr"` in `_rejection_reason`.
3. Applies economic filters (coint p, residual ADF p, return-corr, half-life,
   beta-drift).
4. Sorts and truncates to `max_pairs`.
5. Returns `selected_candidates`; each enters the book at full size, scaled only
   by `gross_per_pair/gross` in `generate_pair_positions` and then by the
   portfolio caps in `combine_pair_positions`.

So inclusion is binary and post-inclusion weighting ignores relative evidence
strength.

## Design: two construction modes

`PairSelectionConfig.construction_mode` selects the construction path:

- **`fdr_hard`** — current behavior, unchanged. **Hard invariant:** the portfolio
  artifacts (`returns`, `equity`, `target_weights`, `costs`, `summary`) are
  numerically identical to the pre-change path (parity test, in the spirit of
  audit R-0); every selected candidate carries `evidence_weight = 1.0`, so the
  combined target is bit-identical. The selection ledger (`folds.json`) gains a
  constant `evidence_weight = 1.0` field — a backward-compatible superset, not a
  numeric change.
- **`shrunk_candidates`** — the BH-FDR gate and the residual-ADF gate (a
  wrong-null redundancy per audit S-2) are **demoted to recorded diagnostics**,
  not inclusion switches. Admission uses a deliberately wider net; each admitted
  candidate carries a continuous `evidence_weight ∈ (0, 1]`; the portfolio caps
  do the shrinking.

### What stays fixed across both modes (so the comparison isolates the variable)

- The **economic tradeability filters** — `min/max_half_life`, `max_beta_drift`,
  `min_abs_return_corr`, `min_abs_beta` — remain admission filters in both
  modes. These are tradeability constraints, not significance gates. Holding the
  half-life cap fixed also prevents the S-3 convergence-truncation confound from
  leaking into the result.
- Folds, symbols, `ExecutionConfig`, `max_gross`, `max_symbol_abs_weight`,
  formation/test windows.

### Admission and weighting in `shrunk_candidates` (pre-registered)

To avoid the evidence-weight map becoming an untracked tuned knob (audit S-5 /
V-2), exactly **one** primary specification is registered up front; any variant
is a counted additional trial.

- **Admission (wider net):** keep all raw candidates that pass the economic
  filters and a *loose* nominal cointegration screen (reuse `max_coint_pvalue`);
  drop the BH-FDR gate and the `max_pairs` hard truncation (or raise the cap
  well above the typical selected count). Record BH outcome and FDR cutoff in
  the ledger exactly as today.
- **Strength:** `sᵢ = |coint_statᵢ|` (the Engle–Granger statistic is the
  available t-type strength, computed on the formation window only — preserves
  the train-only invariant, `stat_arb.md` design invariant 1).
- **Shrinkage (the "local smoothing of weak signals" from Da–Nagel–Xiu):** one
  empirical-Bayes / James–Stein step across the fold's admitted set,
  `ŝᵢ = s̄ + c·(sᵢ − s̄)` with `c = max(0, 1 − (k−2)·σ̂²/Σⱼ(sⱼ − s̄)²)`, then
  `evidence_weightᵢ = max(0, ŝᵢ) / maxⱼ max(0, ŝⱼ)`. Strong pairs keep ~full
  size; weak pairs are pulled toward zero but not dropped.
- A `ridge` variant (shrink combined weights toward the gross budget, per
  Da–Nagel–Xiu's "ridge-on-weights ≈ plain cross-sectional regression alphas")
  is registered as a *secondary, counted* trial, not the primary.

## Insertion points (concrete, minimal)

- `PairSelectionConfig`: `construction_mode: Literal["fdr_hard","shrunk_candidates"] = "fdr_hard"`.
  The James–Stein shrinkage is parameter-free (intensity is data-determined), so
  no shrinkage hyperparameters are introduced. The new field auto-serializes into
  the claim packet via `asdict(selection_cfg)` and the config hash — satisfying
  the audit S-5 knob-registry requirement for free.
- `PairCandidate`: `evidence_weight: float = 1.0`. With the default,
  `_candidate_to_dict` and `folds.json` carry it into the selection ledger;
  `fdr_hard` is unchanged.
- `scan_cointegrated_pairs_with_report`: branch on mode for
  admission; compute and attach `evidence_weight` in `shrunk_candidates`.
- `run_stat_arb_walk_forward`: multiply each pair's `target` by
  `candidate.evidence_weight` **before** `combine_pair_positions`.
  The existing clip + `scale_to_max_gross` then perform the portfolio-level
  shrink. No change to `combine_pair_positions` or the accounting.
- CLI `scripts/stat_arb_wfo.py`: add `--construction_mode`.

## Honesty mechanism (already wired)

`pair_trial_sharpes` is appended **per candidate** and feeds
`pair_set_dsr = deflated_sharpe_ratio(portfolio.returns, …)`. A wider net
therefore raises the trial count `N`, which raises `E[max Sharpe]` and
**deflates the portfolio Sharpe more**. The
experiment's win condition is that net Sharpe improves *after* paying this larger
deflation — i.e., the kept weak pairs add more signal than search noise. This is
exactly the empirical test of Da–Nagel–Xiu's "weak signals still help" claim on
our data, and it requires no new deflation code.

## Metrics and decision rule (falsification, pre-registered)

Run both modes on identical folds via `stat_arb_wfo` and compare:

- **Primary:** net OOS portfolio Sharpe and `pair_set_dsr` (deflated).
- **Secondary:** PSR, turnover, cost drag (`avg_turnover`, `total_cost`),
  gross/net exposure, pair turnover, per-fold breakdown, regime-split
  (audit V-5).

**Decision:** `shrunk_candidates` is judged superior **iff** it improves net
Sharpe **and** the improvement survives DSR deflation against the larger trial
family **and** does not raise turnover/cost drag disproportionately. Otherwise
the result is a clean negative — equally publishable as a claim packet, and a
confirmation that the current gate is not leaving feasible Sharpe on the table.

## First megacap WFO run

Run command:

```bash
python -m scripts.stat_arb_wfo --symbols AAPL,MSFT,GOOG,AMZN,META,NVDA \
    --start_date 2020-01-01 --formation_bars 504 --test_bars 63 --max_pairs 10 \
    --construction_mode shrunk_candidates
```

Compared with `fdr_hard` on the same symbols/folds:

| Metric | `fdr_hard` | `shrunk_candidates` |
| --- | ---: | ---: |
| Net Sharpe | 0.0913916969 | 0.1386878612 |
| PSR | 0.5764115609 | 0.6144896003 |
| `pair_set_dsr` | 0.0003073155 | 5.4474869305e-14 |
| `pair_trial_count` | 4 | 18 |
| Avg selected pairs | 0.2222222222 | 1.0 |
| Avg gross | 0.0627773053 | 0.2804536128 |
| Avg turnover | 0.0055372720 | 0.0342192716 |
| Total cost | 0.0037476789 | 0.0234405307 |
| Max drawdown | 0.0320308284 | 0.1199824883 |

Judgment: this is a clean negative under the pre-registered rule. Raw net Sharpe
and PSR improved, but the wider candidate family raised trial count, turnover,
costs, gross exposure, and drawdown; `pair_set_dsr` fell by many orders of
magnitude. The default should stay `fdr_hard` unless a broader pre-registered
run overturns this result after deflation.

## Negative control

The regression test `test_shrunk_candidates_noise_control_expands_trials_and_deflates_dsr`
injects unrelated synthetic symbols beside a true cointegrated pair. With a
loosened nominal cointegration screen, `shrunk_candidates` admits the spurious
noise pair at subunit evidence weight while `fdr_hard` rejects it; the selected
trial count rises and `pair_set_dsr` falls. This is the core honesty invariant:
wider admission must be paid for in the DSR ledger.

## Threats to validity

- **Multiplicity:** handled by the auto-deflation above; additionally, only the
  one primary evidence-weight spec is pre-registered — variants are counted
  trials (audit V-2 / S-5).
- **Look-ahead:** `evidence_weight` must use formation-window statistics only.
- **S-3 interaction:** the half-life cap is held fixed across modes so
  convergence truncation does not masquerade as a selection effect.

## Out of scope

Default strategy change; the learned-factor / cost-aware-policy redesign
(Attention Factors, Epstein–Wang–Choi–Pelger 2025). That redesign is deferred
until this experiment reports whether the current pairs subsystem has enough
signal density to justify it.
