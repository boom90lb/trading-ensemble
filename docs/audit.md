# trading-ensemble: An Audit of Model Biases and System Design Assumptions

**A refactor-oriented structural review**

---

## Abstract

This audit dissects the `trading-ensemble` repository along two axes: *model biases* (statistical pathologies baked into each ensemble member and the blending machinery) and *system design assumptions* (premises the harness takes as given about data, execution, and evaluation). The central thesis: the evaluation harness is unusually honest by retail-research standards — purged walk-forward CV, next-open costed fills, DSR/PBO plumbing, and a genuine leakage-test culture — but several members and one plumbing seam mean the harness is currently producing honest measurements of the wrong system. Four findings are Tier-0 (they change what every reported number means); the remainder are organized into a consolidated register mapping each finding to a phase in a sequenced refactor program (R0–R4) with pinned invariants. A closing section argues from the fundamental law of active management that the binding constraint on "competitive in the stock market" is not any single member's quality but breadth, and that the refactor program should be read as preparation for universe expansion rather than as an end in itself.

---

## §1 Scope and method

The audit covers the directional ensemble path (`src/models/`, `src/trading.py`, `scripts/training.py`, `scripts/backtest.py`, `scripts/sweep.py`), the statistical-arbitrage path (`src/arbitrage/`, `scripts/stat_arb*.py`), the data and feature layer, and the validation stack (`src/validation/`, `src/conformal/`). Findings carry stable IDs by theme — **T** (Tier-0 truth-value), **M** (member model bias), **D** (decision layer), **E** (execution), **S** (stat-arb), **I** (infrastructure/data), **V** (validation theory) — each with a code anchor, a mechanism, a consequence, and a refactor phase. §11 consolidates them into a single register; §12 sequences the refactors so that each phase's exit criterion is a pinned test, in the spirit of the repository's existing regression-test discipline (`tests/test_real_pipeline_regressions.py`). Severity is assigned by *epistemic* impact: a finding is Tier-0 if its presence means the current backtest reports cannot be interpreted as measurements of the intended system, regardless of whether the bias inflates or deflates performance.

A methodological note on the audit itself: every claim below was traced through the actual call graph rather than inferred from docstrings, because this codebase has already demonstrated (the "synthetic-vs-real lesson" in `docs/operations.md`) that its documentation describes intent faithfully while the plumbing occasionally diverges.

---

## §2 What the harness already gets right

Credit where due, because the refactor program should preserve these properties as invariants, not rediscover them. The purge/embargo implementation in `PurgedWalkForward` correctly removes training rows whose forward-label window overlaps the test slice (López de Prado 2018, §7.4) and the property tests in `tests/test_walk_forward.py` pin both controls. Execution is signal-on-close, fill-at-next-open with half-spread, linear impact, commission, and daily borrow on shorts — no same-bar fills anywhere, enforced by `tests/test_execution.py`. Dividends are handled as explicit cash against split-adjusted prices, which is the correct total-return treatment without back-adjustment look-ahead. The point-in-time sentiment join (B9), train-only clip bounds (audit-b), and per-fold scaler refits are real leakage closures with regression tests. The DSR is computed against an honestly enumerated trial grid rather than a hand-waved N (Bailey & López de Prado 2014), PBO uses genuine CSCV (Bailey et al. 2017), and the stat-arb path applies Benjamini–Hochberg FDR control to the pair scan with a per-fold selection ledger — a discipline most published pairs-trading studies lack. Finally, the README's framing ("the value is in the harness around them") is the correct epistemic posture.

The gap, then, is precise: a harness that faithfully measures *whatever it is fed* has been fed several components whose train-time and serve-time behaviors differ.

---

## §3 Tier-0 findings: what the current numbers mean

### T-1. Train/serve feature-scale parity break in the backtest path

**Anchor:** `scripts/backtest.py::main` → `run_symbol_wfo` → `TradingStrategy.run_segment`; contrast `scripts/training.py::train_symbol_wfo`.

Fold models are fit on `transform_features` output — min-max-scaled price columns, standardized technicals, log-transformed volume. The backtest, however, constructs `full_df` via `build_features` (whose docstring states the frame is "intentionally UNSCALED") and `run_segment` feeds raw rows directly to `model.predict`. No `fit_scalers`/`transform_features` call exists anywhere on the backtest path, and the fold artifacts do not persist the fold-fit `FeatureEngineer` state.

The consequence is member-specific. XGBoost's split thresholds were learned on features in $[0,1]$ and $\mathcal{N}(0,1)$ units; raw inputs (close ≈ 150, RSI ≈ 0–100) route nearly all rows through the same leaves, collapsing the member toward a constant. Prophet's extra-regressor coefficients were fit on scaled covariates and are applied to raw ones. ARIMA is unaffected (univariate on `y_train`, `use_exog=False`), and the RL members are consistent *by accident* — their environments are built from `train_raw` at fit and the raw frame at predict. The meta-learner weights were solved in scaled-OOF space and then applied to this raw-space mixture. `scripts/sweep.py::run_trial` calls the same `run_symbol_wfo`, so the sweep's "honest DSR" deflates the Sharpe of the skewed system; `scripts/prediction.py` shares the defect. Meanwhile, training-time `ensemble.evaluate()` uses the *scaled* test slice, so `fold_metrics.json` and the backtest report quietly measure two different systems.

This is the single highest-leverage fix in the repository. Until it lands, no directional-path number — Sharpe, PSR, PBO, DSR — admits the interpretation the README assigns it. Refactor: serialize the fold-fit transform (scalers + clip bounds) inside the fold artifact and apply it in `run_segment`; pin with a golden parity test asserting training-eval and backtest-path member outputs are identical on identical rows. *(Phase R0.)*

### T-2. Forecast-LSTM: label-in-window leakage plus inference zero-fill

**Anchor:** `src/models/lstm.py::fit`, `_create_sequences`, `predict`.

`fit` places the target as column 0 of `train_data` and `_create_sequences` includes that column *inside the input windows*. Since `target_h[t] = close[t+h]`, the window ending at row $i{+}L{-}1$ contains `close[(i+L+h)-1]` while the label is `close[i+L+h]` — the label is a one-bar continuation of a feature already present in the window. Training therefore reduces to copying a near-future feature. At inference the target column does not exist, so `predict` inserts a constant zero where the most informative training feature sat — total train/serve skew layered on top of the leak — and the `transfer_params` fallback silently re-initializes parameters on shape mismatch. The member is then (per `docs/operations.md`) silently dropped from the backtest blend on length grounds anyway, but it still contaminates training-time evaluation and the meta-learner's OOF intersection when configured. The member is broken at three independent levels; it should be excised or rebuilt from scratch on a returns target with the label strictly excluded from inputs. *(Phase R0 quarantine; R1 rebuild.)*

### T-3. Sentiment: 100-article truncation and the deprecated scorer on the production path

**Anchor:** `src/sentiment_analysis.py::SentimentAnalyzer.fetch_news`; `scripts/training.py::main`.

`fetch_news` requests `limit=100` over the *entire* backtest window, sorted `published_utc desc`, with no pagination (Polygon's `next_url` is ignored). For a multi-year window only the ~100 most recent articles per symbol are ever scored: early bars receive structurally zero sentiment and the signal is concentrated at the end of the sample — a recency-localized feature that interacts with expanding-window folds in the worst possible way. Separately, `scripts/training.py` instantiates the keyword `SentimentAnalyzer` (explicitly deprecated in its own docstring) rather than the FinBERT `TransformerSentimentAnalyzer` that Phase 3.1 shipped. Any `--use_sentiment` run is therefore measuring a truncated keyword artifact. Refactor: paginate per sub-window with a PIT article cache, wire the transformer scorer, or hard-disable the flag until done. *(Phase R0, conditional on sentiment use.)*

### T-4. Fold-replay alignment is integer-fragile

**Anchor:** `scripts/training.py` (`np.savez(... train_idx, test_idx)`); `scripts/backtest.py::_fold_date_range`.

The backtest reconstructs `usable` from a fresh fetch and indexes it with integer positions saved at training time. The cache-coverage scheme mitigates drift, but any vendor revision, cache deletion, or upstream row-count change silently shifts which *dates* each fold tests — there is no assertion that the replayed index matches the trained one. Given the project's existing taste for content-addressed gates, the fix is natural: persist the fold's test **date span** plus a hash of the `usable` index slice, and fail loud on mismatch. *(Phase R0.)*

---

## §4 Member-level model biases

### M-1. ARIMA: fold-constant forecasts and a misdated target

Two distinct problems. First, dating: ARIMA is fit on `y_train = target_h`, i.e., the close series *shifted forward by h* but indexed at $t$. Forecasting "steps ahead" from the end of that relabeled series and aligning positionally to test rows misdates every prediction by $h$ bars; the model is not producing $\hat{P}_{t+h} \mid \mathcal{F}_t$ in any coherent sense. Second, and more damaging in the live loop: the ensemble's `required_history` for a forecast-only configuration is 1, so `run_segment` feeds a single trailing row each bar, and `ARIMAModel.predict` calls `forecast(steps=1)` from the *frozen* train-end state. The fitted model never observes a single test-period bar. The result is one identical $\hat{y}$ for the entire fold; the member's "signal" degenerates to $\propto (\text{const}/P_t - 1)/\sigma_t$ — pure mean reversion toward an arbitrary anchor fixed at the fold boundary. In training-time `evaluate()` the same model instead emits a multi-step extrapolation whose error grows with fold depth, so the two reporting surfaces see different pathologies from the same member. Refactor: rolling state updates per bar (statsmodels `apply`/`extend`) on a *returns* series, or retire the member. *(Phase R1.)*

### M-2. Prophet: an unregularized ~200-regressor linear model wearing a trend

`ProphetModel.fit` adds **every** non-OHLCV, non-label column as an extra regressor. After `create_lagged_features` (all numeric columns × lags {1,2,5,10}) that is on the order of 150–200 regressors on ~10³ training rows, each entering Prophet's linear component with its default weak prior. Prophet here is not a seasonality model; it is a massively overparameterized linear regression with a piecewise-linear trend bolted on, whose out-of-fold behavior is dominated by trend extrapolation from the last changepoint (a known failure mode; Taylor & Letham 2018 position Prophet for business series with strong seasonal structure, not daily equity prices). Under T-1 its regressor inputs are additionally scale-skewed at serve time. Refactor: either a strict regressor diet (≤5 hand-chosen, stationary covariates) with tightened `prior_scale`, or removal — daily single-name equities offer Prophet's seasonal machinery essentially nothing. *(Phase R1.)*

### M-3. XGBoost: level targets meet trees that cannot extrapolate

The target is the future *price level*. Tree ensembles predict within the convex hull of training targets (Hastie, Tibshirani & Friedman 2009); on an uptrending name, test-period prices exceed the train-target range, so the member's forecast saturates near the training maximum, manufacturing a systematic short bias precisely when the underlying trends — the regime in which the 2020–2026 megacap universe spent most of its time. Level targets also make MAE/RMSE incomparable across folds and symbols. Refactor: target $= \log(P_{t+h}/P_t)$, features restricted to stationary transforms. This is the standard resolution and it simultaneously repairs the meta-learner's price-MSE side metrics. *(Phase R1.)*

### M-4. RL members: non-stationary observations

The trading environments are constructed from `train_raw` — *unscaled* frames — so policy observations include absolute prices and moving averages in dollars. A policy trained on AAPL at \$150–190 receives \$210 at test time as a literally out-of-distribution observation; nothing in the architecture confers price-level invariance. The `transform_features` "RL-specific normalization" branch never touches this path. Refactor: standardize observations inside the environment (log-returns, rolling z-scores, vol-normalized ranges), making the observation space stationary by construction. *(Phase R1.)*

### M-5. RL members: reward–execution objective mismatch

The environment reward is close-to-close next-bar return minus 10 bp proportional turnover. The execution model the policy is later judged under fills at next *open* with half-spread + commission + quadratic-in-participation impact + borrow on shorts. The policy is optimized against a different fill convention and a different (and short-side-blind) cost structure than the one that scores it. Refactor: mirror `ExecutionConfig` inside the env reward (open-referenced returns, borrow on negative positions), so the gradient points at the objective of record. *(Phase R2.)*

### M-6. GRPO: group ranking on a deterministic single path, plus a padding artifact

`TradingEnvironment` is deterministic given actions; the only stochasticity across the $G$ rollouts from a shared start state is the policy's own sampling. Ranking rollouts by realized cumulative reward on one historical path and applying a pairwise DPO-style loss therefore rewards *action sequences that happened to align with that path* — distilled path memorization, with no reference-policy KL anchor to resist log-std collapse (contrast Shao et al. 2024, where group members face genuinely stochastic completions). Additionally, `_rollout_one` zero-pads early-terminated rollouts and `_sequence_log_prob` sums log-probs over padded steps, so length differences leak into preference comparisons via the likelihood of dummy zero-actions on zero-observations. The PPO members share the single-path overfitting concern in milder form (the M10 seed study correctly measures the resulting variance). Refactor: mask pads in the sequence log-prob; train across pooled symbols and/or block-bootstrapped paths; add a KL term. Longer term, the honest position is that on one realized daily path per symbol, model-free RL has very little to learn that a supervised signal does not already capture. *(Phases R1–R2.)*

---

## §5 Ensemble and decision-layer assumptions

### D-1. Horizon incoherence: 5-day labels, daily decisions

Every forecast member is trained to predict the $h{=}5$-bar forward move, yet `run_segment` re-derives a fresh ternary decision *every bar* from the latest overlapping forecast. The system holds no position for the horizon its labels describe (only `prediction.py` does, and documents it); instead, autocorrelated overlapping forecasts generate sign chatter and turnover whose costs are real while the predictive content was specified at a different frequency. The purge correctly handles the *statistical* side of overlap; the *economic* design remains mismatched. Refactor options: (i) decide at horizon cadence with explicit holds, (ii) relabel at the decision frequency ($h{=}1$) and let the signal IR carry per-member horizon metadata, or (iii) Bayesian-update a persistent target from overlapping forecasts. Any of these is coherent; the current hybrid is not. *(Phases R1–R2.)*

### D-2. Volatility-unit mismatch in `forecast_to_position`

`position = clip(target\_vol \cdot \mathbb{E}[r_h]/\sigma_{daily})` divides an $h$-bar expected return by a *1-bar* realized vol. Under √-time scaling the resulting "forecast Sharpe" is inflated by ≈ $\sqrt{h} \approx 2.24$, pushing positions into the cap far more often than intended. Saturated positions destroy magnitude information (the blend becomes nearly sign-only), distort the `signal_threshold` semantics, and compress the meta-learner's regression target (`ideal_position` shares the mismatch, so the *fit* is internally consistent but operates in a clipped, low-information space). Refactor: $\sigma_h = \sigma_{daily}\sqrt{h}$, or work throughout in per-bar return units. *(Phase R1.)*

### D-3. The simplex meta-learner cannot say "no"

`_fit_meta_learner_weights` solves constrained least squares with $w \ge 0,\ \sum w = 1$ against the perfect-foresight target. The simplex constraint *forces full allocation across members even when every member is skill-less* — there is no cash vertex, no intercept, no shrinkage. The objective is cost-blind: a member achieving its MSE via high-turnover positions is priced identically to a calm one. And the policy-member OOF that feeds the fit consists of $k$ fresh RL refits whose seed-level variance the M10 study itself documents as dominant — the weights are partially noise-fit by construction. Refactor: relax to $\sum w \le 1$ (implicit cash), add a turnover penalty $\lambda \sum_t |\Delta(P w)_t|$ to the objective, and shrink toward equal weight (Jorion-style); report weight stability across folds as a first-class diagnostic. *(Phase R2.)*

### D-4. The order layer discards the ensemble's central output

`calculate_signal` collapses a continuous target position to {LONG, SHORT, FLAT}; `submit_signal_order` early-returns when the direction is unchanged, so *size is frozen at entry* no matter how conviction evolves; flips execute as full close-then-reopen (double costs); the entry threshold has no exit hysteresis, inviting chatter around 0.1. Meanwhile "confidence" is a second multiplicative shrinkage stack (|position| blended with sign-agreement, then scaled by conformal band width) applied on top of a position that was already vol-sized — the same uncertainty is charged twice or thrice with no coherent decision-theoretic interpretation. Refactor: delete the ternary pathway. The ensemble already emits a target weight; the decision layer should be a rebalancer that trades the *difference* between current and target weights, with a no-trade band sized from costs (Gârleanu & Pedersen 2013 give the closed form under quadratic costs; Davis & Norman 1990 the proportional-cost band intuition). Fold band-width and agreement into the target itself, once. *(Phase R2.)*

### D-5. Dead risk configuration

`TradingConfig.stop_loss` and `take_profit` are validated in `__post_init__` and consumed nowhere in `src/trading.py`. Documented risk controls that do not execute are worse than absent ones — they license false beliefs. Implement (per-position exit checks in the bar loop, fills via the same next-open queue) or delete. *(Phase R2.)*

### D-6. Conformal layer: dependent realizations and a target one step removed from PnL

The EnbPI-variant calibration over PurgedWalkForward OOF residuals is a defensible adaptation (Xu & Xie 2021; Foygel Barber et al. 2021). Two caveats. The ACI feedback realizes a band every bar against an $h$-bar outcome, so consecutive coverage events share $h{-}1$ bars of return — positively dependent miss streaks that shift the effective meaning of $\gamma$ (Gibbs & Candès 2021's guarantees survive, being adversarial, but the calibration target drifts from its nominal interpretation). Second, the bands cover the *ideal position*, a clipped transform of realized return — coverage in that space has no direct PnL semantics, and its width then re-enters as yet another confidence multiplier (D-4). Keep the machinery; relocate its output into the single sizing function. *(Phase R2, notes.)*

---

## §6 Execution and accounting assumptions

### E-1. Impact is linear in the wrong denominator

`slippage_bps` charges impact $\propto |\text{notional}|/\text{portfolio value}$ — linear, and relative to *own equity* rather than market liquidity. A \$10k account trading AAPL pays the same bps as a \$10B fund trading the same fraction of itself. The empirical literature is consistent on both counts: impact is concave (≈ square-root) in participation measured against *ADV* (Almgren et al. 2005; Tóth et al. 2011; Frazzini, Israel & Moskowitz 2018). The current form overstates costs for small accounts and catastrophically understates them at size, which means the harness cannot answer the capacity question that "competitive" ultimately turns on. Refactor: $\text{impact bps} = c\,\sigma_{daily}\sqrt{Q/\text{ADV}}$ with per-symbol volume from the existing bars. *(Phase R4.)*

### E-2. Auction microstructure (note only)

MOO fills at the open print plus half-spread is a reasonable daily-bar abstraction; strictly, opening auctions have no quoted spread but do have imbalance-driven dislocation. Acceptable at current granularity; revisit if sizing grows.

### E-3. Per-symbol capital silos

Each symbol is backtested as an independent \$10k book; there is no shared cash, netting, or book-level vol targeting, and PBO/baseline comparisons are per-symbol. Equal-weight averaging of per-symbol returns (`combine_symbol_returns`) approximates a portfolio only under permanent full deployment. A real book's risk, borrow, and impact interact across names. *(Phase R4.)*

### E-4. Zombie orders (defect)

`ExecutionModel.step` fills only orders with `submit_bar == bar_idx − 1`; an order kept queued because its symbol lacked OHLC that bar can *never* fill on any later bar — it lingers silently until `drop_pending`. Either fill at the next available bar or drop-with-log immediately. *(Phase R0, one-liner.)*

---

## §7 Statistical-arbitrage path

The stat-arb subsystem is the most methodologically self-aware part of the repository — train-only formation, FDR gating, fold-flattening, `pair_set_dsr`. The remaining assumptions are sharper for it.

### S-1. Min-of-orderings p-values poison the FDR input

`_candidate_from_pair` is evaluated in both orderings and the survivor is `min` by cointegration p-value; BH then runs over these minima with $m$ = number of raw pairs. The minimum of two (highly correlated but non-identical) tests is stochastically smaller than uniform under the null, so the BH gate is anti-conservative — the scan admits more false pairs than `fdr_alpha` advertises. Fix: feed BH the Bonferroni-adjusted $p^* = \min(1, 2\min(p_{yx}, p_{xy}))$, or fix the orientation ex ante (e.g., higher-priced name as $y$). *(Phase R3.)*

### S-2. ADF-on-residuals uses the wrong null distribution

The residual `adfuller` check applies standard Dickey–Fuller critical values to residuals from an *estimated* cointegrating regression; the correct reference distribution is Phillips–Ouliaris (1990) / Engle–Granger, which `statsmodels.coint` already supplies. The second gate is therefore an anti-conservative redundancy whose recorded `adf_pvalue` is misleading as a diagnostic. Drop it or relabel it as a non-inferential descriptive statistic. *(Phase R3.)*

### S-3. Convergence truncation: half-life ≤ 60 inside 63-bar folds forced flat

Selection admits half-lives up to 60 bars; the test window is 63 bars and `_force_fold_flat` zeroes the final targets. An OU-type spread entered mid-fold with $\text{HL} \sim 40$ is overwhelmingly likely to be cut before reversion completes — the design systematically truncates the very profit cycle the strategy is predicated on, converting "pairs trading" into "short-window z-momentum-reversal." The deliberate rejection of carry rules is defensible engineering; the half-life cap must then be tied to it: $\text{HL}_{max} \lesssim \text{test\_bars}/3$ (so a typical trade completes ~2 half-lives), or carry must be designed. *(Phase R3.)*

### S-4. Cointegration among six megacaps is mostly a factor bet

AAPL/MSFT/GOOG/AMZN/META/NVDA share one dominant common factor; pairwise EG "cointegration" in this universe largely captures the stability of relative factor loadings, which the 2020–2023 regime sequence repeatedly broke (the `beta_drift` filter is the right instinct, applied one layer too late). The docs already disclaim factor neutrality; elevate the disclaimer to design: residualize returns against a factor model first (PCA factors per Avellaneda & Lee 2010, or sector ETFs) and run the stationarity machinery on residuals — and broaden the universe, since pairs methods earn their keep on breadth (Gatev, Goetzmann & Rouwenhorst 2006 used the full CRSP cross-section). *(Phase R4.)*

### S-5. Multiplicity beyond the pair ledger

`pair_set_dsr` honestly deflates over selected pair-books, but the CLI knobs (`entry_z`, `exit_z`, `stop_z`, `z_window`, `formation_bars`, …) live outside any trial registry, and per-fold FDR does not control the family across ~16 rolling folds. Also note the stop-loss state machine is non-sticky: after a $|z| \ge 4$ stop, re-entry is permitted the moment $|z|$ re-crosses entry while the spread is still dislocating — divergence bleed. *(Phase R3.)*

---

## §8 Data-layer assumptions

**I-1 (documented, restated for the register).** The universe is survivorship-tainted by construction; the as-of guard drops late listings but cannot resurrect delisted names. For any "competitive" claim this graduates from caveat to blocker — delisting-complete data (CRSP/Norgate) is a prerequisite, not an enhancement.

**I-2.** Single-vendor monoculture with trust-on-filename caching: no cross-vendor reconciliation, no checksum on cached frames, vendor revisions invisible (interacts with T-4).

**I-3.** `clean_data_for_training` makes its >30%-NaN column-drop decision on the *full* series before splitting — a structural (column-selection) leak, minor but real — and `fillna(0)` plants absurd warmup values (RSI = 0 beside close = 150) in early train rows.

**I-4.** `transform_features` applies `bfill()` — within a test transform this pulls *future* rows backward into earlier NaNs. The backtest path skips transforms entirely (T-1), so today this only contaminates training-time fold metrics; after the T-1 fix it would contaminate the live path. Replace with ffill-then-train-derived-constant.

**I-5.** The scaler shape-mismatch branches silently pad/truncate fitted parameters with zeros/ones to "match" drifted column sets — silent corruption dressed as robustness. Fail loud; column-set drift between fit and transform is always a bug upstream.

**I-6 (documented).** Basic-tier dividend 403s + the `--no_dividends` recommendation mean cross-run results mix price-return and total-return conventions; tag every results artifact with the convention used.

---

## §9 Validation-theory assumptions

**V-1. PBO is computed over the wrong selection set.** CSCV runs over {ensemble, BuyAndHold, MA-X, TSMOM} — four strategies, giving rank percentiles in {0.2, 0.4, 0.6, 0.8}. But the selection event that actually occurred was the 18-trial sweep (plus development iterations); the four-strategy PBO answers a question nobody asked. The sweep already materializes per-trial daily return series — assemble the $T \times 18$ matrix and run CSCV over *that*. Free, honest, and far more informative. *(Phase R3.)*

**V-2. The trial ledger undercounts researcher degrees of freedom.** DSR deflates against the 18-point grid, but member-set choices, default thresholds, the 28-concern audit cycle, and every pre-sweep iteration are uncounted trials in the Harvey–Liu (2015) sense. MLflow has been recording all of it: mine the tracking store into an append-only experiment ledger and let global $N$ feed `expected_max_sharpe`. Complement DSR with selection-robust tests over the full strategy panel — White's (2000) Reality Check, Hansen's (2005) SPA, or a Model Confidence Set (Hansen, Lunde & Nason 2011). *(Phase R3.)*

**V-3. Seed trials are not independent.** The False Strategy Theorem's $E[\max]$ presumes independent trials; seeds share the data path, so the seed-set DSR in `rl_seed_eval` under-deflates. Keep mean ± std as primary (as the script already does) and label the seed-DSR as a lower bound on deflation.

**V-4. Sharpe convention mix.** `calculate_performance_metrics` divides geometric annualized *excess* return by arithmetic √252-scaled daily vol, while PSR/DSR correctly operate on periodic moments — harmonize the headline number to the periodic convention to avoid cross-report drift.

**V-5. Expanding windows encode regime stationarity.** Expanding training assumes the feature→forward-return map is stable from 2020 onward — a sample containing a pandemic crash, ZIRP melt-up, 2022 rate shock, and AI run. Report rolling-window results alongside expanding, and per-fold metrics *by regime label*, before trusting aggregates.

---

## §10 The strategy-level constraint: breadth

One assumption sits above all components: that a 4–6 name daily-bar system can be made "somewhat competitive" by improving its parts. The fundamental law of active management (Grinold 1989; Grinold & Kahn 2000) says $IR \approx IC \cdot \sqrt{BR}$: with four weakly independent names at daily horizon, breadth is so small that even an excellent information coefficient yields a thin information ratio before costs — and §6 shows costs are currently modeled in a way that cannot certify capacity anyway. The refactor program below is therefore best read as *preparation for breadth*: a parity-clean harness (R0), stationary representations (R1), a cost-aware decision layer (R2), and honest statistics (R3) are exactly the components that survive scaling the universe from 4 names to 400 (R4) — which is where competitiveness, if it exists, will come from. This reframing also dissolves a false dichotomy in the current design: the directional path and the stat-arb path converge at scale into one cross-sectional architecture (score → residualize → portfolio construct → execute), of which today's two pipelines are special cases.

**Empirical update — cost-bound before signal-bound (post-audit residual slice).** The R4 residualized architecture was not deferred to last; a thin vertical slice was built and run early, and at breadth. On a survivorship-free S&P 500 panel (574 priced names, 2020–2026 — the I-1 blocker addressed with a point-in-time membership mask, the 59 unpriced names confirmed as real delistings), the Avellaneda–Lee 2010 residual-reversion signal (per-name regression on PCA eigenportfolios) carries a *real gross edge* — **gross annualized Sharpe ≈ +0.23** — that **nets ≈ −0.65** once next-open costs are charged, with cost exceeding gross in roughly half the folds (a toll-booth regime; turnover ≈ 0.31/period). The headline constraint of this section is therefore *solved*: 487 names traded per fold, breadth is no longer the bottleneck, and the edge survives at breadth — yet the system is now **cost-bound before it is signal-bound.** The cost layer is moreover the binding *lever*, not merely a tax: a single no-trade band (0 → 0.004) cuts turnover 0.31 → 0.14 and lifts net Sharpe **−0.65 → −0.01 (breakeven)** while raising gross to +0.42 — R2's decision layer alone nearly closes a gap no amount of added IC or BR could touch. (Breakeven is not profit; the band must ultimately be sized from a Gârleanu–Pedersen cost model, not grid-search, since each band value is a counted DSR trial — and net ≈ 0 is the current best, not a positive result.) The architecture is not wrong — it produces exactly the residual mean-reversion edge the literature predicts — but R2's cost-aware decision layer and R3's net-edge accounting are not merely preparation for breadth: they are the *currently binding* work, and they gate every signal-side expansion.

The same lens disciplines the standing temptation to add alpha sophistication — multi-factor style overlays, value/momentum integration (Asness, Moskowitz & Pedersen 2013), regime-change detection (HMM, vol-state, or exterior-algebra/Clifford-bivector formulations), event-driven spread constraints. Two of these are already this program, not new scope: factor-residualized "synthetic twins" *are* S-4 and the slice above; integrated cross-sectional scoring *is* the convergence argued in this section. The remainder are signal-side bets, and the empirical result places them squarely behind the cost gate — each enriches a numerator the cost denominator is currently consuming, so their merit is unmeasurable (and unbankable) until a result promotes past `gross_edge` to `net_edge`. A matching discipline on targets: the 1.5–2.2 Sharpe band cited for institutional multi-factor EMN is an *execution* number (prime-broker financing, sub-bp commissions, a lending desk, crossing networks), and §6's E-1 shows this harness cannot yet certify capacity at any size — so the honest near-term bar is not 1.5 Sharpe but **net Sharpe > 0 under a capacity-aware cost model**, which no result has cleared.

---

## §11 Consolidated register

| ID | Anchor | Assumption / bias (one line) | Primary consequence | Tier | Phase |
|---|---|---|---|---|---|
| T-1 | backtest.py / run_segment | Models are scale-free across train/serve | Backtest measures a different system than was trained | 0 | R0 |
| T-2 | lstm.py | Label excluded from inputs (false) + zero-fill at serve | Member triply broken; contaminates OOF & training eval | 0 | R0/R1 |
| T-3 | sentiment_analysis.py / training.py | Full-window news coverage (false: 100-article cap, keyword scorer) | Recency-localized pseudo-feature | 0* | R0 |
| T-4 | split_idx.npz replay | Integer indices stay date-aligned across fetches | Silent fold drift on data revision | 0 (latent) | R0 |
| M-1 | arima.py | Member conditions on test-period data (false: frozen state) | Fold-constant forecast = anchored reversion artifact | 1 | R1 |
| M-2 | prophet.py | Seasonal model (actually ~200-regressor linear reg.) | Overfit trend extrapolation | 1 | R1 |
| M-3 | xgboost_model.py | Trees can regress price *levels* | Structural short bias in uptrends | 1 | R1 |
| M-4 | lstm_ppo env | Stationary observation space (false: raw prices) | OOD policy inputs at test | 1 | R1 |
| M-5 | env reward vs ExecutionModel | Training objective = scoring objective (false) | Policy optimizes wrong fills/costs | 1 | R2 |
| M-6 | xlstm_grpo.py | Group ranking informative on deterministic path | Path memorization; pad log-prob artifact | 1 | R1/R2 |
| D-1 | training labels vs run_segment | Label horizon = decision horizon (5 vs 1) | Noise turnover from overlapping forecasts | 1 | R1/R2 |
| D-2 | mapping.py | σ-units match return horizon (off by √h) | Position saturation, sign-only blend | 1 | R1 |
| D-3 | ensemble meta-learner | Simplex weights can express "no skill" (cannot) | Forced allocation; cost-blind, noise-fit weights | 1 | R2 |
| D-4 | trading.py signal/order | Ternary + frozen size preserves the signal (no) | Magnitude discarded; double shrinkage; flip costs | 1 | R2 |
| D-5 | TradingConfig | stop/take-profit execute (they don't) | Phantom risk controls | 2 | R2 |
| D-6 | conformal + ACI loop | Independent coverage events; PnL-relevant target | γ semantics drift; redundant confidence layer | 2 | R2 |
| E-1 | costs.py | Impact linear in own-PV fraction | No capacity inference; mispriced at size | 2 | R4 |
| E-3 | per-symbol loop | Independent capital silos ≈ portfolio | No netting/risk budgeting; coarse PBO set | 2 | R4 |
| E-4 | ExecutionModel.step | Queued orders eventually fill (never) | Silent zombie orders | 2 | R0 |
| S-1 | pairs.py orderings | min-p of two tests ~ Uniform (no) | Anti-conservative FDR | 2 | R3 |
| S-2 | pairs.py adfuller | DF tables valid on estimated residuals (no) | Misleading second gate | 2 | R3 |
| S-3 | HL≤60 vs 63-bar flat folds | Trades can complete within fold | Convergence truncation | 1 | R3 |
| S-4 | 6-megacap universe | Pairwise hedge ≈ neutrality | Hidden factor bet | 1 | R4 |
| S-5 | stat-arb CLI knobs | All trials counted | Untracked multiplicity; non-sticky stop | 2 | R3 |
| I-1..6 | data layer | See §8 | Survivorship, vendor trust, bfill, silent scaler pads | 2–3 | R0–R4 |
| V-1 | backtest PBO | PBO set = selection set (no) | Decorative PBO; real one is free | 2 | R3 |
| V-2 | sweep DSR | Grid = all trials (no) | Under-deflation vs researcher DoF | 2 | R3 |
| V-3..5 | metrics/CV | Seed independence; Sharpe convention; regime stationarity | Mild distortions | 3 | R3 |

\* T-3 is Tier-0 only for `--use_sentiment` runs.

---

## §12 The refactor program

### R-0 · Parity and ground truth *(restores the meaning of every number)*

Falsify before fixing — three cheap probes, one afternoon: (i) dump `get_model_contributions` for one fold's test rows via both the training-eval path and the backtest path and diff per-member output distributions (confirms T-1); (ii) assert `Var(ŷ_{arima})` within a backtest fold ≈ 0 (confirms M-1's live-loop face); (iii) histogram XGBoost backtest outputs against training-eval outputs (T-1 + M-3 jointly). Then: persist the fold-fit `FeatureEngineer` inside each fold artifact and apply it in `run_segment`; replace integer split replay with date spans + an index hash gate; quarantine the forecast LSTM; fix sentiment pagination or hard-disable the flag; fix zombie orders. **Exit invariant:** `test_train_serve_prediction_parity` — identical rows through both paths yield bit-identical member outputs — wired as a CI contract alongside the existing regression suite. Re-run the full backtest matrix to establish the *honest baseline* that all later phases are measured against; expect the numbers to move, possibly a lot, in either direction.

### R-1 · Representation *(stationarity everywhere)*

Canonical target becomes $h$-bar log return; members emit a **standardized score** (expected return in per-bar σ units) tagged with horizon metadata — a small signal IR through which every member must pass, ending the price/position dimensional ambiguity permanently (the B8 fix, completed). Fix the √h scaling in `forecast_to_position`; rebuild or retire ARIMA (rolling state, returns), Prophet (regressor diet or removal), LSTM (returns target, label excluded); standardize RL observations in-env; mask GRPO pads. Resolve D-1 by choosing one horizon discipline. **Exit invariant:** a property test asserting every member's score distribution is approximately scale-invariant under a synthetic 10× price-level shift of the input series.

### R-2 · Decision layer *(one sizing function, cost-aware)*

Replace the ternary signal/order pathway with a target-weight rebalancer: trade $w_{target} - w_{current}$ under a no-trade band derived from the cost model (Gârleanu–Pedersen partial adjustment as the default policy). Collapse confidence, agreement, and conformal width into the target-weight computation — exactly once. Meta-learner: $\sum w \le 1$, turnover penalty, shrinkage to equal weight, weight-stability diagnostic. Align RL reward with `ExecutionConfig`. Implement or delete stop/take-profit. **Exit invariant:** a turnover-conservation test — for a frozen signal stream, realized turnover responds monotonically and predictably to the band parameter.

### R-3 · Statistical honesty *(deflate against reality)*

PBO over the sweep's $T \times N_{trials}$ return matrix; MLflow-mined global trial ledger feeding `expected_max_sharpe`; SPA/MCS as a second opinion; stat-arb fixes S-1/S-2/S-3/S-5 (Bonferroni'd orderings, drop the residual-ADF gate, HL cap tied to `test_bars`, sticky stops, knob registry). **Exit invariant:** every results artifact embeds the trial-ledger count it was deflated against, and the deflation is recomputable from the ledger alone.

### R-4 · Market realism and breadth *(where "competitive" is decided)*

Square-root impact in %ADV; shared-capital portfolio with netting and book-level vol targeting; factor pre-residualization for the stat-arb path; universe expansion to a delisting-complete cross-section. At this point the directional and stat-arb pipelines should merge into one cross-sectional architecture (score → residualize → optimize → execute) of which both are configurations. **Gating discipline (per §10's empirical update):** R4 is signal-and-breadth work, and the residual slice shows the system is cost-bound first — so R4 is gated behind R2's cost-aware decision layer and R3's net-edge accounting rather than run beside them. Signal-side proposals that arrive under the R4 banner — regime conditioning, value/momentum integration, event-driven spread constraints — inherit that gate: deferred behind a `net_edge`-class core, since each enriches a numerator the cost denominator currently consumes. Where regime detection is pursued, the exotic formulations (exterior/geometric-algebra bivector flows) are earned only after a cheap proxy — a Gaussian HMM, a vol-state flag, a cross-sectional-dispersion break — first shows regime conditioning adds net lift at all; start at the cheap rung. **Exit invariant:** capacity curves — net Sharpe as a function of AUM — reported per strategy, which is the artifact an allocator (or you, deciding whether to deploy) actually needs.

Sequencing rationale: R0 before everything because every later measurement is otherwise uninterpretable; R1 before R2 because the decision layer should be built against stationary, correctly-scaled inputs; R3 in parallel with R2 (it touches reporting, not the trade path); R4 last because realism upgrades are only worth calibrating against a system that is at least internally true.

---

## §13 References

Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *J. Risk*, 3, 5–39. · Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). Direct estimation of equity market impact. *Risk*, 18(7). · Avellaneda, M., & Lee, J.-H. (2010). Statistical arbitrage in the US equities market. *Quant. Finance*, 10(7). · Bailey, D., & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *J. Risk*, 15(2). · Bailey, D., & López de Prado, M. (2014). The deflated Sharpe ratio. *J. Portfolio Mgmt*, 40(5). · Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. (2017). The probability of backtest overfitting. *J. Comp. Finance*, 20(4). · Beck, M., et al. (2024). xLSTM: Extended long short-term memory. arXiv:2405.04517. · Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1). · Davis, M., & Norman, A. (1990). Portfolio selection with transaction costs. *Math. OR*, 15(4). · Engle, R., & Granger, C. (1987). Co-integration and error correction. *Econometrica*, 55(2). · Foygel Barber, R., Candès, E., Ramdas, A., & Tibshirani, R. (2021). Predictive inference with the jackknife+. *Ann. Statist.*, 49(1). · Frazzini, A., Israel, R., & Moskowitz, T. (2018). Trading costs. SSRN 3229719. · Gârleanu, N., & Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs. *J. Finance*, 68(6). · Gatev, E., Goetzmann, W., & Rouwenhorst, K. G. (2006). Pairs trading. *RFS*, 19(3). · Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*. · Grinold, R. (1989). The fundamental law of active management. *J. Portfolio Mgmt*, 15(3). · Grinold, R., & Kahn, R. (2000). *Active Portfolio Management*, 2e. · Hansen, P. R. (2005). A test for superior predictive ability. *JBES*, 23(4). · Hansen, P. R., Lunde, A., & Nason, J. (2011). The model confidence set. *Econometrica*, 79(2). · Harvey, C., & Liu, Y. (2015). Backtesting. *J. Portfolio Mgmt*, 42(1). · Harvey, C., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *RFS*, 29(1). · Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, 2e. · López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. · Phillips, P. C. B., & Ouliaris, S. (1990). Asymptotic properties of residual based tests for cointegration. *Econometrica*, 58(1). · Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347. · Shao, Z., et al. (2024). DeepSeekMath: GRPO. arXiv:2402.03300. · Taylor, S., & Letham, B. (2018). Forecasting at scale. *Am. Statistician*, 72(1). · Tóth, B., et al. (2011). Anomalous price impact and the critical nature of liquidity. *PRX*, 1(2). · White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5). · Xu, C., & Xie, Y. (2021). Conformal prediction interval for dynamic time-series (EnbPI). *ICML*.
