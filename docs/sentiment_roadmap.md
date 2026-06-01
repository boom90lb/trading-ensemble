# Sentiment Analysis Roadmap

This document tracks the sentiment-signal stack and the planned upgrade path. It
closes methodology gap **M11** (keyword-only sentiment) by documenting the
interim FinBERT scorer that shipped in Phase 3.1 and the distillation roadmap
that follows.

## Current state (Phase 3.1)

Two analyzers in [`src/sentiment_analysis.py`](../src/sentiment_analysis.py)
share one point-in-time (PIT) feature contract:

| Analyzer | Scoring | Status |
| --- | --- | --- |
| `SentimentAnalyzer` | keyword positive/negative counts → `(pos−neg)/(pos+neg)` | **deprecated**, kept as zero-dependency fallback |
| `TransformerSentimentAnalyzer` | FinBERT (`ProsusAI/finbert`) softmax → `P(pos) − P(neg) ∈ [−1, 1]` | **interim production** |

Both produce the same three columns per symbol —
`{symbol}_sentiment_{score,volume,momentum}` — and both route through the shared
`_run_sentiment_pipeline` → `_bucket_articles_by_bar` → `_build_sentiment_features`
path, so the **B9 point-in-time leakage guard** (article belongs to bar `t` iff
`bar_close[t-1] < published_utc <= bar_close[t]`, no across-bar ffill) lives in
exactly one place and cannot drift between the two scorers.

### Design choices

- **Continuous, signed score** `P(pos) − P(neg)`, not a discrete label. Downstream
  feature construction and `trading.py` signal blending want signal *strength*,
  not just direction. Neutral probability mass is implicit in the gap from ±1.
- **Label order read from `model.config.id2label`, never hardcoded.** ProsusAI's
  FinBERT class order has bitten many integrators; the wrong index silently flips
  the sign of the entire signal. `_resolve_label_index` resolves the positive and
  negative indices at construction and logs them.
- **Per-bar aggregation matches the keyword path** (mean article score, article
  count as `volume`, recent-half minus older-half `momentum`) so swapping scorers
  requires no downstream change.
- **Chunk-and-mean-pool for long articles** (`score_long_article`): articles
  under 512 tokens take a single fast pass; longer ones slide overlapping
  512-token windows and average the chunk scores, retaining the article tail that
  plain truncation would discard.
- **Lazy `transformers`/`torch` import** inside `__init__` keeps
  `import src.sentiment_analysis` cheap for callers using only the keyword path
  or the PIT helpers.

### Documented limitations of FinBERT (carry forward to distillation)

- Trained on 2014 Reuters TRC2 + Financial PhraseBank. Domain shift to current
  news (post-2020 macro regime, AI/crypto, geopolitics) is significant.
- 512-token truncation loses information on long articles; chunk-and-pool only
  partially mitigates (it averages independently-scored windows, losing
  cross-chunk context).
- BERT-base backbone caps accuracy at roughly 2019 SOTA.
- Title+description only — full article bodies are not fetched from Polygon.

## Distillation roadmap (future sprint — NOT implemented here)

The goal is a small, fast, current-domain student model with a continuous
regression head, distilled from a strong teacher. White-box knowledge
distillation (KD) because we control both teacher and student architectures.

### 1. Teacher

- **Backbone:** DeBERTa-v3-large, or **ModernBERT-base** for current-year domain
  coverage and long-context (8k) support.
- **Fine-tuning corpora:** aggregate financial-sentiment datasets — Financial
  PhraseBank (FPB), SEntFiN, FiQA — plus **soft labels from a strong LLM**
  (e.g. Claude Sonnet or GPT-4o) over *our own* news archive, to cover the
  domain shift that the public corpora miss.
- **Head:** continuous regression (sentiment ∈ [−1, 1]), not 3-way
  classification, so the student inherits signal strength.

### 2. Student (white-box KD)

- **Architecture:** DeBERTa-v3-small.
- **KD objective:** combine
  - **logit / output matching** (MSE or KL on the teacher's continuous head
    output, temperature-scaled if classification logits are retained),
  - **intermediate-layer matching** (hidden-state MSE with a learned linear
    projection from student to teacher width),
  - **attention-map matching** (KL between teacher and student attention
    distributions on aligned layers).
- **Data:** distill on the news archive (unlabeled is fine — the teacher provides
  the targets), so the student is calibrated to the deployment distribution.

### 3. Quantization

- **INT8** via ONNX Runtime or `torch.ao.quantization`, **calibrated on a
  held-out sample of the news archive** (not synthetic data) so activation ranges
  match production.
- Validate post-quantization that the continuous score correlation with the
  full-precision teacher stays above an agreed threshold (e.g. Spearman ρ ≥ 0.95)
  before promoting.

### 4. Latency / throughput targets

Depend on the deployment regime:

- **Real-time per-headline:** ~5 ms via ONNX + INT8 on CPU.
- **Throughput batch scoring:** ~1000 headlines/sec on a single GPU.

### Out of scope (flagged, not built)

- The distillation pipeline itself (teacher fine-tuning, KD training loop,
  quantization, calibration) — this document is the scaffolding only.
- A point-in-time *news* archive with revision history. The current fetch is
  Polygon's `published_utc`, which is sufficient for the B9 bar-bucketing guard
  but is not a versioned archive.
