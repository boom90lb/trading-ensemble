"""Tests for TransformerSentimentAnalyzer (Phase 3.1, FinBERT interim).

The real ProsusAI/finbert weights (~440MB) are NOT downloaded in CI. Unit tests
monkeypatch the HuggingFace `from_pretrained` calls with a fake tokenizer/model
returning canned logits, so we can assert:
  - label order is read from model.config.id2label (NOT hardcoded), and the
    signed score uses the resolved pos/neg indices even under a permuted head;
  - score() returns continuous values in [-1, 1] and batches correctly;
  - score_long_article() chunks >max_length inputs and mean-pools;
  - analyze_sentiment aggregates to the (score, volume, momentum) schema;
  - create_sentiment_features preserves the B9 point-in-time bar bucketing.

One real-weights integration test is gated on RUN_FINBERT_TESTS=1.
"""

from __future__ import annotations

import os
from typing import Dict, List
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.sentiment_analysis import TransformerSentimentAnalyzer


# --- Fakes -------------------------------------------------------------------


class _FakeEncoding(dict):
    """Mimic a HF BatchEncoding: dict of tensors with a no-op .to(device)."""

    def to(self, device):  # noqa: D401 - mimic transformers API
        return self


class _FakeTokenizer:
    """Deterministic word-split tokenizer.

    `score_long_article` only needs encode/decode/num_special_tokens_to_add to be
    self-consistent; the model's logits are driven by keywords in the decoded
    text, not the ids. The tokenizer hands the batch text to the model on each
    call (dunder __call__ can't be patched per-instance, so wire it via a ref).
    """

    def __init__(self, model: "_FakeModel"):
        self._model = model
        # Round-trippable word<->id vocab so chunk decode preserves keywords.
        self._vocab: Dict[str, int] = {}
        self._inv: Dict[int, str] = {}

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            i = len(self._vocab) + 100
            self._vocab[word] = i
            self._inv[i] = word
        return self._vocab[word]

    def __call__(self, batch, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        self._model._current_batch = list(batch)
        ids = torch.zeros((len(batch), 1), dtype=torch.long)
        return _FakeEncoding(input_ids=ids, attention_mask=torch.ones_like(ids))

    def encode(self, text, add_special_tokens=False, truncation=False):
        return [self._id(w) for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self._inv.get(i, "w") for i in ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class _FakeConfig:
    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label


class _FakeModel:
    """Returns logits driven by a keyword in the text so scores are predictable.

    The output column order follows `id2label`: whichever index is labeled
    "positive" gets the high logit for bullish text, etc. This lets a permuted
    head test catch a hardcoded-index regression.
    """

    def __init__(self, id2label: Dict[int, str]):
        self.config = _FakeConfig(id2label)
        self._id2label = id2label
        self._current_batch: List[str] = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        batch_texts = self._current_batch
        n_classes = len(self._id2label)
        logits = torch.zeros((len(batch_texts), n_classes))
        # Map label string -> column index.
        col = {v.lower(): k for k, v in self._id2label.items()}
        for row, text in enumerate(batch_texts):
            t = text.lower()
            if "bull" in t:
                logits[row, col["positive"]] = 5.0
            elif "bear" in t:
                logits[row, col["negative"]] = 5.0
            elif "neutral" in col:
                logits[row, col["neutral"]] = 5.0
        return _ModelOutput(logits)


class _ModelOutput:
    def __init__(self, logits):
        self.logits = logits


def _build_analyzer(id2label: Dict[int, str], api_key: str = "test-key") -> TransformerSentimentAnalyzer:
    """Construct a TransformerSentimentAnalyzer with fakes patched in."""
    model = _FakeModel(id2label)
    tok = _FakeTokenizer(model)

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tok), patch(
        "transformers.AutoModelForSequenceClassification.from_pretrained", return_value=model
    ):
        analyzer = TransformerSentimentAnalyzer(api_key=api_key, device="cpu")
    return analyzer


# Standard ProsusAI/finbert order and a permuted order for the label-trap test.
PROSUS_ORDER = {0: "positive", 1: "negative", 2: "neutral"}
PERMUTED_ORDER = {0: "neutral", 1: "negative", 2: "positive"}


# --- Lazy import -------------------------------------------------------------


def test_module_import_does_not_load_transformers():
    """Importing the module must not import transformers (lazy __init__ import)."""
    import subprocess
    import sys

    code = "import sys; import src.sentiment_analysis; print('transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr


# --- Label-order resolution (the FinBERT trap) -------------------------------


def test_resolves_label_indices_from_config():
    a = _build_analyzer(PROSUS_ORDER)
    assert a.pos_idx == 0
    assert a.neg_idx == 1


def test_resolves_permuted_label_order():
    a = _build_analyzer(PERMUTED_ORDER)
    assert a.pos_idx == 2
    assert a.neg_idx == 1


def test_signed_score_invariant_to_label_permutation():
    """Bullish text scores positive and bearish negative regardless of head order.

    A hardcoded `logits[:,0]-logits[:,1]` would invert under PERMUTED_ORDER.
    """
    for order in (PROSUS_ORDER, PERMUTED_ORDER):
        a = _build_analyzer(order)
        scores = a.score(["bull rally", "bear crash"])
        assert scores[0] > 0.9, order
        assert scores[1] < -0.9, order


def test_missing_label_class_raises():
    with pytest.raises(ValueError, match="positive"):
        _build_analyzer({0: "negative", 1: "neutral"})


# --- score() -----------------------------------------------------------------


def test_score_returns_continuous_in_range():
    a = _build_analyzer(PROSUS_ORDER)
    scores = a.score(["bull", "bear", "calm waters"])
    assert scores.shape == (3,)
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)
    assert scores[2] == pytest.approx(0.0, abs=1e-6)  # neutral text


def test_score_empty_returns_empty():
    a = _build_analyzer(PROSUS_ORDER)
    assert a.score([]).shape == (0,)


def test_score_batches_cover_all_inputs():
    a = _build_analyzer(PROSUS_ORDER)
    texts = ["bull"] * 5 + ["bear"] * 5
    scores = a.score(texts, batch_size=3)  # 4 batches
    assert scores.shape == (10,)
    assert np.all(scores[:5] > 0.9)
    assert np.all(scores[5:] < -0.9)


# --- score_long_article() ----------------------------------------------------


def test_short_article_single_pass():
    a = _build_analyzer(PROSUS_ORDER)
    # window = max_length(512) - 2 special = 510; short text takes the fast path.
    val = a.score_long_article("bull rally strong quarter")
    assert val > 0.9


def test_long_article_chunks_and_mean_pools():
    a = _build_analyzer(PROSUS_ORDER)
    a.max_length = 10  # window = 8 tokens
    # 20 "bull" words → multiple chunks, all bullish → mean stays bullish.
    text = " ".join(["bull"] * 20)
    val = a.score_long_article(text, chunk_overlap=2)
    assert val > 0.9


def test_long_article_mean_pools_mixed_chunks():
    a = _build_analyzer(PROSUS_ORDER)
    a.max_length = 10  # window = 8
    # Each decoded chunk is uniform "w w ..." (fake decode), so to exercise the
    # mean we score a list directly instead. Here we just assert chunking runs
    # without error and stays in range for a long neutral-after-decode article.
    text = " ".join(["word"] * 40)
    val = a.score_long_article(text)
    assert -1.0 <= val <= 1.0


def test_empty_text_scores_zero():
    a = _build_analyzer(PROSUS_ORDER)
    assert a.score_long_article("") == 0.0


# --- analyze_sentiment bucket aggregation ------------------------------------


def _article(title: str, published_utc: str = "2023-01-03T15:00:00Z") -> dict:
    return {"title": title, "description": "", "published_utc": published_utc}


def test_analyze_sentiment_empty_bucket():
    a = _build_analyzer(PROSUS_ORDER)
    assert a.analyze_sentiment([]) == {"score": 0.0, "volume": 0.0, "momentum": 0.0}


def test_analyze_sentiment_aggregates():
    a = _build_analyzer(PROSUS_ORDER)
    out = a.analyze_sentiment([_article("bull"), _article("bull"), _article("bear"), _article("bear")])
    assert out["volume"] == 4.0
    assert out["score"] == pytest.approx(0.0, abs=1e-6)  # 2 bull + 2 bear
    # momentum = mean(recent half=[bull,bull]) - mean(older half=[bear,bear]) > 0
    assert out["momentum"] > 0.9


# --- PIT preservation (B9 guard via shared pipeline) -------------------------


def _et_dates(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="D", tz="America/New_York")


def test_create_sentiment_features_preserves_pit_bucketing():
    """Post-close news must attach to the NEXT bar, same as the keyword analyzer."""
    a = _build_analyzer(PROSUS_ORDER)
    dates = _et_dates("2023-01-03", 3)
    # 22:00 UTC Jan 3 = 17:00 ET, after the 16:00 close → belongs to bar Jan 4.
    articles = [_article("bull rally", published_utc="2023-01-03T22:00:00Z")]

    with patch.object(a, "fetch_news", return_value=articles):
        df = a.create_sentiment_features("AAPL", dates)

    assert df.loc[dates[0], "AAPL_sentiment_volume"] == 0.0
    assert df.loc[dates[1], "AAPL_sentiment_volume"] == 1.0
    assert df.loc[dates[1], "AAPL_sentiment_score"] > 0.9  # bullish


def test_create_sentiment_features_no_across_bar_ffill():
    a = _build_analyzer(PROSUS_ORDER)
    dates = _et_dates("2023-01-03", 5)
    articles = [
        _article("bull", published_utc="2023-01-03T15:00:00Z"),  # bar 0
        _article("bull", published_utc="2023-01-06T15:00:00Z"),  # bar 3
    ]
    with patch.object(a, "fetch_news", return_value=articles):
        df = a.create_sentiment_features("AAPL", dates)

    assert df.loc[dates[0], "AAPL_sentiment_volume"] == 1.0
    assert df.loc[dates[1], "AAPL_sentiment_volume"] == 0.0
    assert df.loc[dates[3], "AAPL_sentiment_volume"] == 1.0


def test_create_sentiment_features_no_api_key():
    a = _build_analyzer(PROSUS_ORDER, api_key="")
    dates = _et_dates("2023-01-03", 3)
    df = a.create_sentiment_features("AAPL", dates)
    assert df.empty or df.shape[1] == 0


# --- Env-gated real-weights integration --------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_FINBERT_TESTS") != "1",
    reason="set RUN_FINBERT_TESTS=1 to download ProsusAI/finbert and run live",
)
def test_real_finbert_label_order_and_sign():
    """Verify against the real model: id2label resolves and signs are correct.

    This is the in-repo guard for the FinBERT label-order trap — it asserts the
    real ProsusAI/finbert head produces a positive score for bullish text and
    negative for bearish, with the indices resolved from config (not assumed)."""
    a = TransformerSentimentAnalyzer(api_key="test-key", device="cpu")
    scores = a.score(
        [
            "The company reported record profits and raised guidance.",
            "The company missed estimates and slashed its dividend amid mounting losses.",
        ]
    )
    assert scores[0] > 0.0
    assert scores[1] < 0.0
