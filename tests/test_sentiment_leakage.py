"""Regression tests for B9 (sentiment-join leak) + B12 (tz-naive bar index).

These tests pin down the strict point-in-time semantics introduced in Phase 2.8:
- News with published_utc > bar_close attaches to a LATER bar, not the bar it
  was published in by calendar-date string slicing.
- No across-bar ffill: a bar with no news in its (prev_close, bar_close] window
  has zero sentiment, regardless of what neighboring bars have.
- Bar DatetimeIndex coming out of DataLoader is tz-aware (America/New_York).
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.data_loader import BAR_TZ, DEFAULT_REQUEST_TIMEOUT_SECONDS, _ensure_bar_tz
from src.sentiment_analysis import SentimentAnalyzer


# --- Fixtures ----------------------------------------------------------------


def _et_dates(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="D", tz="America/New_York")


def _article(published_utc: str, *, title: str = "growth", description: str = "") -> dict:
    return {"title": title, "description": description, "published_utc": published_utc}


@pytest.fixture
def analyzer():
    """SentimentAnalyzer with a stub API key — no real HTTP calls in these tests."""
    a = SentimentAnalyzer(api_key="test-key")
    return a


# --- B9: strict point-in-time bucketing -------------------------------------


def test_news_after_bar_close_does_not_leak_into_same_bar(analyzer):
    """News published at 22:00 UTC on Jan 3 (= 17:00 ET, after the 16:00 close)
    must NOT appear in bar 2023-01-03; it belongs to bar 2023-01-04."""
    dates = _et_dates("2023-01-03", 3)
    articles = [_article("2023-01-03T22:00:00Z", title="growth profit")]

    with patch.object(analyzer, "fetch_news", return_value=articles):
        df = analyzer.create_sentiment_features("AAPL", dates)

    # Same-bar leak guard: Jan 3 sees no sentiment from a post-close article.
    assert df.loc[dates[0], "AAPL_sentiment_score"] == 0.0
    assert df.loc[dates[0], "AAPL_sentiment_volume"] == 0.0
    # Next bar (Jan 4) gets the article, since 22:00 UTC Jan 3 <= 21:00 UTC Jan 4 (16:00 ET).
    assert df.loc[dates[1], "AAPL_sentiment_score"] > 0.0
    assert df.loc[dates[1], "AAPL_sentiment_volume"] == 1.0


def test_news_before_bar_close_attaches_to_same_bar(analyzer):
    """News at 20:00 UTC Jan 3 (= 15:00 ET, intraday) goes to bar 2023-01-03."""
    dates = _et_dates("2023-01-03", 3)
    articles = [_article("2023-01-03T20:00:00Z", title="growth profit")]

    with patch.object(analyzer, "fetch_news", return_value=articles):
        df = analyzer.create_sentiment_features("AAPL", dates)

    assert df.loc[dates[0], "AAPL_sentiment_score"] > 0.0
    assert df.loc[dates[0], "AAPL_sentiment_volume"] == 1.0
    assert df.loc[dates[1], "AAPL_sentiment_volume"] == 0.0


def test_no_across_bar_ffill(analyzer):
    """A bar between two news days must have zero sentiment, not the prior bar's."""
    dates = _et_dates("2023-01-03", 5)
    articles = [
        _article("2023-01-03T15:00:00Z", title="growth"),  # bar 0
        _article("2023-01-06T15:00:00Z", title="growth"),  # bar 3
    ]

    with patch.object(analyzer, "fetch_news", return_value=articles):
        df = analyzer.create_sentiment_features("AAPL", dates)

    assert df.loc[dates[0], "AAPL_sentiment_volume"] == 1.0
    assert df.loc[dates[1], "AAPL_sentiment_volume"] == 0.0  # would be 1.0 if ffill leaked
    assert df.loc[dates[2], "AAPL_sentiment_volume"] == 0.0
    assert df.loc[dates[3], "AAPL_sentiment_volume"] == 1.0
    assert df.loc[dates[4], "AAPL_sentiment_volume"] == 0.0


def test_news_after_last_bar_close_is_dropped(analyzer):
    """News published after the last bar's close has no bar to attach to and is ignored."""
    dates = _et_dates("2023-01-03", 3)
    articles = [_article("2023-01-05T22:00:00Z", title="growth")]  # 17:00 ET on last bar

    with patch.object(analyzer, "fetch_news", return_value=articles):
        df = analyzer.create_sentiment_features("AAPL", dates)

    assert (df["AAPL_sentiment_volume"] == 0.0).all()
    assert (df["AAPL_sentiment_score"] == 0.0).all()


def test_pre_window_news_attaches_to_first_bar(analyzer):
    """News published before the backtest window opens is available at the first bar."""
    dates = _et_dates("2023-01-03", 3)
    articles = [_article("2022-12-30T18:00:00Z", title="growth")]

    with patch.object(analyzer, "fetch_news", return_value=articles):
        df = analyzer.create_sentiment_features("AAPL", dates)

    assert df.loc[dates[0], "AAPL_sentiment_volume"] == 1.0
    assert df.loc[dates[1], "AAPL_sentiment_volume"] == 0.0


def test_no_api_key_returns_empty_features(analyzer):
    analyzer.api_key = None
    dates = _et_dates("2023-01-03", 3)
    df = analyzer.create_sentiment_features("AAPL", dates)
    assert df.empty or df.shape[1] == 0
    assert list(df.index) == list(dates)


def test_empty_dates_returns_empty(analyzer):
    df = analyzer.create_sentiment_features("AAPL", pd.DatetimeIndex([], tz="America/New_York"))
    assert df.empty


def test_fetch_news_passes_default_timeout(monkeypatch):
    calls = {"timeout": None}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "OK", "results": []}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["timeout"] = timeout
        assert url.endswith("/v2/reference/news")
        assert headers["Authorization"] == "Bearer test-key"
        return _Resp()

    monkeypatch.setattr("src.sentiment_analysis.requests.get", fake_get)
    out = SentimentAnalyzer(api_key="test-key").fetch_news(
        "AAPL",
        start_date="2023-01-01",
        end_date="2023-01-31",
    )

    assert out == []
    assert calls["timeout"] == DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_no_articles_returns_zeros_not_nan(analyzer):
    dates = _et_dates("2023-01-03", 3)
    with patch.object(analyzer, "fetch_news", return_value=[]):
        df = analyzer.create_sentiment_features("AAPL", dates)
    assert df.shape == (3, 0) or (df.values == 0).all()


def test_tz_naive_index_is_localized_with_warning(analyzer, caplog):
    """Compatibility path: tz-naive input is treated as ET with a warning logged."""
    import logging

    naive = pd.date_range(start="2023-01-03", periods=3, freq="D")
    articles = [_article("2023-01-03T22:00:00Z", title="growth")]

    with patch.object(analyzer, "fetch_news", return_value=articles), caplog.at_level(logging.WARNING):
        df = analyzer.create_sentiment_features("AAPL", naive)

    assert any("tz-naive" in rec.message for rec in caplog.records)
    # Same leak-guard semantics apply: post-close news goes to next bar.
    assert df.iloc[0]["AAPL_sentiment_volume"] == 0.0
    assert df.iloc[1]["AAPL_sentiment_volume"] == 1.0


# --- B12: tz-aware DatetimeIndex helper -------------------------------------


def test_ensure_bar_tz_localizes_naive_index():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=pd.date_range("2023-01-03", periods=3))
    out = _ensure_bar_tz(df)
    assert out.index.tz is not None
    assert str(out.index.tz) == BAR_TZ


def test_ensure_bar_tz_converts_other_tz():
    idx = pd.date_range("2023-01-03", periods=3, tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    out = _ensure_bar_tz(df)
    assert str(out.index.tz) == BAR_TZ
    # Underlying instants preserved across conversion.
    assert (out.index.tz_convert("UTC") == idx).all()


def test_ensure_bar_tz_idempotent_on_correct_tz():
    idx = pd.date_range("2023-01-03", periods=3, tz=BAR_TZ)
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    out = _ensure_bar_tz(df)
    assert (out.index == idx).all()
    assert str(out.index.tz) == BAR_TZ
