# src/sentiment_analysis.py
"""Sentiment analysis module for news data.

Two analyzers share the same point-in-time (PIT) feature contract:

- :class:`SentimentAnalyzer` — legacy keyword-count scorer. **Deprecated**; kept
  for backward compatibility and as a no-dependency fallback. Prefer the
  transformer analyzer for any new work.
- :class:`TransformerSentimentAnalyzer` — FinBERT (or any HF sequence-classifier)
  producing a continuous ``P(pos) - P(neg) ∈ [-1, 1]`` score per article.

Both analyzers reuse :func:`_bucket_articles_by_bar` (the B9 leakage guard) and
:func:`_build_sentiment_features` so the strict UTC→ET bar bucketing lives in
exactly one place. An analyzer only supplies a per-bar ``score_bucket`` callback;
it never re-implements the leak-prone join.
"""

import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.config import POLYGON_API_KEY

logger = logging.getLogger(__name__)

# Per-bar sentiment summary callback: given the articles bucketed into one bar,
# return {"score": float, "volume": float, "momentum": float}.
ScoreBucketFn = Callable[[List[Dict]], Dict[str, float]]


def _bucket_articles_by_bar(dates: pd.DatetimeIndex, articles: List[Dict]) -> Dict[int, List[Dict]]:
    """Bucket news articles into trading bars under strict point-in-time semantics.

    Article belongs to bar ``t`` iff ``bar_close[t-1] < published_utc <= bar_close[t]``,
    where ``bar_close[t] = dates[t] + 16h`` interpreted in America/New_York. This is
    the B9 leakage guard (Phase 2.8): an article published after a bar's 16:00 ET
    close attaches to the NEXT bar, never the bar it was published in by
    calendar-date string slicing. Articles published after the last bar's close
    have no bar to attach to and are dropped.

    Args:
        dates: tz-aware (America/New_York) DatetimeIndex of trading-bar timestamps.
        articles: news articles with an RFC3339 ``published_utc`` field.

    Returns:
        Mapping from integer bar position → list of articles attached to that bar.
        Bars with no news are absent from the mapping (caller fills 0).
    """
    # Bar close times in UTC: 16:00 ET → UTC, used as the strict cutoff.
    bar_close_utc = (dates + pd.Timedelta(hours=16)).tz_convert("UTC")
    n_bars = len(dates)

    articles_by_bar: Dict[int, List[Dict]] = {}
    for article in articles:
        published_str = article.get("published_utc", "")
        if not published_str:
            continue
        try:
            t = pd.Timestamp(published_str)
        except (ValueError, TypeError):
            continue
        if t.tz is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        # searchsorted(side="left") returns smallest i with bar_close[i] >= t.
        idx = int(bar_close_utc.searchsorted(t, side="left"))
        if idx >= n_bars:
            # Published after the last bar's close; no bar to attach to.
            continue
        articles_by_bar.setdefault(idx, []).append(article)
    return articles_by_bar


def _build_sentiment_features(
    symbol: str,
    dates: pd.DatetimeIndex,
    articles: List[Dict],
    score_bucket: ScoreBucketFn,
) -> pd.DataFrame:
    """Assemble the (score, volume, momentum) feature frame from bucketed articles.

    Shared skeleton for all analyzers. Performs the PIT bucketing once via
    :func:`_bucket_articles_by_bar`, then delegates per-bar scoring to
    ``score_bucket``. Bars with no news get a zero-sentiment summary. The output
    is indexed identically to ``dates`` with columns
    ``{symbol}_sentiment_{score,volume,momentum}``.
    """
    sentiment_df = pd.DataFrame(index=dates)
    articles_by_bar = _bucket_articles_by_bar(dates, articles)
    for i, date in enumerate(dates):
        bar_articles = articles_by_bar.get(i, [])
        sentiment = score_bucket(bar_articles)
        sentiment_df.loc[date, f"{symbol}_sentiment_score"] = sentiment["score"]
        sentiment_df.loc[date, f"{symbol}_sentiment_volume"] = sentiment["volume"]
        sentiment_df.loc[date, f"{symbol}_sentiment_momentum"] = sentiment["momentum"]
    return sentiment_df.fillna(0)


def _aggregate_article_scores(article_scores: List[float]) -> Dict[str, float]:
    """Aggregate per-article continuous scores into a (score, volume, momentum) summary.

    ``score`` = mean article score; ``volume`` = article count; ``momentum`` =
    mean(recent half) − mean(older half), with articles assumed ordered newest-first
    (Polygon ``sort=published_utc, order=desc``). This matches the keyword analyzer's
    aggregation so downstream trading.py logic is identical regardless of scorer.
    """
    if not article_scores:
        return {"score": 0.0, "volume": 0.0, "momentum": 0.0}

    avg_score = float(np.mean(article_scores))
    volume = float(len(article_scores))

    momentum = 0.0
    if len(article_scores) >= 2:
        half = len(article_scores) // 2
        recent = article_scores[:half]
        older = article_scores[half:]
        momentum = float(np.mean(recent) - np.mean(older)) if recent and older else 0.0

    return {"score": avg_score, "volume": volume, "momentum": momentum}


def _run_sentiment_pipeline(
    symbol: str,
    dates: pd.DatetimeIndex,
    api_key: Optional[str],
    fetch_news: Callable[..., List[Dict]],
    score_bucket: ScoreBucketFn,
) -> pd.DataFrame:
    """End-to-end PIT sentiment pipeline shared by all analyzers.

    Guards (no key / empty dates) → tz-localize → fetch → bucket (B9 guard) →
    per-bar ``score_bucket``. The bucketing and bar bookkeeping live in
    :func:`_bucket_articles_by_bar` / :func:`_build_sentiment_features`; an
    analyzer supplies only ``fetch_news`` and ``score_bucket``.

    Args:
        symbol: Ticker symbol.
        dates: trading-bar timestamps. tz-aware America/New_York is required for
            correct joins; a tz-naive index is localized to ET with a warning.
        api_key: news-API key; absent → empty (column-less) frame, no fetch.
        fetch_news: ``(symbol, start_date, end_date) -> List[Dict]`` article fetch.
        score_bucket: per-bar summary callback (see :data:`ScoreBucketFn`).

    Returns:
        Feature frame indexed identically to ``dates``.
    """
    sentiment_df = pd.DataFrame(index=dates)
    if not api_key:
        logger.warning("No news API key provided, returning empty sentiment features")
        return sentiment_df
    if len(dates) == 0:
        return sentiment_df

    try:
        if dates.tz is None:
            logger.warning(
                "create_sentiment_features received tz-naive index; "
                "assuming America/New_York for the point-in-time join."
            )
            dates = dates.tz_localize("America/New_York")

        min_date = dates.min().strftime("%Y-%m-%d")
        max_date = dates.max().strftime("%Y-%m-%d")

        articles = fetch_news(symbol, start_date=min_date, end_date=max_date)

        if not articles:
            logger.info(f"No news articles found for {symbol} between {min_date} and {max_date}")
            return pd.DataFrame(index=dates).fillna(0)

        return _build_sentiment_features(symbol, dates, articles, score_bucket)

    except Exception as e:
        logger.error(f"Error creating sentiment features for {symbol}: {e}")
        return pd.DataFrame(index=dates)


class SentimentAnalyzer:
    """Keyword-count sentiment analyzer for financial news (Polygon.io API).

    .. deprecated::
        This keyword scorer is a coarse heuristic kept for backward compatibility
        and as a zero-dependency fallback. Prefer
        :class:`TransformerSentimentAnalyzer` (FinBERT), which produces a
        continuous, signed sentiment score. The point-in-time bar bucketing is
        identical between the two (shared :func:`_bucket_articles_by_bar`).
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the sentiment analyzer.

        Args:
            api_key: Polygon.io API key (default: use from config)
        """
        self.api_key = POLYGON_API_KEY if api_key is None else api_key
        if not self.api_key:
            logger.warning("No Polygon API key provided, sentiment analysis will be limited")

        # Cache for sentiment data
        self.sentiment_cache: Dict[str, Dict[pd.Timestamp, Dict]] = {}  # {symbol: {date: sentiment_data}}

    def fetch_news(
        self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """Fetch news articles for a symbol from Polygon.io.

        Args:
            symbol: Ticker symbol
            start_date: Start date (format: YYYY-MM-DD)
            end_date: End date (format: YYYY-MM-DD)
            limit: Maximum number of articles to fetch

        Returns:
            List of news articles
        """
        if not self.api_key:
            logger.warning("No Polygon API key provided")
            return []

        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Default to 30 days before end date
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=30)
            start_date = start_dt.strftime("%Y-%m-%d")

        try:
            # Prepare API parameters
            params = {
                "ticker": symbol,
                "published_utc.gte": f"{start_date}T00:00:00Z",
                "published_utc.lte": f"{end_date}T23:59:59Z",
                "limit": limit,
                "sort": "published_utc",
                "order": "desc",
            }

            # Make API request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get("https://api.polygon.io/v2/reference/news", params=params, headers=headers)  # type: ignore
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "status" in data and data["status"] != "OK":
                error_msg = data.get("error", "Unknown API error")
                logger.error(f"API error: {error_msg}")
                return []

            # Extract results
            return data.get("results", [])

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def analyze_sentiment(self, articles: List[Dict]) -> Dict[str, float]:
        """Analyze sentiment of news articles.

        This is a simplified implementation that uses keywords for sentiment.
        In production, use a more sophisticated sentiment analysis model.

        Args:
            articles: List of news articles

        Returns:
            Dictionary with sentiment scores
        """
        if not articles:
            return {"score": 0.0, "volume": 0.0, "momentum": 0.0}

        # Positive and negative keywords (simplified)
        positive_keywords = [
            "positive",
            "bullish",
            "upbeat",
            "optimistic",
            "growth",
            "profit",
            "up",
            "gain",
            "surge",
            "rally",
            "strong",
            "beat",
            "exceed",
            "outperform",
            "upgrade",
            "buy",
            "recommend",
            "success",
            "innovation",
            "partnership",
            "launch",
            "record",
            "dividend",
            "buyback",
        ]

        negative_keywords = [
            "negative",
            "bearish",
            "pessimistic",
            "downbeat",
            "decline",
            "loss",
            "down",
            "drop",
            "fall",
            "plunge",
            "weak",
            "miss",
            "underperform",
            "downgrade",
            "sell",
            "avoid",
            "fail",
            "layoff",
            "cut",
            "delay",
            "recall",
            "investigation",
            "lawsuit",
            "debt",
            "default",
            "bankruptcy",
        ]

        # Per-article keyword score in [-1, 1], then aggregate via the shared summary.
        scores = []
        for article in articles:
            title = article.get("title", "").lower()
            description = article.get("description", "").lower()

            # Count positive and negative keywords
            positive_count = sum(1 for word in positive_keywords if word in title or word in description)
            negative_count = sum(1 for word in negative_keywords if word in title or word in description)

            # Calculate article score
            total_count = positive_count + negative_count
            if total_count > 0:
                score = (positive_count - negative_count) / total_count
            else:
                score = 0.0

            scores.append(float(score))

        return _aggregate_article_scores(scores)

    def create_sentiment_features(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create strict point-in-time sentiment features for a symbol.

        See :func:`_run_sentiment_pipeline` for the bucketing semantics (B9 guard).
        Per-bar scoring uses the keyword :meth:`analyze_sentiment`.
        """
        return _run_sentiment_pipeline(
            symbol=symbol,
            dates=dates,
            api_key=self.api_key,
            fetch_news=self.fetch_news,
            score_bucket=self.analyze_sentiment,
        )


def _article_text(article: Dict) -> str:
    """Concatenate title + description for scoring (mirrors the keyword path's fields)."""
    title = article.get("title", "") or ""
    description = article.get("description", "") or ""
    return f"{title}. {description}".strip()


class TransformerSentimentAnalyzer:
    """Transformer sentiment analyzer (FinBERT by default) over financial news.

    Produces a continuous, signed sentiment score per article,
    ``P(positive) - P(negative) ∈ [-1, 1]`` (neutral mass is implicit in the gap
    from ±1), rather than discrete labels — so downstream feature construction has
    signal *strength*, not just direction. Per-bar aggregation matches the keyword
    analyzer (mean score, article count as volume, recent−older momentum) so the
    feature schema and trading.py logic are unchanged.

    The point-in-time bar join (B9 guard) is shared with the keyword analyzer via
    :func:`_run_sentiment_pipeline` / :func:`_bucket_articles_by_bar`.

    Label-order safety: FinBERT's class order is read from
    ``model.config.id2label`` at construction time — it is NOT hardcoded, because
    the wrong order silently flips the signal (a classic FinBERT trap).
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "auto",
        api_key: Optional[str] = None,
        max_length: int = 512,
    ):
        """Initialize the analyzer.

        Args:
            model_name: HuggingFace sequence-classification model id.
            device: "auto" (CUDA if available else CPU), or an explicit torch device.
            api_key: Polygon.io key for news fetch (default: config). The news
                fetch is delegated to a :class:`SentimentAnalyzer` instance so the
                Polygon-specific request logic lives in one place.
            max_length: tokenizer truncation / chunk window length.
        """
        # Lazy heavy imports: keep `import src.sentiment_analysis` cheap and avoid
        # forcing transformers/torch resolution for callers using only the keyword path.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device).eval()

        # Resolve positive/negative class indices from the model config — never
        # hardcode the order. id2label values are case-insensitive label strings.
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.pos_idx = self._resolve_label_index(id2label, "positive")
        self.neg_idx = self._resolve_label_index(id2label, "negative")
        logger.info(
            "TransformerSentimentAnalyzer(%s) id2label=%s → pos=%d neg=%d on %s",
            model_name,
            id2label,
            self.pos_idx,
            self.neg_idx,
            self.device,
        )

        # News fetch is reused from the keyword analyzer (Polygon-specific).
        self._fetcher = SentimentAnalyzer(api_key=api_key)
        self.api_key = self._fetcher.api_key

    @staticmethod
    def _resolve_label_index(id2label: Dict[int, str], target: str) -> int:
        """Find the class index whose label matches ``target`` (e.g. 'positive')."""
        for idx, label in id2label.items():
            if target in label:
                return idx
        raise ValueError(
            f"Model id2label={id2label} has no '{target}' class; "
            "TransformerSentimentAnalyzer expects a pos/neg/neutral head."
        )

    def fetch_news(self, symbol, start_date=None, end_date=None, limit: int = 100) -> List[Dict]:
        """Fetch news (delegates to the keyword analyzer's Polygon fetch)."""
        return self._fetcher.fetch_news(symbol, start_date=start_date, end_date=end_date, limit=limit)

    def score(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Score texts to continuous sentiment in ``[-1, 1]`` (``P(pos) - P(neg)``).

        Texts longer than ``max_length`` tokens are truncated here; use
        :meth:`score_long_article` for the chunk-and-mean-pool path that retains
        the article tail.
        """
        if not texts:
            return np.zeros(0, dtype=float)

        out: List[np.ndarray] = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                probs = self.model(**enc).logits.softmax(dim=-1).cpu().numpy()
                out.append(probs[:, self.pos_idx] - probs[:, self.neg_idx])
        return np.concatenate(out)

    def score_long_article(self, text: str, chunk_overlap: int = 50, batch_size: int = 32) -> float:
        """Score a long article by sliding 512-token windows and mean-pooling.

        Articles within ``max_length`` tokens take the single-pass fast path. Longer
        articles are split into overlapping ``max_length``-token windows (special
        tokens reserved), each window decoded and scored via :meth:`score`, and the
        chunk scores are averaged. Mean-pool (unweighted) per the sprint scope.
        """
        if not text:
            return 0.0

        # Encode once without special tokens / truncation to inspect length.
        token_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        # Reserve room for [CLS]/[SEP] (or model-specific specials).
        window = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        if window <= 0:  # pathological tokenizer config
            window = self.max_length

        if len(token_ids) <= window:
            return float(self.score([text], batch_size=batch_size)[0])

        stride = max(1, window - chunk_overlap)
        chunks: List[str] = []
        for start in range(0, len(token_ids), stride):
            chunk_ids = token_ids[start : start + window]
            if not chunk_ids:
                break
            chunks.append(self.tokenizer.decode(chunk_ids, skip_special_tokens=True))
            if start + window >= len(token_ids):
                break

        chunk_scores = self.score(chunks, batch_size=batch_size)
        return float(np.mean(chunk_scores)) if len(chunk_scores) else 0.0

    def analyze_sentiment(self, articles: List[Dict]) -> Dict[str, float]:
        """Per-bar summary: score each article continuously, then aggregate.

        Drop-in replacement for the keyword analyzer's bucket scorer. Articles are
        scored via the chunk-aware :meth:`score_long_article` so long stories keep
        their tail; aggregation (mean / count / recent−older momentum) is shared
        with the keyword path via :func:`_aggregate_article_scores`.
        """
        if not articles:
            return {"score": 0.0, "volume": 0.0, "momentum": 0.0}
        article_scores = [self.score_long_article(_article_text(a)) for a in articles]
        return _aggregate_article_scores(article_scores)

    def create_sentiment_features(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create strict point-in-time sentiment features (FinBERT scoring).

        Identical PIT contract and column schema as the keyword analyzer; only the
        per-bar scoring differs. See :func:`_run_sentiment_pipeline`.
        """
        return _run_sentiment_pipeline(
            symbol=symbol,
            dates=dates,
            api_key=self.api_key,
            fetch_news=self.fetch_news,
            score_bucket=self.analyze_sentiment,
        )
