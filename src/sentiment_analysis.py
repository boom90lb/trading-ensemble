# src/sentiment_analysis.py
"""Sentiment analysis module for news data."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.config import POLYGON_API_KEY

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analyzer for financial news using Polygon.io API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the sentiment analyzer.

        Args:
            api_key: Polygon.io API key (default: use from config)
        """
        self.api_key = api_key or POLYGON_API_KEY
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
            response = requests.get("https://api.polygon.io/v2/reference/news", params=params, headers=headers)
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

        # Calculate sentiment scores
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
                score = 0

            scores.append(score)

        # Calculate aggregate sentiment
        avg_score = np.mean(scores) if scores else 0

        # Calculate sentiment volume (number of articles)
        volume = len(articles)

        # Calculate sentiment momentum (change in sentiment over time)
        momentum: float = 0.0
        if volume >= 2:
            # Split articles into two halves by time
            half = volume // 2
            recent_scores = scores[:half]
            older_scores = scores[half:]

            recent_avg = np.mean(recent_scores) if recent_scores else 0
            older_avg = np.mean(older_scores) if older_scores else 0

            momentum = float(recent_avg - older_avg)

        return {"score": float(avg_score), "volume": float(volume), "momentum": momentum}

    def create_sentiment_features(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create sentiment features for a symbol over a date range.

        Args:
            symbol: Ticker symbol
            dates: DatetimeIndex of dates

        Returns:
            DataFrame with sentiment features
        """
        if not self.api_key:
            logger.warning("No Polygon API key provided, returning empty sentiment features")
            return pd.DataFrame(index=dates)

        try:
            # Initialize DataFrame
            sentiment_df = pd.DataFrame(index=dates)

            # Get min and max dates
            if len(dates) == 0:
                return sentiment_df

            min_date = min(dates).strftime("%Y-%m-%d")
            max_date = max(dates).strftime("%Y-%m-%d")

            # Fetch news articles
            articles = self.fetch_news(symbol, start_date=min_date, end_date=max_date)

            if not articles:
                logger.info(f"No news articles found for {symbol} between {min_date} and {max_date}")
                return sentiment_df

            # Group articles by date
            articles_by_date: Dict[pd.Timestamp, List[Dict]] = {}
            for article in articles:
                # Parse date from published_utc
                published_utc = article.get("published_utc", "")
                if published_utc:
                    date_str = published_utc.split("T")[0]  # Extract YYYY-MM-DD part
                    date = pd.Timestamp(date_str)

                    if date not in articles_by_date:
                        articles_by_date[date] = []

                    articles_by_date[date].append(article)

            # Calculate sentiment for each date
            for date in dates:
                date_articles = articles_by_date.get(date, [])
                sentiment = self.analyze_sentiment(date_articles)

                # Add sentiment features
                sentiment_df.loc[date, f"{symbol}_sentiment_score"] = sentiment["score"]
                sentiment_df.loc[date, f"{symbol}_sentiment_volume"] = sentiment["volume"]
                sentiment_df.loc[date, f"{symbol}_sentiment_momentum"] = sentiment["momentum"]

            # Forward fill sentiment for dates with no news
            sentiment_df = sentiment_df.fillna(method="ffill")  # type: ignore

            # Fill remaining NaN values with 0
            sentiment_df = sentiment_df.fillna(0)

            return sentiment_df

        except Exception as e:
            logger.error(f"Error creating sentiment features for {symbol}: {e}")
            return pd.DataFrame(index=dates)
