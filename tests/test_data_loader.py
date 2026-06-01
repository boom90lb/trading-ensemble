# tests/test_data_loader.py
"""Data-loader tests: tz-awareness, range-encoded cache coverage, dividends."""

import os

import pandas as pd
import pytest

from src.data_loader import (
    BAR_TZ,
    DataLoader,
    _parse_cache_range,
    _parse_dividends_payload,
    _range_covers,
)


# --------------------------------------------------------------------------- #
# Pure cache-range helpers (§4.4)
# --------------------------------------------------------------------------- #
def test_parse_cache_range_roundtrip():
    assert _parse_cache_range(
        "AAPL_1d_2020-01-01_2021-12-31.parquet", "AAPL", "1d"
    ) == ("2020-01-01", "2021-12-31")


def test_parse_cache_range_min_sentinel():
    # A None start was encoded as the "min" sentinel -> parses back to None.
    assert _parse_cache_range("AAPL_1d_min_2021-12-31.parquet", "AAPL", "1d") == (
        None,
        "2021-12-31",
    )


def test_parse_cache_range_rejects_legacy_unranged():
    # Legacy {symbol}_{interval}.parquet has no range -> unknown coverage.
    assert _parse_cache_range("AAPL_1d.parquet", "AAPL", "1d") is None


def test_parse_cache_range_rejects_wrong_symbol():
    assert _parse_cache_range("MSFT_1d_2020-01-01_2021-12-31.parquet", "AAPL", "1d") is None


def test_range_covers_contained():
    assert _range_covers("2019-01-01", "2022-12-31", "2020-01-01", "2021-12-31")


def test_range_covers_start_too_late():
    # Cache starts after the request -> NOT covered (the bug this fixes).
    assert not _range_covers("2021-01-01", "2021-12-31", "2019-01-01", "2021-12-31")


def test_range_covers_end_too_early():
    assert not _range_covers("2019-01-01", "2022-06-30", "2020-01-01", "2022-12-31")


def test_range_covers_none_start_only_by_sentinel():
    # req_start None ("from the beginning") needs a sentinel-start cache.
    assert _range_covers(None, "2022-12-31", None, "2021-12-31")
    assert not _range_covers("2019-01-01", "2022-12-31", None, "2021-12-31")


# --------------------------------------------------------------------------- #
# Fetch + cache integration with a mocked Twelvedata endpoint
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _ts_payload(start, end):
    dates = pd.bdate_range(start=start, end=end)
    values = [
        {
            "datetime": d.strftime("%Y-%m-%d"),
            "open": "100.0",
            "high": "101.0",
            "low": "99.0",
            "close": "100.5",
            "volume": "1000",
        }
        for d in dates
    ]
    return {"values": values}


@pytest.fixture
def loader_with_counter(tmp_path, monkeypatch):
    """A DataLoader whose API calls are mocked and counted."""
    calls = {"n": 0, "last_params": None}

    def fake_get(url, params=None):
        calls["n"] += 1
        calls["last_params"] = params
        return _FakeResp(_ts_payload(params["start_date"], params["end_date"]))

    monkeypatch.setattr("src.data_loader.requests.get", fake_get)
    loader = DataLoader(api_key="test", cache_dir=tmp_path)
    return loader, calls, tmp_path


def test_fetch_localizes_index_to_bar_tz(loader_with_counter):
    loader, _, _ = loader_with_counter
    df = loader.fetch_historical_data("AAPL", "1d", "2021-01-01", "2021-03-01")
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert str(df.index.tz) == BAR_TZ


def test_cache_filename_encodes_requested_range(loader_with_counter):
    loader, _, tmp_path = loader_with_counter
    loader.fetch_historical_data("AAPL", "1d", "2021-01-01", "2021-03-01")
    files = list(tmp_path.glob("AAPL_1d_*.parquet"))
    assert len(files) == 1
    assert files[0].name == "AAPL_1d_2021-01-01_2021-03-01.parquet"


def test_wider_request_refetches_not_silent_slice(loader_with_counter):
    loader, calls, tmp_path = loader_with_counter
    # Narrow fetch builds a 2021-only cache.
    loader.fetch_historical_data("AAPL", "1d", "2021-01-01", "2021-03-01")
    assert calls["n"] == 1
    # Wider request is NOT covered -> must refetch (the §4.4 bug guard).
    loader.fetch_historical_data("AAPL", "1d", "2019-01-01", "2021-03-01")
    assert calls["n"] == 2
    assert len(list(tmp_path.glob("AAPL_1d_*.parquet"))) == 2


def test_covered_subrange_served_from_cache(loader_with_counter):
    loader, calls, _ = loader_with_counter
    loader.fetch_historical_data("AAPL", "1d", "2021-01-01", "2021-12-31")
    assert calls["n"] == 1
    # A sub-range of the cached window must NOT trigger a new API call.
    df = loader.fetch_historical_data("AAPL", "1d", "2021-02-01", "2021-02-15")
    assert calls["n"] == 1
    assert not df.empty
    assert df.index.min() >= pd.Timestamp("2021-02-01").tz_localize(BAR_TZ)


def test_legacy_unranged_cache_is_ignored(loader_with_counter):
    loader, calls, tmp_path = loader_with_counter
    # Simulate a pre-§4.4 cache file with unknown coverage.
    payload = _ts_payload("2021-01-01", "2021-12-31")
    legacy = pd.DataFrame(payload["values"]).set_index("datetime")
    legacy.index = pd.to_datetime(legacy.index).tz_localize(BAR_TZ)
    legacy.to_parquet(tmp_path / "AAPL_1d.parquet")
    # The legacy file's coverage can't be trusted -> a refetch happens.
    loader.fetch_historical_data("AAPL", "1d", "2021-01-01", "2021-06-30")
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# §4.2 tz assert in the training cleanup
# --------------------------------------------------------------------------- #
def test_clean_data_asserts_tz_aware_index():
    from scripts.training import clean_data_for_training

    naive = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.date_range("2021-01-01", periods=3, freq="D"),
    )
    with pytest.raises(AssertionError):
        clean_data_for_training(naive)

    aware = naive.copy()
    aware.index = aware.index.tz_localize(BAR_TZ)
    # Should not raise.
    clean_data_for_training(aware)


# --------------------------------------------------------------------------- #
# Dividends parsing (§4.1) — pure
# --------------------------------------------------------------------------- #
def test_parse_dividends_payload_normal():
    payload = {
        "meta": {"symbol": "AAPL"},
        "dividends": [
            {"ex_date": "2021-02-05", "amount": "0.205"},
            {"ex_date": "2021-05-07", "amount": 0.22},
        ],
    }
    s = _parse_dividends_payload(payload)
    assert list(s.index) == [
        pd.Timestamp("2021-02-05", tz=BAR_TZ),
        pd.Timestamp("2021-05-07", tz=BAR_TZ),
    ]
    assert s.loc[pd.Timestamp("2021-02-05", tz=BAR_TZ)] == pytest.approx(0.205)
    assert str(s.index.tz) == BAR_TZ


def test_parse_dividends_payload_empty_and_error():
    assert _parse_dividends_payload({"dividends": []}).empty
    assert _parse_dividends_payload({"status": "error", "message": "x"}).empty
    assert _parse_dividends_payload({"meta": {}}).empty  # no 'dividends' key
    assert _parse_dividends_payload("garbage").empty


def test_parse_dividends_payload_sums_same_ex_date():
    # Regular + special dividend on the same ex-date -> summed.
    payload = {
        "dividends": [
            {"ex_date": "2021-02-05", "amount": 0.2},
            {"ex_date": "2021-02-05", "amount": 0.5},
        ]
    }
    s = _parse_dividends_payload(payload)
    assert len(s) == 1
    assert s.iloc[0] == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# fetch_dividends caching/filtering with a mocked endpoint
# --------------------------------------------------------------------------- #
def test_fetch_dividends_caches_filters_and_reuses(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None):
        calls["n"] += 1
        assert url.endswith("/dividends")
        return _FakeResp(
            {
                "dividends": [
                    {"ex_date": "2021-02-05", "amount": 0.2},
                    {"ex_date": "2021-08-06", "amount": 0.22},
                ]
            }
        )

    monkeypatch.setattr("src.data_loader.requests.get", fake_get)
    loader = DataLoader(api_key="test", cache_dir=tmp_path)

    s = loader.fetch_dividends("AAPL", "2021-01-01", "2021-12-31")
    assert calls["n"] == 1
    assert len(s) == 2
    assert (tmp_path / "AAPL_dividends_2021-01-01_2021-12-31.parquet").exists()

    # Covered sub-range is served from cache (no new API call) and filtered.
    s2 = loader.fetch_dividends("AAPL", "2021-03-01", "2021-12-31")
    assert calls["n"] == 1
    assert list(s2.index) == [pd.Timestamp("2021-08-06", tz=BAR_TZ)]


def test_fetch_dividends_no_key_returns_empty(tmp_path):
    loader = DataLoader(api_key=None, cache_dir=tmp_path)
    loader.api_key = None  # force no-key path even if env has one
    assert loader.fetch_dividends("AAPL", "2021-01-01", "2021-12-31").empty


# --------------------------------------------------------------------------- #
# Live integration (§4.1 verification) — gated by a real API key
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.environ.get("TWELVEDATA_API_KEY"),
    reason="needs live TWELVEDATA_API_KEY",
)
def test_aapl_split_adjusted_and_dividends_live(tmp_path):
    loader = DataLoader(cache_dir=tmp_path)
    df = loader.fetch_historical_data("AAPL", "1d", "2020-06-01", "2020-12-31")
    assert not df.empty
    split_day = pd.Timestamp("2020-08-31", tz=BAR_TZ)
    pre = df.loc[df.index < split_day, "close"].iloc[-1]
    post = df.loc[df.index >= split_day, "close"].iloc[0]
    # AAPL did a 4:1 split on 2020-08-31; a split-adjusted series shows no ~4x
    # jump across it (the §4.1 empirical check the plan asks for).
    assert pre / post < 1.5
    divs = loader.fetch_dividends("AAPL", "2019-01-01", "2021-12-31")
    assert not divs.empty  # AAPL pays regular dividends
