"""Tests for the Phase 5.1 structured-logging helper (src/logging_utils.py).

JAX-free and filesystem-isolated: every test configures logging into a
pytest ``tmp_path`` and restores the root logger's original handlers on
teardown so the test session's own logging isn't clobbered.
"""

import logging
from logging.handlers import RotatingFileHandler

import pytest

from src.logging_utils import (
    _BACKUP_COUNT,
    _MAX_BYTES,
    _SymbolDefaultFilter,
    configure_logging,
    get_symbol_logger,
)


@pytest.fixture
def restore_root_logger():
    """Snapshot/restore the root logger so configure_logging is sandboxed."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        for h in list(root.handlers):
            if getattr(h, "_te_managed", False):
                root.removeHandler(h)
                h.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def _managed_handlers():
    return [
        h for h in logging.getLogger().handlers
        if getattr(h, "_te_managed", False)
    ]


def test_creates_log_file(tmp_path, restore_root_logger):
    path = configure_logging(run_name="unit", log_dir=tmp_path)
    assert path.exists()
    assert path.parent == tmp_path
    assert path.name.startswith("run_unit_")
    assert path.suffix == ".log"


def test_default_run_name(tmp_path, restore_root_logger):
    path = configure_logging(log_dir=tmp_path)
    assert path.name.startswith("run_")
    # No run_name -> "run_{ts}.log", not "run_None_...".
    assert "None" not in path.name


def test_installs_console_and_rotating_file_handler(tmp_path, restore_root_logger):
    configure_logging(log_dir=tmp_path)
    managed = _managed_handlers()
    types = sorted(type(h).__name__ for h in managed)
    assert types == ["RotatingFileHandler", "StreamHandler"]


def test_console_level_info_by_default(tmp_path, restore_root_logger):
    configure_logging(verbose=False, log_dir=tmp_path)
    console = [h for h in _managed_handlers()
               if type(h).__name__ == "StreamHandler"][0]
    assert console.level == logging.INFO


def test_console_level_debug_when_verbose(tmp_path, restore_root_logger):
    configure_logging(verbose=True, log_dir=tmp_path)
    console = [h for h in _managed_handlers()
               if type(h).__name__ == "StreamHandler"][0]
    assert console.level == logging.DEBUG


def test_file_handler_always_debug(tmp_path, restore_root_logger):
    # Even with a quiet console, the file captures DEBUG for the full record.
    configure_logging(verbose=False, log_dir=tmp_path)
    fh = [h for h in _managed_handlers()
          if isinstance(h, RotatingFileHandler)][0]
    assert fh.level == logging.DEBUG


def test_file_captures_debug_not_gated_by_console(tmp_path, restore_root_logger):
    path = configure_logging(verbose=False, log_dir=tmp_path)
    logging.getLogger("t.debug").debug("DEBUG_MARKER_42")
    for h in _managed_handlers():
        h.flush()
    assert "DEBUG_MARKER_42" in path.read_text(encoding="utf-8")


def test_rotation_config(tmp_path, restore_root_logger):
    configure_logging(log_dir=tmp_path)
    fh = [h for h in _managed_handlers()
          if isinstance(h, RotatingFileHandler)][0]
    assert fh.maxBytes == _MAX_BYTES
    assert fh.backupCount == _BACKUP_COUNT
    assert _MAX_BYTES > 0 and _BACKUP_COUNT > 0


def test_idempotent_no_handler_stacking(tmp_path, restore_root_logger):
    configure_logging(log_dir=tmp_path)
    assert len(_managed_handlers()) == 2
    configure_logging(log_dir=tmp_path)
    # Second call removes the prior managed pair before re-adding.
    assert len(_managed_handlers()) == 2


def test_symbol_logger_injects_tag(tmp_path, restore_root_logger):
    path = configure_logging(log_dir=tmp_path)
    base = logging.getLogger("t.symbol")
    get_symbol_logger(base, "MSFT").info("tagged message")
    for h in _managed_handlers():
        h.flush()
    content = path.read_text(encoding="utf-8")
    assert "[MSFT]" in content
    assert "tagged message" in content


def test_plain_record_gets_default_symbol(tmp_path, restore_root_logger):
    # A module-level logger with no extra={"symbol":...} must not raise on the
    # symbol-bearing file format; the filter backfills "-".
    path = configure_logging(log_dir=tmp_path)
    logging.getLogger("t.plain").warning("no symbol here")
    for h in _managed_handlers():
        h.flush()
    content = path.read_text(encoding="utf-8")
    assert "[-]" in content
    assert "no symbol here" in content


def test_symbol_default_filter_unit():
    f = _SymbolDefaultFilter()
    rec = logging.LogRecord("n", logging.INFO, "p", 1, "msg", None, None)
    assert not hasattr(rec, "symbol")
    assert f.filter(rec) is True
    assert rec.symbol == "-"
    # An already-tagged record is left untouched.
    rec2 = logging.LogRecord("n", logging.INFO, "p", 1, "msg", None, None)
    rec2.symbol = "GOOG"
    assert f.filter(rec2) is True
    assert rec2.symbol == "GOOG"
