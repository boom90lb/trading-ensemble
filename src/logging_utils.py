"""Structured logging setup for the trading-ensemble scripts (Phase 5.1).

Replaces the per-script ``logging.basicConfig`` calls with a single
``configure_logging`` entry point that installs:

  * a **console** handler at INFO (DEBUG when ``--verbose``), and
  * a **file** handler that always captures DEBUG to ``logs/run_{ts}.log``,
    wrapped in a ``RotatingFileHandler`` so a runaway RL/sweep run cannot
    fill the disk (size cap + a few backups), independent of the console
    verbosity.

Records can carry a per-symbol tag so the file is greppable by ticker and
MLflow-sliceable: use :func:`get_symbol_logger` to obtain a
``LoggerAdapter`` that injects ``extra={"symbol": ...}`` on every call. The
format string renders that tag; a filter supplies a ``"-"`` default so the
many module-level loggers that don't know a symbol still format cleanly.
"""

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

from src.config import LOGS_DIR

# One log file per run; the timestamp gives natural per-run separation. The
# RotatingFileHandler size cap is a safety net against a single pathological
# run (e.g. a long PPO fit logging per-step) rather than a per-day rotation.
_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_BACKUP_COUNT = 5

# Console stays terse; the file carries the symbol tag for post-hoc slicing.
_CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(symbol)s] - %(message)s"

# Third-party loggers that are noisy at INFO and never carry signal we want.
_QUIET_LIBRARIES = ("prophet", "cmdstanpy", "matplotlib", "urllib3")


class _SymbolDefaultFilter(logging.Filter):
    """Ensure every record has a ``symbol`` attribute.

    Module-level loggers emit records without ``extra={"symbol": ...}``; the
    file format references ``%(symbol)s`` so those records would otherwise
    raise ``KeyError`` at format time. This backfills ``"-"`` for them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "symbol"):
            record.symbol = "-"
        return True


def configure_logging(
    verbose: bool = False,
    run_name: Optional[str] = None,
    log_dir: Union[str, Path] = LOGS_DIR,
) -> Path:
    """Install console + rotating-file logging on the root logger.

    Idempotent w.r.t. handler duplication: any handlers we previously
    installed (tagged via ``_te_managed``) are removed first, so repeated
    calls in the same process (e.g. tests, notebooks) don't stack handlers.

    Args:
        verbose: console level DEBUG when True, else INFO. The file handler
            always records DEBUG regardless, so ``--verbose`` only changes
            what scrolls past on the terminal, never what's persisted.
        run_name: optional slug embedded in the log filename
            (``run_{run_name}_{ts}.log``); falls back to ``run_{ts}.log``.
        log_dir: directory for the log file (created if missing).

    Returns:
        The path to the run's log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"run_{run_name}_{ts}" if run_name else f"run_{ts}"
    log_path = log_dir / f"{stem}.log"

    root = logging.getLogger()
    # Root passes everything through; handlers gate by their own level.
    root.setLevel(logging.DEBUG)

    # Drop handlers we installed on a prior call (don't touch foreign ones).
    for handler in list(root.handlers):
        if getattr(handler, "_te_managed", False):
            root.removeHandler(handler)
            handler.close()

    symbol_filter = _SymbolDefaultFilter()

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    console.addFilter(symbol_filter)
    console._te_managed = True  # type: ignore[attr-defined]
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    file_handler.addFilter(symbol_filter)
    file_handler._te_managed = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    for lib in _QUIET_LIBRARIES:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging to %s", log_path)
    return log_path


def get_symbol_logger(
    logger: logging.Logger, symbol: str
) -> logging.LoggerAdapter:
    """Wrap ``logger`` so every record carries ``extra={"symbol": symbol}``.

    The returned adapter formats with the symbol tag in the file log and is
    sliceable downstream (e.g. by MLflow run name == symbol).
    """
    return logging.LoggerAdapter(logger, {"symbol": symbol})
