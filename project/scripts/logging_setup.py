"""
Structured logging setup for Brainfast pipelines and the Flask server.

Pattern follows brainreg / fancylog:
- Console: human-readable, INFO level by default
- File: rotating JSON-structured, always DEBUG level (survives terminal close)

Usage in pipeline (scripts/main.py)::

    from scripts.logging_setup import configure_logging
    log = configure_logging(output_dir=paths.outputs, debug=args.debug)
    log.info("Pipeline started")

Usage in sub-modules::

    import logging
    log = logging.getLogger("idlebrain.registration")
    log.info("Conforming warp for slice %d", idx)

All ``idlebrain.*`` child loggers inherit the root idlebrain handler — no
per-module setup required.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_ROOT_LOGGER_NAME = "idlebrain"
_LOG_FILE_NAME = "idlebrain.log"
_MAX_LOG_BYTES = 5_000_000  # 5 MB
_BACKUP_COUNT = 3


def configure_logging(
    output_dir: Path | None = None,
    *,
    debug: bool = False,
    logger_name: str = _ROOT_LOGGER_NAME,
) -> logging.Logger:
    """Configure the idlebrain root logger.

    Safe to call multiple times — extra calls add no extra handlers if the
    logger already has handlers attached.

    Parameters
    ----------
    output_dir : Path | None
        Directory in which to write the rotating log file.  If ``None``,
        file logging is skipped.
    debug : bool
        When ``True`` the console handler emits DEBUG messages.
    logger_name : str
        Root logger name (default ``"idlebrain"``).

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger  # already configured

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)

    # ── Rotating file handler ─────────────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            output_dir / _LOG_FILE_NAME,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                '{"time":"%(asctime)s","level":"%(levelname)s",'
                '"module":"%(name)s","msg":%(message)r}'
            )
        )
        logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the idlebrain namespace.

    Parameters
    ----------
    name : str
        Short module name, e.g. ``"registration"`` → ``"idlebrain.registration"``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
