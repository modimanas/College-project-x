"""Logging utilities for the detection module.

What this module does
---------------------
- Provides a consistent JSON-ish log format with timestamps.
- Keeps loggers namespaced under ``attendance.<component>``.

Why it exists
-------------
The detection service will run long-lived background workers (RTSP readers,
trackers, matchers). Clear logs are the main production debugging surface.
"""

from __future__ import annotations

import logging
import os


def get_logger(component: str) -> logging.Logger:
    """Return a configured logger for a given detection component.

    Args:
        component: Short name like ``api`` or ``worker``.

    Returns:
        A standard library ``logging.Logger``.
    """

    name = f"attendance.det.{component}"
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = os.getenv("ATTEND_LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)

    handler = logging.StreamHandler()
    fmt = (
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
    )
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

