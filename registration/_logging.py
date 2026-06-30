"""Central logging setup. Import get_logger() everywhere instead of print()."""
import logging
import os
import sys

_CONFIGURED = False


def _configure():
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("ATTEND_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger("attendance")
    root.setLevel(level)
    root.handlers = [handler]
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"attendance.{name}")
