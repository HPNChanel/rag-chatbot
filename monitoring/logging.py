"""Structured logging utilities for deployment."""

from __future__ import annotations

import json
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Emit log records as compact JSON."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.__dict__.get("extra_data"):
            payload.update(record.__dict__["extra_data"])
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_dir: Optional[Path] = None) -> Logger:
    """Configure structured logging and return the application logger."""

    logger = logging.getLogger("ragx")
    logger.setLevel(logging.INFO)
    handler: logging.Handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "ragx.log"
        handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.propagate = False
    return logger


__all__ = ["configure_logging", "JsonFormatter"]

