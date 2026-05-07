"""
Logging utilities.

Two destinations:
  1. stderr / file via standard logging (server-side debugging).
  2. an in-memory ring buffer that the Streamlit UI reads to show a live log.

The ring buffer captures structured records (level, agent, message, time,
optional payload like code or result snippet) so the UI can render them
nicely instead of just dumping text.

Usage:
    from .logging_util import get_logger, ring_buffer
    log = get_logger("data_insight")
    log.info("built profile", extra={"payload": profile_str})
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LogRecord:
    ts: float
    level: str
    agent: str
    message: str
    payload: Optional[str] = None  # (code, result snippet)
    duration_ms: Optional[float] = None


class RingBufferHandler(logging.Handler):
    """A logging.Handler that keeps the last N records in memory."""
    def __init__(self, capacity: int = 500):
        super().__init__()
        self._buf: deque[LogRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = getattr(record, "payload", None)
            duration_ms = getattr(record, "duration_ms", None)
            agent = getattr(record, "agent", record.name)
            entry = LogRecord(
                ts=record.created,
                level=record.levelname,
                agent=agent,
                message=record.getMessage(),
                payload=payload,
                duration_ms=duration_ms,
            )
            with self._lock:
                self._buf.append(entry)
        except Exception:
            # Never let logging crash the app
            pass

    def snapshot(self) -> list[LogRecord]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


# Module-level singleton - the UI reads from this.
ring_buffer = RingBufferHandler(capacity=1000)

_configured = False

def configure_logging(level: int = logging.INFO) -> None:
    """Idempotently configure root logging once."""
    global _configured
    if _configured:
        return
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    ring_buffer.setLevel(level)

    root = logging.getLogger("data_analyst_agent")
    root.setLevel(level)
    # Avoid double-adding on hot reloads
    if not any(isinstance(h, RingBufferHandler) for h in root.handlers):
        root.addHandler(stderr_handler)
        root.addHandler(ring_buffer)
    root.propagate = False
    _configured = True


def get_logger(agent: str) -> logging.LoggerAdapter:
    """Return a logger adapter that injects `agent=...` into every record."""
    configure_logging()
    base = logging.getLogger(f"data_analyst_agent.{agent}")
    return logging.LoggerAdapter(base, extra={"agent": agent})


# Small context manager to time a block and log the duration in ms.
class timed:
    def __init__(self, log: logging.LoggerAdapter, msg: str):
        self.log = log
        self.msg = msg

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt_ms = (time.perf_counter() - self._t0) * 1000
        if exc is None:
            self.log.info(self.msg, extra={"duration_ms": dt_ms})
        else:
            self.log.error(f"{self.msg} FAILED: {exc}",
                           extra={"duration_ms": dt_ms})