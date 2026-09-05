"""Lightweight white-box tracing for the PolyU RAG pipeline.

The tracer is dependency-free and records a complete nested span tree as JSONL.
It intentionally avoids prompt/answer logging by default; callers can opt in via
capture_inputs/capture_outputs or attach sanitized metadata.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional


def _safe(value: Any, limit: int = 4000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, dict):
        return {str(k): _safe(v, limit) for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple)):
        return [_safe(v, limit) for v in list(value)[:100]]
    return str(value)[:limit]


@dataclass
class TraceSpan:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    started_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    inputs: Any = None
    outputs: Any = None
    status: str = "running"
    error: Optional[str] = None
    ended_at: Optional[float] = None

    def finish(self, status: str = "ok", error: Optional[BaseException] = None) -> None:
        self.ended_at = time.time()
        self.status = status
        if error is not None:
            self.error = f"{type(error).__name__}: {error}"

    @property
    def duration_ms(self) -> Optional[float]:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000.0, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": _safe(self.metadata),
            "inputs": _safe(self.inputs),
            "outputs": _safe(self.outputs),
            "error": self.error,
        }


class WhiteBoxTracer:
    """Record nested RAG spans and persist them to a JSONL sink."""

    def __init__(self, enabled: Optional[bool] = None, sink_path: Optional[str] = None):
        self.enabled = (
            os.getenv("RAG_TRACING_ENABLED", "true").lower() in {"1", "true", "yes", "y"}
            if enabled is None else bool(enabled)
        )
        self.sink_path = sink_path or os.getenv("RAG_TRACE_FILE", "logs/rag_traces.jsonl")
        self.trace_id = uuid.uuid4().hex
        self.spans: list[TraceSpan] = []
        self._stack: list[str] = []

    def _write(self, span: TraceSpan) -> None:
        if not self.enabled or not self.sink_path:
            return
        directory = os.path.dirname(self.sink_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.sink_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")

    @contextmanager
    def span(
        self,
        name: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Any = None,
        capture_inputs: bool = False,
    ) -> Iterator[TraceSpan]:
        span = TraceSpan(
            name=name,
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self._stack[-1] if self._stack else None,
            started_at=time.time(),
            metadata=dict(metadata or {}),
            inputs=inputs if capture_inputs else None,
        )
        self.spans.append(span)
        if not self.enabled:
            yield span
            return
        self._stack.append(span.span_id)
        try:
            yield span
            span.finish("ok")
        except Exception as exc:
            span.finish("error", exc)
            raise
        finally:
            self._stack.pop()
            self._write(span)

    def annotate(self, **metadata: Any) -> None:
        if not self.spans:
            return
        self.spans[-1].metadata.update(metadata)

    def set_output(self, value: Any, *, capture: bool = False) -> None:
        if self.spans and capture:
            self.spans[-1].outputs = value

    def summary(self) -> Dict[str, Any]:
        durations = [s.duration_ms for s in self.spans if s.duration_ms is not None]
        return {
            "trace_id": self.trace_id,
            "span_count": len(self.spans),
            "total_duration_ms": round(sum(durations), 3),
            "stages": [s.to_dict() for s in self.spans],
        }


@contextmanager
def traced_span(tracer: Optional[WhiteBoxTracer], name: str, **metadata: Any):
    """No-op-safe convenience context manager."""
    if tracer is None:
        yield None
        return
    with tracer.span(name, metadata=metadata) as span:
        yield span
