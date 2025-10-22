"""
Observability framework for distributed tracing, metrics, and logging.

Implements PRIME_DIRECTIVE section 4 (observability and auditability requirements).
Every service MUST emit traces, metrics, and structured logs.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from pydantic import BaseModel, Field

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
)


class SpanContext(BaseModel):
    """Trace span context."""

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None


class Span:
    """Distributed tracing span.

    Represents a single unit of work in a distributed trace.
    Compatible with OpenTelemetry span model.
    """

    def __init__(
        self,
        operation_name: str,
        context: SpanContext,
        service_name: str,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize span.

        Args:
            operation_name: Operation being traced
            context: Span context (trace ID, span ID, parent)
            service_name: Service emitting the span
            tags: Optional tags/attributes
        """
        self.operation_name = operation_name
        self.context = context
        self.service_name = service_name
        self.tags = tags or {}
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.status = "ok"
        self.logs: list[Dict[str, Any]] = []

    def set_tag(self, key: str, value: Any) -> None:
        """Set a span tag/attribute."""
        self.tags[key] = value

    def log_event(self, event: str, **fields: Any) -> None:
        """Log an event within the span."""
        self.logs.append(
            {
                "timestamp": time.time(),
                "event": event,
                **fields,
            }
        )

    def set_status(self, status: str) -> None:
        """Set span status ('ok' or 'error')."""
        self.status = status

    def finish(self) -> None:
        """Mark span as finished."""
        self.end_time = time.time()

    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms() if self.end_time else None,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs,
        }


class ServiceObservability:
    """Observability manager for a service.

    Provides unified interface for tracing, metrics, and structured logging.
    """

    def __init__(self, service_name: str, version: str) -> None:
        """Initialize observability.

        Args:
            service_name: Service name
            version: Service version
        """
        self.service_name = service_name
        self.version = version
        self.logger = logging.getLogger(service_name)
        self._active_spans: Dict[str, Span] = {}
        self._metrics: Dict[str, list[float]] = {}

    @contextmanager
    def trace_operation(
        self,
        operation: str,
        parent_context: Optional[SpanContext] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Trace an operation with a span.

        Args:
            operation: Operation name
            parent_context: Optional parent span context
            tags: Optional span tags

        Yields:
            Span object for the operation

        Example:
            with observability.trace_operation("process_meeting") as span:
                span.set_tag("meeting_id", meeting_id)
                # ... do work ...
                span.log_event("summary_generated")
        """
        # Create span context
        context = SpanContext(
            trace_id=parent_context.trace_id if parent_context else str(uuid.uuid4()),
            parent_span_id=parent_context.span_id if parent_context else None,
        )

        # Create and start span
        span = Span(
            operation_name=operation,
            context=context,
            service_name=self.service_name,
            tags=tags or {},
        )
        span.set_tag("service.version", self.version)

        self._active_spans[span.context.span_id] = span

        try:
            self.logger.debug(
                "Started span: %s (trace_id=%s, span_id=%s)",
                operation,
                context.trace_id,
                context.span_id,
            )
            yield span
            span.set_status("ok")
        except Exception as exc:
            span.set_status("error")
            span.set_tag("error", True)
            span.set_tag("error.message", str(exc))
            span.log_event("exception", exception=str(exc))
            raise
        finally:
            span.finish()
            self._active_spans.pop(span.context.span_id, None)
            
            # Export span (in production, send to OTLP collector)
            self._export_span(span)

    def _export_span(self, span: Span) -> None:
        """Export completed span.

        In production, this would send to an OpenTelemetry collector.
        For now, just log it.
        """
        self.logger.debug(
            "Span completed: %s duration=%.2fms status=%s",
            span.operation_name,
            span.duration_ms(),
            span.status,
            extra={"span_data": span.to_dict()},
        )

    def emit_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Emit a metric value.

        Args:
            name: Metric name (e.g., "agent.invocation.duration_ms")
            value: Metric value
            tags: Optional metric tags

        Example:
            observability.emit_metric("agent.invocation.duration_ms", 1234.5, {"agent": "summarizer"})
        """
        metric_key = f"{self.service_name}.{name}"
        
        if metric_key not in self._metrics:
            self._metrics[metric_key] = []
        
        self._metrics[metric_key].append(value)

        self.logger.debug(
            "Metric: %s=%s tags=%s",
            name,
            value,
            tags or {},
            extra={"metric_name": name, "metric_value": value, "metric_tags": tags},
        )

    def log_event(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log a structured event.

        Args:
            level: Log level ('debug', 'info', 'warning', 'error')
            message: Log message
            context: Additional context dictionary
            correlation_id: Optional correlation ID for related events

        Example:
            observability.log_event("info", "Meeting summarized", 
                                   context={"meeting_id": "123"}, 
                                   correlation_id=trace_id)
        """
        log_data = {
            "service": self.service_name,
            "version": self.version,
            "message": message,
            "correlation_id": correlation_id,
            **(context or {}),
        }

        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, extra=log_data)

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for collected metrics.

        Returns:
            Dictionary of metric names to statistics (count, sum, avg, min, max)
        """
        summary = {}
        
        for metric_name, values in self._metrics.items():
            if not values:
                continue
            
            summary[metric_name] = {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        
        return summary

    def reset_metrics(self) -> None:
        """Reset collected metrics."""
        self._metrics.clear()


# Global observability instance (can be initialized per service)
_observability_instance: Optional[ServiceObservability] = None


def init_observability(service_name: str, version: str) -> ServiceObservability:
    """Initialize global observability instance.

    Args:
        service_name: Service name
        version: Service version

    Returns:
        ServiceObservability instance
    """
    global _observability_instance
    _observability_instance = ServiceObservability(service_name, version)
    return _observability_instance


def get_observability() -> ServiceObservability:
    """Get global observability instance.

    Returns:
        ServiceObservability instance

    Raises:
        RuntimeError: If observability not initialized
    """
    if _observability_instance is None:
        raise RuntimeError("Observability not initialized. Call init_observability() first.")
    return _observability_instance


# Convenience functions


def trace_operation(
    operation: str,
    parent_context: Optional[SpanContext] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> Generator[Span, None, None]:
    """Trace an operation (convenience wrapper).

    Args:
        operation: Operation name
        parent_context: Optional parent span context
        tags: Optional span tags

    Yields:
        Span object
    """
    obs = get_observability()
    return obs.trace_operation(operation, parent_context, tags)


def emit_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Emit a metric (convenience wrapper).

    Args:
        name: Metric name
        value: Metric value
        tags: Optional tags
    """
    obs = get_observability()
    obs.emit_metric(name, value, tags)


def log_event(
    level: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Log an event (convenience wrapper).

    Args:
        level: Log level
        message: Message
        context: Optional context
        correlation_id: Optional correlation ID
    """
    obs = get_observability()
    obs.log_event(level, message, context, correlation_id)

