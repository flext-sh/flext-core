"""Basic observability implementations for development and testing."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager

from flext_core.protocols import (
    FlextLoggerProtocol,
    FlextMetricsProtocol,
    FlextSpanProtocol,
    FlextTracerProtocol,
)
from flext_core.result import FlextResult

GeneratorT = Generator

# =============================================================================
# FOUNDATION IMPLEMENTATIONS - Minimal implementations for development/testing
# =============================================================================


class FlextConsoleLogger:
    """Simple console-based logger implementing FlextLoggerProtocol.

    IMPLEMENTS: FlextLoggerProtocol from protocols.py
    Provides minimal logging functionality for development and testing.
    """

    def __init__(self, name: str = "flext-console") -> None:
        """Initialize console logger."""
        # Expose underlying stdlib logger on `_logger`
        self._logger = logging.getLogger(name)
        self.name = name

    def trace(self, message: str, **kwargs: object) -> None:
        """Log trace message to console."""
        self._logger.debug(
            "TRACE: %s %s",
            message,
            json.dumps(kwargs) if kwargs else "",
        )

    def debug(self, message: str, **kwargs: object) -> None:
        """Log debug message to console."""
        self._logger.debug(message, extra={"context": kwargs} if kwargs else None)

    def info(self, message: str, **kwargs: object) -> None:
        """Log info message to console."""
        self._logger.info(message, extra={"context": kwargs} if kwargs else None)

    def warning(self, message: str, **kwargs: object) -> None:
        """Log a warning message to console."""
        self._logger.warning(message, extra={"context": kwargs} if kwargs else None)

    def warn(self, message: str, **kwargs: object) -> None:
        """Alias for warning."""
        self.warning(message, **kwargs)

    def error(self, message: str, **kwargs: object) -> None:
        """Log error message to console."""
        self._logger.error(message, extra={"context": kwargs} if kwargs else None)

    def critical(self, message: str, **kwargs: object) -> None:
        """Log critical message to console."""
        self._logger.critical(message, extra={"context": kwargs} if kwargs else None)

    def fatal(self, message: str, **kwargs: object) -> None:
        """Alias for critical."""
        self.critical(message, **kwargs)

    def exception(
        self,
        message: str,
        *,
        exc_info: bool = True,
        **kwargs: object,
    ) -> None:
        """Log exception message to console with automatic traceback information."""
        if exc_info:
            self._logger.error(message, extra={"context": kwargs} if kwargs else None)
        else:
            self._logger.error(message, extra={"context": kwargs} if kwargs else None)

    def audit(self, message: str, **kwargs: object) -> None:
        """Audit log (no-op implementation)."""
        self._logger.info("AUDIT: %s %s", message, json.dumps(kwargs) if kwargs else "")


class FlextNoOpSpan:
    """No-operation span implementing FlextSpanProtocol."""

    def set_tag(self, key: str, value: str) -> None:
        """No-op set tag."""

    def log_event(self, event_name: str, payload: dict[str, object]) -> None:
        """No-op log event."""

    def finish(self) -> None:
        """No-op finish span."""

    def add_context(self, key: str, value: object) -> None:
        """Alias for set_tag."""
        del key, value

    def add_error(self, error: Exception) -> None:
        """Record error on span (no-op implementation)."""
        del error


class FlextNoOpTracer:
    """No-operation tracer implementing FlextTracerProtocol."""

    def start_span(self, operation_name: str) -> FlextSpanProtocol:
        """Start no-op span."""
        del operation_name  # Unused argument
        return FlextNoOpSpan()

    def inject_context(self, headers: dict[str, str]) -> None:
        """No-op inject context."""

    @contextmanager
    def trace_operation(self, operation_name: str) -> Generator[FlextSpanProtocol]:
        """Trace operation with no-op span."""
        span = self.start_span(operation_name)
        try:
            yield span
        finally:
            span.finish()

    @contextmanager
    def business_span(self, operation: str, **kwargs: object) -> Generator[object]:
        """Create business span context manager."""
        del kwargs
        with self.trace_operation(operation) as span:
            yield span

    @contextmanager
    def technical_span(self, operation: str, **kwargs: object) -> Generator[object]:
        """Technical span context manager."""
        del kwargs
        with self.trace_operation(operation) as span:
            yield span

    @contextmanager
    def error_span(self, operation: str, **kwargs: object) -> Generator[object]:
        """Error span context manager."""
        del kwargs
        with self.trace_operation(operation) as span:
            yield span


class FlextInMemoryMetrics:
    """In-memory metrics collector implementing FlextMetricsProtocol."""

    def __init__(self) -> None:
        """Initialize in-memory metrics storage."""
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}

    def increment_counter(self, name: str, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, tags)
        self._counters[key] = self._counters.get(key, 0) + 1

    def record_gauge(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record gauge metric."""
        key = self._make_key(name, tags)
        self._gauges[key] = value

    def record_histogram(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record histogram metric."""
        key = self._make_key(name, tags)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

    def get_counter(self, name: str, tags: dict[str, str] | None = None) -> int:
        """Get counter value."""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, tags: dict[str, str] | None = None) -> float | None:
        """Get gauge value."""
        key = self._make_key(name, tags)
        return self._gauges.get(key)

    def get_histogram(
        self,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> list[float]:
        """Get histogram values."""
        key = self._make_key(name, tags)
        return self._histograms.get(key, [])

    # Simple metric API
    def increment(
        self,
        name: str,
        value: int = 1,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Increment counter by value."""
        key = self._make_key(name, tags)
        self._counters[key] = self._counters.get(key, 0) + int(value)

    def gauge(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record gauge value."""
        self.record_gauge(name, value, tags)

    def histogram(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record histogram value."""
        self.record_histogram(name, value, tags)

    def get_metrics(self) -> dict[str, object]:
        """Get all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": dict(self._histograms),
        }

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()

    def _make_key(self, name: str, tags: dict[str, str] | None) -> str:
        """Create key from name and tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"


class FlextSimpleObservability:
    """Simple observability implementation implementing ObservabilityProtocol."""

    def __init__(
        self,
        logger: FlextLoggerProtocol | None = None,
        tracer: FlextTracerProtocol | None = None,
        metrics: FlextMetricsProtocol | None = None,
    ) -> None:
        """Initialize with optional components."""
        self.logger = logger or FlextConsoleLogger()
        self.tracer = tracer or FlextNoOpTracer()
        self.metrics = metrics or FlextInMemoryMetrics()

    def record_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> FlextResult[None]:
        """Record metric value."""
        try:
            self.metrics.record_gauge(name, value, tags)
            return FlextResult.ok(None)
        except Exception as e:
            return FlextResult.fail(f"Failed to record metric: {e}")

    def start_trace(self, operation_name: str) -> FlextResult[str]:
        """Start distributed trace."""
        try:
            self.tracer.start_span(operation_name)
            # Generate trace ID (simplified for foundation implementation)
            trace_id = f"trace_{hash(operation_name)}"
            return FlextResult.ok(trace_id)
        except Exception as e:
            return FlextResult.fail(f"Failed to start trace: {e}")

    def health_check(self) -> FlextResult[dict[str, object]]:
        """Perform health check."""
        return FlextResult.ok(
            {
                "status": "healthy",
                "logger": "available",
                "tracer": "available",
                "metrics": "available",
                "implementation": "simple_observability",
            },
        )


# =============================================================================
# ADDITIONAL COMPONENTS FOR TESTING
# =============================================================================


class FlextSimpleAlerts:
    """Minimal alerts component for testing environments."""

    def info(self, message: str, **kwargs: object) -> None:
        """Record info alert."""
        del message, kwargs

    def warning(self, message: str, **kwargs: object) -> None:
        """Record warning alert."""
        del message, kwargs

    @staticmethod
    def error(message: str, **kwargs: object) -> None:
        """Record error alert."""
        del message, kwargs

    @staticmethod
    def critical(message: str, **kwargs: object) -> None:
        """Record critical alert."""
        del message, kwargs


class _SimpleHealth:
    """Minimal health component for testing environments."""

    @staticmethod
    def health_check() -> dict[str, object]:
        return {"status": "healthy"}

    @staticmethod
    def ready_check() -> bool:
        return True

    @staticmethod
    def live_check() -> bool:
        return True


class FlextMinimalObservability:
    """Composite with testing-friendly attributes and methods.

    Provides `.log`, `.trace`, `.metrics`, `.alerts`, `.health` attributes
    and also implements the core ObservabilityProtocol methods for
    isinstance checks in tests.
    """

    def __init__(self) -> None:
        """Initialize observability instance."""
        self.log = FlextConsoleLogger()
        self.trace = FlextNoOpTracer()
        self.metrics = FlextInMemoryMetrics()
        self.alerts = FlextSimpleAlerts()
        self.health = _SimpleHealth()

    # Protocol compliance helpers
    def record_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> FlextResult[None]:
        """Record metric with gauge value."""
        try:
            self.metrics.gauge(name, value, tags)
            return FlextResult.ok(None)
        except Exception as e:  # pragma: no cover - defensive
            return FlextResult.fail(str(e))

    def start_trace(self, operation_name: str) -> FlextResult[str]:
        """Start trace operation."""
        try:
            _ = self.trace.start_span(operation_name)
            return FlextResult.ok(f"trace_{hash(operation_name)}")
        except Exception as e:  # pragma: no cover - defensive
            return FlextResult.fail(str(e))

    def health_check(self) -> FlextResult[dict[str, object]]:
        """Perform health check."""
        return FlextResult.ok(self.health.health_check())


# =============================================================================
# FACTORY FUNCTIONS - Simple creation patterns
# =============================================================================


def get_console_logger(name: str = "flext-console") -> FlextConsoleLogger:
    """Get console logger instance.

    Args:
      name: Logger name

    Returns:
      FlextConsoleLogger instance

    """
    return FlextConsoleLogger(name)


def get_noop_tracer() -> FlextNoOpTracer:
    """Get no-op tracer instance.

    Returns:
      FlextNoOpTracer instance

    """
    return FlextNoOpTracer()


def get_memory_metrics() -> FlextInMemoryMetrics:
    """Get in-memory metrics instance.

    Returns:
      InMemoryMetrics instance

    """
    return FlextInMemoryMetrics()


def get_simple_observability(
    logger_name: str = "flext-console",
) -> FlextSimpleObservability:
    """Get simple observability implementation.

    Args:
      logger_name: Name for console logger

    Returns:
      SimpleObservability instance with console logger,
      no-op tracer, and memory metrics

    """
    return FlextSimpleObservability(
        logger=get_console_logger(logger_name),
        tracer=get_noop_tracer(),
        metrics=get_memory_metrics(),
    )


# =============================================================================
# GLOBAL INSTANCE - Singleton pattern for simple usage
# =============================================================================


_global_observability: FlextMinimalObservability | None = None


def get_global_observability() -> FlextMinimalObservability:
    """Get global observability instance (singleton).

    Returns:
      Global SimpleObservability instance

    """
    global _global_observability  # noqa: PLW0603
    if _global_observability is None:
        _global_observability = FlextMinimalObservability()
    return _global_observability


def reset_global_observability() -> None:
    """Reset global observability instance (for testing)."""
    global _global_observability  # noqa: PLW0603
    _global_observability = None


def get_observability(
    *,
    log_level: str = "INFO",
    force_recreate: bool = False,
) -> FlextMinimalObservability:
    """Factory returning a process-wide singleton.

    Args:
      log_level: Log level configuration (unused in minimal implementation)
      force_recreate: If True, recreate the global instance

    """
    del log_level
    global _global_observability  # noqa: PLW0603
    if force_recreate or _global_observability is None:
        _global_observability = FlextMinimalObservability()
    return _global_observability


# =============================================================================
# EXPORTS - Foundation implementations only
# =============================================================================

__all__: list[str] = [
    "ConsoleLogger",
    "FlextConsoleLogger",
    "FlextInMemoryMetrics",
    "FlextMinimalObservability",
    "FlextNoOpSpan",
    "FlextNoOpTracer",
    "FlextSimpleAlerts",
    "FlextSimpleObservability",
    "InMemoryMetrics",
    "MinimalObservability",
    "NoOpTracer",
    "SimpleAlerts",
    "get_console_logger",
    "get_global_observability",
    "get_memory_metrics",
    "get_noop_tracer",
    "get_observability",
    "get_simple_observability",
    "reset_global_observability",
]


# =============================================================================
# ALIASES FOR CONVENIENCE
# =============================================================================

# Aliases for testing convenience
ConsoleLogger = FlextConsoleLogger
NoOpTracer = FlextNoOpTracer
InMemoryMetrics = FlextInMemoryMetrics
SimpleAlerts = FlextSimpleAlerts
MinimalObservability = FlextMinimalObservability
