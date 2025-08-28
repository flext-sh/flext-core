"""Basic observability implementations for development and testing.

SINGLE CONSOLIDATED MODULE following FLEXT architectural patterns.
All observability functionality consolidated into FlextObservability.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager

from flext_core.result import FlextResult

GeneratorT = Generator


class FlextObservability:
    """SINGLE CONSOLIDATED CLASS for all observability functionality.

    Following FLEXT architectural patterns - consolidates ALL observability functionality
    including console logging, tracing, metrics, alerts, and health checks into one main class
    with nested classes for organization.

    CONSOLIDATED CLASSES: FlextConsole + FlextSpan + FlextTracer + FlextMetrics + FlextObservability + FlextAlerts + _SimpleHealth
    """

    # ==========================================================================
    # NESTED CLASSES FOR ORGANIZATION
    # ==========================================================================

    class Console:
        """Nested console-based logger implementing FlextLoggerProtocol."""

        def __init__(self, name: str = "flext-console") -> None:
            self._logger = logging.getLogger(name)
            self.name = name

        def trace(self, message: str, **kwargs: object) -> None:
            self._logger.debug(
                "TRACE: %s %s", message, json.dumps(kwargs) if kwargs else ""
            )

        def debug(self, message: str, **kwargs: object) -> None:
            self._logger.debug(message, extra={"context": kwargs} if kwargs else None)

        def info(self, message: str, **kwargs: object) -> None:
            self._logger.info(message, extra={"context": kwargs} if kwargs else None)

        def warning(self, message: str, **kwargs: object) -> None:
            self._logger.warning(message, extra={"context": kwargs} if kwargs else None)

        def warn(self, message: str, **kwargs: object) -> None:
            self.warning(message, **kwargs)

        def error(self, message: str, **kwargs: object) -> None:
            self._logger.error(message, extra={"context": kwargs} if kwargs else None)

        def critical(self, message: str, **kwargs: object) -> None:
            self._logger.critical(
                message, extra={"context": kwargs} if kwargs else None
            )

        def fatal(self, message: str, **kwargs: object) -> None:
            self.critical(message, **kwargs)

        def exception(
            self, message: str, *, exc_info: bool = True, **kwargs: object
        ) -> None:
            """Log exception with optional exc_info."""
            # Use exc_info parameter for proper exception logging
            self._logger.error(
                message,
                exc_info=exc_info,
                extra={"context": kwargs} if kwargs else None,
            )

        def audit(self, message: str, **kwargs: object) -> None:
            self._logger.info(
                "AUDIT: %s %s", message, json.dumps(kwargs) if kwargs else ""
            )

    class Span:
        """Nested no-operation span."""

        def set_tag(self, key: str, value: str) -> None:
            pass

        def log_event(self, event_name: str, payload: dict[str, object]) -> None:
            pass

        def add_context(self, key: str, value: object) -> None:
            """Add context to span for API compatibility with tests."""

        def add_error(self, error: Exception) -> None:
            """Add error to span for API compatibility with tests."""

        def finish(self) -> None:
            pass

    class Tracer:
        """Nested no-operation tracer."""

        @contextmanager
        def trace_operation(
            self, operation_name: str
        ) -> Generator[FlextObservability.Span]:
            """Trace operation with given name."""
            span = FlextObservability.Span()
            # Use operation_name for span identification
            span.set_tag("operation", operation_name)
            try:
                yield span
            finally:
                span.finish()

        @contextmanager
        def business_span(
            self, operation_name: str
        ) -> Generator[FlextObservability.Span]:
            """Business span for API compatibility with tests."""
            # Delegate to trace_operation for consistent behavior
            with self.trace_operation(operation_name) as span:
                yield span

        @contextmanager
        def technical_span(
            self, operation_name: str, component: str | None = None
        ) -> Generator[FlextObservability.Span]:
            """Technical span for API compatibility with tests."""
            with self.trace_operation(operation_name) as span:
                if component:
                    span.set_tag("component", component)
                yield span

        @contextmanager
        def error_span(
            self, operation_name: str, error_type: str | None = None
        ) -> Generator[FlextObservability.Span]:
            """Error span for API compatibility with tests."""
            with self.trace_operation(operation_name) as span:
                if error_type:
                    span.set_tag("error_type", error_type)
                yield span

    class Metrics:
        """Nested in-memory metrics collector."""

        def __init__(self) -> None:
            self._counters: dict[str, int] = {}
            self._gauges: dict[str, float] = {}
            self._histograms: dict[str, list[float]] = {}

        def increment_counter(
            self, name: str, tags: dict[str, str] | None = None
        ) -> None:
            key = self._make_key(name, tags)
            self._counters[key] = self._counters.get(key, 0) + 1

        def increment(
            self, name: str, value: int = 1, tags: dict[str, str] | None = None
        ) -> None:
            """Increment counter for API compatibility with tests."""
            if tags:
                key = self._make_key(name, tags)
                self._counters[key] = self._counters.get(key, 0) + value
            else:
                self._counters[name] = self._counters.get(name, 0) + value

        def record_gauge(
            self, name: str, value: float, tags: dict[str, str] | None = None
        ) -> None:
            key = self._make_key(name, tags)
            self._gauges[key] = value

        def gauge(self, name: str, value: float) -> None:
            """Record gauge for API compatibility with tests."""
            self._gauges[name] = value

        def histogram(self, name: str, value: float) -> None:
            """Record histogram value for API compatibility with tests."""
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)

        def get_metrics(self) -> dict[str, object]:
            """Get all collected metrics for API compatibility with tests."""
            return {
                "counters": self._counters.copy(),
                "gauges": self._gauges.copy(),
                "histograms": {k: v.copy() for k, v in self._histograms.items()},
            }

        def clear_metrics(self) -> None:
            """Clear all collected metrics for API compatibility with tests."""
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

        def _make_key(self, name: str, tags: dict[str, str] | None) -> str:
            if not tags:
                return name
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}[{tag_str}]"

    class Observability:
        """Nested simple observability implementation."""

        def __init__(self) -> None:
            # Standard attributes (legacy)
            self.logger = FlextObservability.Console()
            self.tracer = FlextObservability.Tracer()
            self.metrics = FlextObservability.Metrics()

            # Test API compatibility attributes
            self.log = self.logger  # Tests expect .log
            self.trace = self.tracer  # Tests expect .trace
            self.alerts = FlextObservability.Alerts()
            self.health = FlextObservability.Health()

        def record_metric(
            self, name: str, value: float, tags: dict[str, str] | None = None
        ) -> FlextResult[None]:
            try:
                self.metrics.record_gauge(name, value, tags)
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to record metric: {e}")

    class Alerts:
        """Nested minimal alerts component."""

        def info(self, message: str, **kwargs: object) -> None:
            pass

        def warning(self, message: str, **kwargs: object) -> None:
            pass

        def critical(self, message: str, **kwargs: object) -> None:
            """Critical alert for API compatibility with tests."""

        def error(self, message: str, **kwargs: object) -> None:
            """Error alert for API compatibility with tests."""

    class Health:
        """Nested health check component."""

        def __init__(self) -> None:
            self._status = "healthy"

        def check(self) -> dict[str, object]:
            """Health check for API compatibility with tests."""
            return {"status": self._status, "timestamp": "now"}

        def is_healthy(self) -> bool:
            """Check if system is healthy."""
            return self._status == "healthy"


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Consolidated approach
# =============================================================================

# Export nested classes for external access (backward compatibility)
FlextConsole = FlextObservability.Console
FlextSpan = FlextObservability.Span
FlextTracer = FlextObservability.Tracer
FlextMetrics = FlextObservability.Metrics
FlextObservabilitySystem = FlextObservability  # Backward compatibility with old name
FlextAlerts = FlextObservability.Alerts
FlextCoreObservability = FlextObservability.Observability
_SimpleHealth = FlextObservability.Alerts()  # Simple fallback


# Global instance for compatibility - using singleton pattern
class _GlobalObservabilityManager:
    """Singleton manager for global observability instance."""

    _instance: FlextObservability.Observability | None = None

    @classmethod
    def get_instance(cls) -> FlextObservability.Observability:
        """Get or create global observability instance."""
        if cls._instance is None:
            cls._instance = FlextObservability.Observability()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset global observability instance."""
        cls._instance = FlextObservability.Observability()


def get_global_observability(
    log_level: str | None = None, *, force_recreate: bool = False
) -> FlextObservability.Observability:
    """Get global observability instance for compatibility."""
    # log_level parameter is accepted but not used in this no-op implementation
    _ = log_level  # Explicitly mark as unused but needed for API compatibility
    if force_recreate:
        _GlobalObservabilityManager.reset_instance()
    return _GlobalObservabilityManager.get_instance()


def reset_global_observability() -> None:
    """Reset global observability instance for compatibility."""
    _GlobalObservabilityManager.reset_instance()


__all__: list[str] = [
    "FlextObservability",  # ONLY main class exported
]
