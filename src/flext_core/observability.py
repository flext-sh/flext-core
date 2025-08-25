"""Basic observability implementations for development and testing.

SINGLE CONSOLIDATED MODULE following FLEXT architectural patterns.
All observability functionality consolidated into FlextObservability.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager

from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult

# Type aliases for unified approach with FlextProtocols integration - Python 3.13+ syntax
type ObservabilityProtocol = FlextProtocols.Infrastructure.Configurable
type MetricsCollectorProtocol = FlextProtocols.Extensions.Observability
type LoggerServiceProtocol = FlextProtocols.Infrastructure.LoggerProtocol

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

    class Metrics:
        """Nested in-memory metrics collector."""

        def __init__(self) -> None:
            self._counters: dict[str, int] = {}
            self._gauges: dict[str, float] = {}

        def increment_counter(
            self, name: str, tags: dict[str, str] | None = None
        ) -> None:
            key = self._make_key(name, tags)
            self._counters[key] = self._counters.get(key, 0) + 1

        def record_gauge(
            self, name: str, value: float, tags: dict[str, str] | None = None
        ) -> None:
            key = self._make_key(name, tags)
            self._gauges[key] = value

        def _make_key(self, name: str, tags: dict[str, str] | None) -> str:
            if not tags:
                return name
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}[{tag_str}]"

    class Observability:
        """Nested simple observability implementation."""

        def __init__(self) -> None:
            self.logger = FlextObservability.Console()
            self.tracer = FlextObservability.Tracer()
            self.metrics = FlextObservability.Metrics()

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


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Consolidated approach
# =============================================================================

# Export nested classes for external access (backward compatibility)
FlextConsole = FlextObservability.Console
FlextSpan = FlextObservability.Span
FlextTracer = FlextObservability.Tracer
FlextMetrics = FlextObservability.Metrics
FlextObservabilitySystem = FlextObservability  # Backward compatibility with old name
FlextAlerts = Alerts()
FlextCoreObservability = FlextObservability.Observability
_SimpleHealth = Alerts()  # Simple fallback


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


def get_global_observability() -> FlextObservability.Observability:
    """Get global observability instance for compatibility."""
    return _GlobalObservabilityManager.get_instance()


def reset_global_observability() -> None:
    """Reset global observability instance for compatibility."""
    _GlobalObservabilityManager.reset_instance()


__all__: list[str] = [
    "FlextAlerts",
    "FlextConsole",
    "FlextCoreObservability",
    "FlextMetrics",
    "FlextObservability",
    "FlextSpan",
    "FlextTracer",
    "get_global_observability",
    "reset_global_observability",
]
