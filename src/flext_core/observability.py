"""FLEXT Core Observability - Foundation Patterns and Interfaces.

Foundation observability interfaces and patterns for the FLEXT ecosystem.
This module provides the BASE interfaces and minimal implementations that
establish contracts for observability across all FLEXT projects.

**ARCHITECTURE ROLE:**
    Core Foundation Layer → Observability Interfaces → Protocol Definitions

    This module provides ONLY:
    - Abstract interfaces and protocols for observability components
    - Minimal base implementations for development/testing
    - Type definitions and contracts for cross-project compatibility
    - Foundation patterns that flext-observability will implement concretely

**DESIGN PRINCIPLE:**
    flext-core defines WHAT (interfaces/protocols)
    flext-observability implements HOW (concrete implementations)

**Integration Architecture:**
    - flext-core: Defines FlextObservabilityProtocol interfaces
    - flext-observability: Implements production observability stack
    - Other projects: Import interfaces from flext-core, implementations from flext-observability

**Usage Pattern:**
    # Development/Testing (uses minimal implementations from flext-core)
    from flext_core.observability import get_observability
    obs = get_observability()  # Returns minimal implementation

    # Production (uses concrete implementations from flext-observability)
    from flext_observability import get_production_observability
    obs = get_production_observability()  # Returns full implementation

    # Both implement the same FlextObservabilityProtocol interface
    obs.log.info("message")  # Same interface, different implementations

**Foundation Interfaces:**
    - FlextLoggerProtocol: Logging interface contract
    - FlextTracerProtocol: Tracing interface contract
    - FlextMetricsProtocol: Metrics interface contract
    - FlextAlertsProtocol: Alerting interface contract
    - FlextObservabilityProtocol: Complete observability interface

**Minimal Implementations:**
    - ConsoleLogger: Simple console-based logging for development
    - NoOpTracer: No-operation tracing for testing
    - InMemoryMetrics: In-memory metrics for development
    - SimpleAlerts: Basic alerting for development

**Quality Standards:**
    - All protocols must be implementation-agnostic
    - Minimal implementations must be non-blocking
    - Interfaces must support both sync and async patterns
    - Type safety required for all protocol methods

See Also:
    flext-observability: Production observability implementations
    context.py: FlextContext for correlation ID management
    result.py: FlextResult integration patterns

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

# Import FlextContext from dedicated context module
from flext_core.context import FlextContext

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# OBSERVABILITY PROTOCOLS (INTERFACES)
# =============================================================================


@runtime_checkable
class FlextLoggerProtocol(Protocol):
    """Protocol for logging components across FLEXT ecosystem."""

    def trace(self, message: str, **context: object) -> None:
        """Log trace-level information with context."""
        ...

    def debug(self, message: str, **context: object) -> None:
        """Log debug information with context."""
        ...

    def info(self, message: str, **context: object) -> None:
        """Log informational messages with context."""
        ...

    def warn(self, message: str, **context: object) -> None:
        """Log warning messages with context."""
        ...

    def error(
        self,
        message: str,
        *,
        error_code: str | None = None,
        exception: Exception | None = None,
        **context: object,
    ) -> None:
        """Log error messages with full context."""
        ...

    def fatal(self, message: str, **context: object) -> None:
        """Log fatal system errors with context."""
        ...

    def audit(
        self,
        message: str,
        *,
        user_id: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        outcome: str | None = None,
        **context: object,
    ) -> None:
        """Log security and compliance audit events."""
        ...


@runtime_checkable
class FlextSpanProtocol(Protocol):
    """Protocol for tracing span objects."""

    def add_context(self, key: str, value: object) -> None:
        """Add context to the span."""
        ...

    def add_error(self, error: Exception) -> None:
        """Add error information to the span."""
        ...


@runtime_checkable
class FlextTracerProtocol(Protocol):
    """Protocol for distributed tracing components."""

    @contextmanager
    def business_span(
        self, operation_name: str, **context: object
    ) -> Generator[FlextSpanProtocol]:
        """Create span for business operation with context."""
        ...

    @contextmanager
    def technical_span(
        self,
        operation_name: str,
        *,
        component: str | None = None,
        resource: str | None = None,
        **context: object,
    ) -> Generator[FlextSpanProtocol]:
        """Create span for technical operation with context."""
        ...

    @contextmanager
    def error_span(
        self,
        operation_name: str,
        *,
        error_message: str,
        error_code: str | None = None,
        **context: object,
    ) -> Generator[FlextSpanProtocol]:
        """Create span for error context with full error information."""
        ...


@runtime_checkable
class FlextMetricsProtocol(Protocol):
    """Protocol for metrics collection components."""

    def increment(
        self, metric_name: str, value: int = 1, tags: dict[str, str] | None = None
    ) -> None:
        """Increment counter metric with tags."""
        ...

    def histogram(
        self, metric_name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Record histogram value with tags."""
        ...

    def gauge(
        self, metric_name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Set gauge value with tags."""
        ...

    def get_metrics(self) -> dict[str, object]:
        """Get all collected metrics."""
        ...

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        ...


@runtime_checkable
class FlextAlertsProtocol(Protocol):
    """Protocol for alerting components."""

    def info(self, message: str, **context: object) -> None:
        """Send informational alert."""
        ...

    def warning(self, message: str, **context: object) -> None:
        """Send warning alert."""
        ...

    def error(self, message: str, **context: object) -> None:
        """Send error alert."""
        ...

    def critical(self, message: str, **context: object) -> None:
        """Send critical alert."""
        ...


@runtime_checkable
class FlextObservabilityProtocol(Protocol):
    """Complete observability protocol combining all components."""

    @property
    def log(self) -> FlextLoggerProtocol:
        """Access to logging component."""
        ...

    @property
    def trace(self) -> FlextTracerProtocol:
        """Access to tracing component."""
        ...

    @property
    def metrics(self) -> FlextMetricsProtocol:
        """Access to metrics component."""
        ...

    @property
    def alerts(self) -> FlextAlertsProtocol:
        """Access to alerts component."""
        ...


# =============================================================================
# MINIMAL IMPLEMENTATIONS (DEVELOPMENT/TESTING)
# =============================================================================


class ConsoleLogger:
    """Simple console-based logger for development and testing."""

    def __init__(self, level: str = "INFO") -> None:
        self._logger = logging.getLogger("flext-console")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def trace(self, message: str, **context: object) -> None:
        """Log trace-level information."""
        self._log_with_context("DEBUG", f"[TRACE] {message}", context)

    def debug(self, message: str, **context: object) -> None:
        """Log debug information."""
        self._log_with_context("DEBUG", message, context)

    def info(self, message: str, **context: object) -> None:
        """Log informational messages."""
        self._log_with_context("INFO", message, context)

    def warn(self, message: str, **context: object) -> None:
        """Log warning messages."""
        self._log_with_context("WARNING", message, context)

    def error(
        self,
        message: str,
        *,
        error_code: str | None = None,
        exception: Exception | None = None,
        **context: object,
    ) -> None:
        """Log error messages."""
        full_context = dict(context)
        if error_code:
            full_context["error_code"] = error_code
        if exception:
            full_context["exception"] = str(exception)
        self._log_with_context("ERROR", message, full_context)

    def fatal(self, message: str, **context: object) -> None:
        """Log fatal system errors."""
        self._log_with_context("CRITICAL", message, context)

    def audit(
        self,
        message: str,
        *,
        user_id: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        outcome: str | None = None,
        **context: object,
    ) -> None:
        """Log audit events."""
        audit_context = {
            "audit": True,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "outcome": outcome,
            **context,
        }
        self._log_with_context("INFO", f"[AUDIT] {message}", audit_context)

    def _log_with_context(
        self, level: str, message: str, context: dict[str, object]
    ) -> None:
        """Log message with context."""
        correlation_id = FlextContext.get_correlation_id()
        if correlation_id:
            context["correlation_id"] = correlation_id

        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            full_message = f"{message} | {context_str}"
        else:
            full_message = message

        getattr(self._logger, level.lower())(full_message)


class NoOpSpan:
    """No-operation span for development/testing."""

    def add_context(self, key: str, value: object) -> None:
        """No-op add context."""

    def add_error(self, error: Exception) -> None:
        """No-op add error."""


class NoOpTracer:
    """No-operation tracer for development and testing."""

    @contextmanager
    def business_span(
        self, operation_name: str, **context: object
    ) -> Generator[FlextSpanProtocol]:
        """Create no-op business span."""
        _ = operation_name, context  # Acknowledge unused parameters
        yield NoOpSpan()

    @contextmanager
    def technical_span(
        self,
        operation_name: str,
        *,
        component: str | None = None,
        resource: str | None = None,
        **context: object,
    ) -> Generator[FlextSpanProtocol]:
        """Create no-op technical span."""
        _ = (
            operation_name,
            component,
            resource,
            context,
        )  # Acknowledge unused parameters
        yield NoOpSpan()

    @contextmanager
    def error_span(
        self,
        operation_name: str,
        *,
        error_message: str,
        error_code: str | None = None,
        **context: object,
    ) -> Generator[FlextSpanProtocol]:
        """Create no-op error span."""
        _ = (
            operation_name,
            error_message,
            error_code,
            context,
        )  # Acknowledge unused parameters
        yield NoOpSpan()


class InMemoryMetrics:
    """In-memory metrics collector for development and testing."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}

    def increment(
        self, metric_name: str, value: int = 1, tags: dict[str, str] | None = None
    ) -> None:
        """Increment counter metric."""
        key = self._build_key(metric_name, tags)
        self._counters[key] = self._counters.get(key, 0) + value

    def histogram(
        self, metric_name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Record histogram value."""
        key = self._build_key(metric_name, tags)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

    def gauge(
        self, metric_name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Set gauge value."""
        key = self._build_key(metric_name, tags)
        self._gauges[key] = value

    def get_metrics(self) -> dict[str, object]:
        """Get all collected metrics."""
        return {
            "counters": self._counters.copy(),
            "histograms": {
                k: {"values": v, "count": len(v)} for k, v in self._histograms.items()
            },
            "gauges": self._gauges.copy(),
        }

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self._counters.clear()
        self._histograms.clear()
        self._gauges.clear()

    def _build_key(self, metric_name: str, tags: dict[str, str] | None) -> str:
        """Build metric key with tags."""
        if not tags:
            return metric_name
        tag_pairs = [f"{k}={v}" for k, v in sorted(tags.items())]
        return f"{metric_name},{','.join(tag_pairs)}"


class SimpleAlerts:
    """Simple alerting implementation for development and testing."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("flext-alerts")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - ALERT[%(levelname)s] - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def info(self, message: str, **context: object) -> None:
        """Send info alert."""
        self._send_alert("INFO", message, context)

    def warning(self, message: str, **context: object) -> None:
        """Send warning alert."""
        self._send_alert("WARNING", message, context)

    def error(self, message: str, **context: object) -> None:
        """Send error alert."""
        self._send_alert("ERROR", message, context)

    def critical(self, message: str, **context: object) -> None:
        """Send critical alert."""
        self._send_alert("CRITICAL", message, context)

    def _send_alert(self, level: str, message: str, context: dict[str, object]) -> None:
        """Send alert through logger."""
        correlation_id = FlextContext.get_correlation_id()
        alert_data = {
            "message": message,
            "context": context,
            "timestamp": datetime.now(UTC).isoformat(),
            "correlation_id": correlation_id,
            "service": FlextContext.get_service_name(),
        }

        context_str = ", ".join(
            f"{k}={v}" for k, v in alert_data.items() if v is not None
        )
        full_message = f"{message} | {context_str}"

        getattr(self._logger, level.lower())(full_message)


class MinimalObservability:
    """Minimal observability implementation for development and testing."""

    def __init__(self, log_level: str = "INFO") -> None:
        self._log = ConsoleLogger(log_level)
        self._trace = NoOpTracer()
        self._metrics = InMemoryMetrics()
        self._alerts = SimpleAlerts()

    @property
    def log(self) -> FlextLoggerProtocol:
        """Access to logging component."""
        return self._log

    @property
    def trace(self) -> FlextTracerProtocol:
        """Access to tracing component."""
        return self._trace

    @property
    def metrics(self) -> FlextMetricsProtocol:
        """Access to metrics component."""
        return self._metrics

    @property
    def alerts(self) -> FlextAlertsProtocol:
        """Access to alerts component."""
        return self._alerts


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def get_observability(
    *,
    log_level: str = "INFO",
    force_recreate: bool = False,  # noqa: ARG001
) -> FlextObservabilityProtocol:
    """Get observability instance for development and testing.

    This returns a minimal implementation suitable for development.
    For production, use flext-observability package instead.

    Args:
        log_level: Log level for console logger
        force_recreate: Force creation of new instance (creates new instance each time)

    Returns:
        Minimal observability implementation

    """
    # Always create new instance for development/testing isolation
    return MinimalObservability(log_level)


def configure_minimal_observability(
    service_name: str, *, log_level: str = "INFO"
) -> FlextObservabilityProtocol:
    """Configure minimal observability for development/testing.

    Args:
        service_name: Name of the service
        log_level: Log level for console output

    Returns:
        Configured minimal observability instance

    """
    # Set service context
    FlextContext.set_service_name(service_name)

    # Get/create observability instance
    obs = get_observability(log_level=log_level, force_recreate=True)

    # Log configuration
    obs.log.info(
        "Minimal observability configured",
        service=service_name,
        log_level=log_level,
        implementation="flext-core-minimal",
    )

    return obs


# =============================================================================
# BACKWARD COMPATIBILITY (DEPRECATED)
# =============================================================================


class FlextObs:
    """DEPRECATED: Use get_observability() instead.

    This class exists for backward compatibility only.
    New code should use the protocol-based approach with get_observability().
    """

    Log: FlextLoggerProtocol
    Trace: FlextTracerProtocol
    Metrics: FlextMetricsProtocol
    Alert: FlextAlertsProtocol

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Initialize subclass with observability components."""
        # Get default observability instance
        obs = get_observability()
        cls.Log = obs.log
        cls.Trace = obs.trace
        cls.Metrics = obs.metrics
        cls.Alert = obs.alerts
        super().__init_subclass__(**kwargs)


def configure_observability(  # noqa: PLR0913
    service_name: str,
    *,
    log_level: str = "INFO",
    log_format: str = "json",  # noqa: ARG001
    tracing_enabled: bool = False,  # noqa: ARG001
    metrics_enabled: bool = True,  # noqa: ARG001
    alerts_enabled: bool = True,  # noqa: ARG001
    jaeger_endpoint: str = "http://localhost:14268/api/traces",  # noqa: ARG001
) -> None:
    """DEPRECATED: Use configure_minimal_observability() instead.

    This function exists for backward compatibility only.
    """
    configure_minimal_observability(service_name, log_level=log_level)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ConsoleLogger",
    "FlextAlertsProtocol",
    "FlextLoggerProtocol",
    "FlextMetricsProtocol",
    "FlextObs",
    "FlextObservabilityProtocol",
    "FlextSpanProtocol",
    "FlextTracerProtocol",
    "InMemoryMetrics",
    "MinimalObservability",
    "NoOpSpan",
    "NoOpTracer",
    "configure_minimal_observability",
    "configure_observability",
    "get_observability",
]
