"""Simple test coverage for observability.py module.

This test suite focuses on testing the actual API of observability.py.
"""

from __future__ import annotations

import pytest

from flext_core.observability import (
    ConsoleLogger,
    InMemoryMetrics,
    MinimalObservability,
    NoOpTracer,
    SimpleAlerts,
    get_observability,
)
from flext_core.protocols import (
    FlextAlertsProtocol,
    FlextLoggerProtocol,
    FlextMetricsProtocol,
    FlextObservabilityProtocol,
    FlextTracerProtocol,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestConsoleLogger:
    """Test ConsoleLogger implementation."""

    def test_logger_creation(self) -> None:
        """Test logger creation."""
        logger = ConsoleLogger()
        assert hasattr(logger, "_logger")
        assert logger._logger.name == "flext-console"

    def test_logger_methods_exist(self) -> None:
        """Test that all required methods exist."""
        logger = ConsoleLogger()

        assert hasattr(logger, "trace")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warn")
        assert hasattr(logger, "error")
        assert hasattr(logger, "fatal")
        assert hasattr(logger, "audit")

    def test_logger_implements_protocol(self) -> None:
        """Test that logger implements protocol."""
        logger = ConsoleLogger()
        assert isinstance(logger, FlextLoggerProtocol)

    def test_logger_basic_methods(self) -> None:
        """Test basic logging methods work."""
        logger = ConsoleLogger()

        # These should not raise errors
        logger.trace("trace message")
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.fatal("fatal message")
        logger.audit("audit message", user_id="123")


class TestNoOpTracer:
    """Test NoOpTracer implementation."""

    def test_tracer_implements_protocol(self) -> None:
        """Test that tracer implements protocol."""
        tracer = NoOpTracer()
        assert isinstance(tracer, FlextTracerProtocol)

    def test_business_span(self) -> None:
        """Test business span context manager."""
        tracer = NoOpTracer()

        with tracer.business_span("test-operation") as span:
            assert span is not None
            if hasattr(span, "add_context"):
                span.add_context("key", "value")
            if hasattr(span, "add_error"):
                span.add_error(Exception("test error"))

    def test_technical_span(self) -> None:
        """Test technical span context manager."""
        tracer = NoOpTracer()

        with tracer.technical_span("db-query", component="database") as span:
            assert span is not None
            if hasattr(span, "add_context"):
                span.add_context("query", "SELECT * FROM users")

    def test_error_span(self) -> None:
        """Test error span context manager."""
        tracer = NoOpTracer()

        with tracer.error_span(
            "failed-operation",
            error_message="Operation failed",
        ) as span:
            assert span is not None


class TestInMemoryMetrics:
    """Test InMemoryMetrics implementation."""

    def test_metrics_implements_protocol(self) -> None:
        """Test that metrics implements protocol."""
        metrics = InMemoryMetrics()
        assert isinstance(metrics, FlextMetricsProtocol)

    def test_increment_counter(self) -> None:
        """Test counter increment."""
        metrics = InMemoryMetrics()

        metrics.increment("test.counter", 1)
        metrics.increment("test.counter", 5)

        all_metrics = metrics.get_metrics()
        assert "counters" in all_metrics
        counters = all_metrics.get("counters")
        assert isinstance(counters, dict)
        assert "test.counter" in counters

    def test_histogram_recording(self) -> None:
        """Test histogram recording."""
        metrics = InMemoryMetrics()

        metrics.histogram("test.histogram", 10.5)
        metrics.histogram("test.histogram", 20.0)

        all_metrics = metrics.get_metrics()
        assert "histograms" in all_metrics
        histograms = all_metrics.get("histograms")
        assert isinstance(histograms, dict)
        assert "test.histogram" in histograms

    def test_gauge_setting(self) -> None:
        """Test gauge setting."""
        metrics = InMemoryMetrics()

        metrics.gauge("test.gauge", 42.0)
        metrics.gauge("test.gauge", 35.0)  # Should overwrite

        all_metrics = metrics.get_metrics()
        assert "gauges" in all_metrics
        gauges = all_metrics.get("gauges")
        assert isinstance(gauges, dict)
        assert gauges.get("test.gauge") == 35.0

    def test_metrics_with_tags(self) -> None:
        """Test metrics with tags."""
        metrics = InMemoryMetrics()

        tags = {"environment": "test", "service": "api"}
        metrics.increment("tagged.counter", 1, tags)

        all_metrics = metrics.get_metrics()
        assert "counters" in all_metrics

    def test_clear_metrics(self) -> None:
        """Test clearing metrics."""
        metrics = InMemoryMetrics()

        metrics.increment("test", 1)
        metrics_data = metrics.get_metrics()
        counters = metrics_data.get("counters")
        assert isinstance(counters, dict)
        assert len(counters) > 0

        metrics.clear_metrics()
        cleared_metrics = metrics.get_metrics()
        counters = cleared_metrics.get("counters")
        assert isinstance(counters, dict)
        assert len(counters) == 0


class TestSimpleAlerts:
    """Test SimpleAlerts implementation."""

    def test_alerts_implements_protocol(self) -> None:
        """Test that alerts implements protocol."""
        alerts = SimpleAlerts()
        assert isinstance(alerts, FlextAlertsProtocol)

    def test_alert_methods_exist(self) -> None:
        """Test that all alert methods exist."""
        alerts = SimpleAlerts()

        assert hasattr(alerts, "info")
        assert hasattr(alerts, "warning")
        assert hasattr(alerts, "error")
        assert hasattr(alerts, "critical")

    def test_alert_methods_work(self) -> None:
        """Test that alert methods work without errors."""
        alerts = SimpleAlerts()

        # These should not raise errors
        alerts.info("Info alert")
        alerts.warning("Warning alert")
        alerts.error("Error alert")
        alerts.critical("Critical alert")

    def test_alerts_with_context(self) -> None:
        """Test alerts with context."""
        alerts = SimpleAlerts()

        # Should accept keyword arguments
        alerts.info("Alert with context", key="value", code=123)


class TestMinimalObservability:
    """Test MinimalObservability composite implementation."""

    def test_observability_implements_protocol(self) -> None:
        """Test that observability implements protocol."""
        obs = MinimalObservability()
        assert isinstance(obs, FlextObservabilityProtocol)

    def test_observability_components(self) -> None:
        """Test that all components exist."""
        obs = MinimalObservability()

        assert hasattr(obs, "log")
        assert hasattr(obs, "trace")
        assert hasattr(obs, "metrics")
        assert hasattr(obs, "alerts")
        assert hasattr(obs, "health")

    def test_component_types(self) -> None:
        """Test component type compliance."""
        obs = MinimalObservability()

        assert isinstance(obs.log, FlextLoggerProtocol)
        assert isinstance(obs.trace, FlextTracerProtocol)
        assert isinstance(obs.metrics, FlextMetricsProtocol)
        assert isinstance(obs.alerts, FlextAlertsProtocol)

    def test_observability_integration(self) -> None:
        """Test integrated usage."""
        obs = MinimalObservability()

        # Test cross-component usage
        obs.log.info("Starting operation")
        obs.metrics.increment("operation.start", 1)

        with obs.trace.business_span("process-data") as span:
            if hasattr(span, "add_context"):
                span.add_context("step", "processing")
            obs.metrics.gauge("operation.progress", 50.0)

        obs.alerts.info("Operation completed")

    def test_health_component(self) -> None:
        """Test health monitoring component."""
        obs = MinimalObservability()

        health_status = obs.health.health_check()
        assert isinstance(health_status, dict)
        assert "status" in health_status

        assert obs.health.ready_check() is True
        assert obs.health.live_check() is True


class TestFactoryFunction:
    """Test get_observability factory function."""

    def test_get_observability_returns_protocol(self) -> None:
        """Test that factory returns observability protocol."""
        obs = get_observability()
        assert isinstance(obs, FlextObservabilityProtocol)

    def test_get_observability_with_params(self) -> None:
        """Test factory with parameters."""
        obs = get_observability(log_level="DEBUG", force_recreate=True)
        assert isinstance(obs, FlextObservabilityProtocol)

    def test_get_observability_components_work(self) -> None:
        """Test that factory-created observability works."""
        obs = get_observability()

        # Test basic functionality
        obs.log.info("Test message")
        obs.metrics.increment("test.counter", 1)

        with obs.trace.business_span("test-operation"):
            pass


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_logger_with_none_values(self) -> None:
        """Test logger handles None values gracefully."""
        logger = ConsoleLogger()

        # These should not raise errors
        try:
            logger.info("Test with context", context=None)
            logger.error("Error", exception=None, error_code=None)
        except Exception as e:
            pytest.fail(f"Logger should handle None values: {e}")

    def test_metrics_edge_cases(self) -> None:
        """Test metrics with edge case values."""
        metrics = InMemoryMetrics()

        try:
            metrics.increment("test", 0)
            metrics.gauge("test", 0.0)
            metrics.histogram("test", -1.0)  # Negative values should work
        except Exception as e:
            pytest.fail(f"Metrics should handle edge cases: {e}")

    def test_tracer_error_handling(self) -> None:
        """Test tracer handles errors gracefully."""
        tracer = NoOpTracer()

        try:
            with tracer.business_span("test") as span:
                if hasattr(span, "add_context"):
                    span.add_context("", "")  # Empty key
                    span.add_context("key", None)  # None value
                if hasattr(span, "add_error"):
                    span.add_error(ValueError("test error"))
        except Exception as e:
            pytest.fail(f"Tracer should handle edge cases: {e}")


class TestIntegrationScenarios:
    """Test integration scenarios between components."""

    def test_logging_with_metrics(self) -> None:
        """Test logging and metrics integration."""
        obs = get_observability()

        # Log an operation and record metrics
        obs.log.info("Processing batch", batch_size=100)
        obs.metrics.increment("batch.processed", 1)
        obs.metrics.gauge("batch.size", 100)

    def test_tracing_with_logging_and_metrics(self) -> None:
        """Test full observability integration."""
        obs = get_observability()

        with obs.trace.business_span("user-registration") as span:
            obs.log.info("Starting user registration")
            obs.metrics.increment("user.registration.start", 1)

            if hasattr(span, "add_context"):
                span.add_context("user_type", "premium")
            obs.log.debug("Validating user data")

            obs.metrics.histogram("registration.duration", 150.0)
            obs.log.info("User registration completed")
            obs.alerts.info("New user registered successfully")

    def test_error_flow(self) -> None:
        """Test error handling flow across components."""
        obs = get_observability()

        with obs.trace.error_span("failed-operation", error_message="Database error"):
            obs.log.error("Database connection failed", error_code="DB001")
            obs.metrics.increment("errors.database", 1)
            obs.alerts.error("Database connection issue detected")
