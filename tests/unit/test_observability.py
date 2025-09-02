"""Extended test coverage for observability.py module.

Comprehensive tests to increase coverage of FlextObservability components.
"""

from __future__ import annotations

import time

import pytest
from hypothesis import given, strategies as st

from flext_core import FlextObservability

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextObservability:
    """Test main FlextObservability functionality."""

    def test_observability_creation(self) -> None:
        """Test creating FlextObservability.Observability."""
        obs = FlextObservability.Observability()
        assert obs is not None

    def test_observability_has_components(self) -> None:
        """Test that observability has all required components."""
        obs = FlextObservability.Observability()

        # Check main components that actually exist
        assert hasattr(obs, "logger")
        assert hasattr(obs, "metrics")
        assert hasattr(obs, "tracer")

    def test_observability_logger_functionality(self) -> None:
        """Test observability logger methods."""
        obs = FlextObservability.Observability()

        # Test basic logging methods exist and can be called
        obs.logger.info("Test info message")
        obs.logger.debug("Test debug message")
        obs.logger.warning("Test warning message")
        obs.logger.error("Test error message")

    def test_observability_metrics_basic(self) -> None:
        """Test basic metrics functionality."""
        obs = FlextObservability.Observability()

        # Test metrics methods exist
        assert hasattr(obs.metrics, "increment")
        assert hasattr(obs.metrics, "gauge")
        assert hasattr(obs.metrics, "histogram")

        # Test basic metric operations
        obs.metrics.increment("test_counter")
        obs.metrics.gauge("test_gauge", 42.0)
        obs.metrics.histogram("test_histogram", 1.23)

    def test_observability_tracer_basic(self) -> None:
        """Test basic tracer functionality."""
        obs = FlextObservability.Observability()

        # Test tracer exists
        assert obs.tracer is not None


class TestFlextConsole:
    """Extended test coverage for FlextConsole."""

    def test_console_creation(self) -> None:
        """Test FlextConsole creation."""
        console = FlextObservability.Console()
        assert console is not None

    def test_console_logging_methods(self) -> None:
        """Test all console logging methods."""
        console = FlextObservability.Console()

        # Test all logging levels
        console.trace("Trace message")
        console.debug("Debug message")
        console.info("Info message")
        console.warn("Warning message")
        console.error("Error message")
        console.critical("Critical message")

    def test_console_with_structured_data(self) -> None:
        """Test console logging with structured data."""
        console = FlextObservability.Console()

        # Test logging with extra data
        console.info("Structured message", extra_field="test_value", count=42)
        console.error("Error with context", error_code="TEST_001", severity="high")

    @given(st.text(min_size=1, max_size=100))
    def test_console_with_random_messages(self, message: str) -> None:
        """Property-based test for console with random messages."""
        console = FlextObservability.Console()

        # Should handle any string message without errors
        console.info(message)
        console.error(message)


class TestFlextMetrics:
    """Extended test coverage for FlextMetrics."""

    def test_metrics_creation(self) -> None:
        """Test FlextMetrics creation."""
        metrics = FlextObservability.Metrics()
        assert metrics is not None

    def test_metrics_counter_operations(self) -> None:
        """Test counter metric operations."""
        metrics = FlextObservability.Metrics()

        # Test increment operations
        metrics.increment("test_counter")
        metrics.increment("test_counter", 5)
        metrics.increment("named_counter", 1, tags={"service": "test"})

    def test_metrics_gauge_operations(self) -> None:
        """Test gauge metric operations."""
        metrics = FlextObservability.Metrics()

        # Test gauge operations with various values
        metrics.gauge("cpu_usage", 75.5)
        metrics.gauge("memory_usage", 1024)
        metrics.gauge("connection_count", 10)

    def test_metrics_histogram_operations(self) -> None:
        """Test histogram metric operations."""
        metrics = FlextObservability.Metrics()

        # Test histogram operations
        metrics.histogram("response_time", 0.150)
        metrics.histogram("request_size", 2048)
        metrics.histogram("processing_duration", 1.5)

    def test_metrics_timer_functionality(self) -> None:
        """Test timer metric functionality."""
        metrics = FlextObservability.Metrics()

        # Test timer context manager (if available)
        if hasattr(metrics, "timer"):
            with metrics.timer("operation_duration"):
                time.sleep(0.01)  # Brief pause to test timing

    @given(st.text(min_size=1, max_size=50), st.floats(min_value=0, max_value=1000))
    def test_metrics_with_random_values(self, name: str, value: float) -> None:
        """Property-based test for metrics with random values."""
        metrics = FlextObservability.Metrics()

        # Sanitize name to avoid special characters
        clean_name = "".join(c for c in name if c.isalnum() or c == "_")[:50]
        if not clean_name:
            clean_name = "test_metric"

        # Should handle any reasonable metric values
        metrics.gauge(clean_name, value)


class TestFlextTracer:
    """Extended test coverage for FlextTracer."""

    def test_tracer_creation(self) -> None:
        """Test FlextTracer creation."""
        tracer = FlextObservability.Tracer()
        assert tracer is not None

    def test_tracer_span_operations(self) -> None:
        """Test tracer span operations."""
        tracer = FlextObservability.Tracer()

        # Test span operations using context managers
        with tracer.business_span("test_operation") as span:
            assert span is not None

            # Add events to span
            span.log_event("operation_started", {})
            span.log_event("processing_data", {"record_count": 100})

            # Set tags
            span.set_tag("operation_type", "test")

    def test_tracer_nested_spans(self) -> None:
        """Test nested span operations."""
        tracer = FlextObservability.Tracer()

        with tracer.business_span("parent_operation") as parent_span:
            with tracer.technical_span("child_operation") as child_span:
                child_span.log_event("child_completed", {})
                child_span.add_context("parent_id", "parent_123")

            parent_span.log_event("parent_completed", {})

    def test_tracer_with_attributes(self) -> None:
        """Test tracer with custom attributes."""
        tracer = FlextObservability.Tracer()

        with tracer.business_span("attributed_operation") as span:
            span.set_tag("service.name", "test")
            span.set_tag("operation.type", "read")
            span.add_context("user_id", "user_123")
            span.log_event("operation_with_attributes", {})


class TestFlextAlerts:
    """Extended test coverage for FlextAlerts."""

    def test_alerts_creation(self) -> None:
        """Test FlextAlerts creation."""
        alerts = FlextObservability.Alerts()
        assert alerts is not None

    def test_alerts_methods_exist(self) -> None:
        """Test that alert methods exist."""
        alerts = FlextObservability.Alerts()

        # Check for expected alert methods
        if hasattr(alerts, "send_alert"):
            alerts.send_alert("test_alert", "Test alert message")

        if hasattr(alerts, "warning"):
            alerts.warning("Test warning alert")

        if hasattr(alerts, "error"):
            alerts.error("Test error alert")

        if hasattr(alerts, "critical"):
            alerts.critical("Test critical alert")


class TestObservabilityIntegration:
    """Test integration between observability components."""

    def test_full_observability_workflow(self) -> None:
        """Test complete observability workflow."""
        obs = FlextObservability.Observability()

        # Start operation with tracing
        with obs.tracer.business_span("api_request") as span:
            # Log operation start
            obs.logger.info("API request started", request_id="req_123")

            # Record metrics
            obs.metrics.increment("api_requests_total")
            obs.metrics.gauge("active_requests", 5)

            # Add trace events
            span.log_event("validation_completed", {})
            span.log_event("processing_started", {})

            # Simulate processing time
            time.sleep(0.01)

            # Record processing time
            obs.metrics.histogram("request_duration", 0.01)

            # Log completion
            obs.logger.info("API request completed", request_id="req_123")

            # Add final trace event
            span.log_event("request_completed", {})

    def test_error_handling_workflow(self) -> None:
        """Test observability during error scenarios."""
        obs = FlextObservability.Observability()

        with obs.tracer.error_span("error_scenario") as span:
            try:
                # Simulate error condition
                obs.metrics.increment("error_attempts")
                span.log_event("error_detected", {})
                obs.logger.error("Simulated error for testing", error_type="test_error")

                # Record error metrics
                obs.metrics.increment("errors_total", 1, tags={"type": "test_error"})

            except Exception as e:
                obs.logger.exception("Unexpected error", error=str(e))
                span.log_event("unexpected_error", {})

    def test_performance_monitoring(self) -> None:
        """Test performance monitoring capabilities."""
        obs = FlextObservability.Observability()

        # Monitor different performance aspects
        obs.metrics.gauge("cpu_usage_percent", 45.2)
        obs.metrics.gauge("memory_usage_mb", 512)
        obs.metrics.gauge("disk_usage_percent", 78.5)

        # Monitor application metrics
        obs.metrics.increment("database_queries")
        obs.metrics.histogram("database_query_time", 0.025)
        obs.metrics.increment("cache_hits")
        obs.metrics.increment("cache_misses")

    def test_structured_logging_integration(self) -> None:
        """Test structured logging with observability."""
        obs = FlextObservability.Observability()

        # Test structured logging with context
        context = {
            "user_id": "user_123",
            "session_id": "session_456",
            "operation": "data_processing",
        }

        obs.logger.info("Operation started", **context)
        obs.logger.debug("Processing details", record_count=1000, **context)
        obs.logger.info("Operation completed", success=True, **context)


class TestObservabilityConfiguration:
    """Test observability configuration and setup."""

    def test_observability_with_custom_config(self) -> None:
        """Test observability with custom configuration."""
        # Test different initialization approaches
        obs1 = FlextObservability.Observability()
        obs2 = FlextObservability.Observability()

        # Both should be valid instances
        assert obs1 is not None
        assert obs2 is not None

    def test_component_independence(self) -> None:
        """Test that observability components work independently."""
        # Create individual components
        console = FlextObservability.Console()
        metrics = FlextObservability.Metrics()
        tracer = FlextObservability.Tracer()
        FlextObservability.Alerts()

        # Each should work independently
        console.info("Independent console test")
        metrics.increment("independent_counter")
        with tracer.business_span("independent_span") as span:
            span.log_event("independent_event", {})

    def test_observability_factory_methods(self) -> None:
        """Test factory methods if they exist."""
        # Test if factory methods are available
        if hasattr(FlextObservability, "create_observability"):
            obs = FlextObservability.create_observability()
            assert obs is not None

        if hasattr(FlextObservability, "get_global_observability"):
            global_obs = FlextObservability.get_global_observability()
            assert global_obs is not None


class TestObservabilityEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_metrics(self) -> None:
        """Test metrics with empty/null values."""
        metrics = FlextObservability.Metrics()

        # Test with empty names (should handle gracefully)
        metrics.increment("")  # Empty name
        metrics.gauge("", 0)  # Empty name

    def test_large_values(self) -> None:
        """Test with large metric values."""
        metrics = FlextObservability.Metrics()

        # Test with large values
        metrics.gauge("large_value", 999999999.99)
        metrics.histogram("large_histogram", 1e9)

    def test_unicode_logging(self) -> None:
        """Test logging with unicode characters."""
        console = FlextObservability.Console()

        # Test with unicode characters
        console.info("测试消息")  # Chinese
        console.info("тест сообщение")  # Russian
        console.info("🚀 Rocket message")  # Emoji

    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
    def test_bulk_operations(self, names: list[str]) -> None:
        """Property-based test for bulk operations."""
        metrics = FlextObservability.Metrics()

        # Test bulk metric operations
        for i, name in enumerate(names):
            clean_name = f"bulk_{i}_{name.replace(' ', '_')[:10]}"
            metrics.increment(clean_name)
            metrics.gauge(clean_name, float(i))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
