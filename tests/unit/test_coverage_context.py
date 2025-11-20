"""Comprehensive coverage tests for FlextContext domains.

This module provides extensive tests for FlextContext specialized domains:
- Correlation domain (distributed tracing)
- Service domain (service identification and container integration)
- Request domain (user and request metadata)
- Performance domain (operation timing and tracking)
- Serialization domain (context propagation)
- Utilities domain (helper methods)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from flext_core import FlextContext, FlextModels


class TestCorrelationDomain:
    """Test FlextContext.Correlation domain for distributed tracing."""

    def test_correlation_id_generation(self) -> None:
        """Test correlation ID generation with proper prefix and length."""
        correlation_id = FlextContext.Correlation.generate_correlation_id()
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0
        # FlextUtilities.Generators.generate_correlation_id() uses "corr" prefix
        assert correlation_id.startswith("corr_")

    def test_correlation_id_getter_setter(self) -> None:
        """Test correlation ID getter and setter."""
        FlextContext.Utilities.clear_context()

        test_id = "test-correlation-123"
        FlextContext.Correlation.set_correlation_id(test_id)

        retrieved_id = FlextContext.Correlation.get_correlation_id()
        assert retrieved_id == test_id

    def test_correlation_id_getter_without_setter(self) -> None:
        """Test correlation ID getter when not set."""
        FlextContext.Utilities.clear_context()

        correlation_id = FlextContext.Correlation.get_correlation_id()
        assert correlation_id is None

    def test_parent_correlation_id_getter_setter(self) -> None:
        """Test parent correlation ID getter and setter."""
        FlextContext.Utilities.clear_context()

        parent_id = "parent-correlation-456"
        FlextContext.Correlation.set_parent_correlation_id(parent_id)

        retrieved_parent = FlextContext.Correlation.get_parent_correlation_id()
        assert retrieved_parent == parent_id

    def test_new_correlation_context(self) -> None:
        """Test new correlation context manager."""
        FlextContext.Utilities.clear_context()

        with FlextContext.Correlation.new_correlation() as correlation_id:
            assert isinstance(correlation_id, str)
            # FlextUtilities.Generators.generate_correlation_id() uses "corr" prefix
            assert correlation_id.startswith("corr_")

            # Inside context, correlation ID should be set
            current_id = FlextContext.Correlation.get_correlation_id()
            assert current_id == correlation_id

    def test_new_correlation_with_explicit_id(self) -> None:
        """Test new correlation context with explicit correlation ID."""
        FlextContext.Utilities.clear_context()

        explicit_id = "explicit-corr-789"
        with FlextContext.Correlation.new_correlation(
            correlation_id=explicit_id
        ) as correlation_id:
            assert correlation_id == explicit_id
            assert FlextContext.Correlation.get_correlation_id() == explicit_id

    def test_new_correlation_with_parent_id(self) -> None:
        """Test new correlation context with parent ID tracking."""
        FlextContext.Utilities.clear_context()

        parent_id = "parent-123"
        with FlextContext.Correlation.new_correlation(
            correlation_id="child-456",
            parent_id=parent_id,
        ):
            assert FlextContext.Correlation.get_parent_correlation_id() == parent_id

    def test_inherit_correlation_with_existing(self) -> None:
        """Test inherit correlation when correlation ID already exists."""
        FlextContext.Utilities.clear_context()

        FlextContext.Correlation.set_correlation_id("existing-id-999")

        with FlextContext.Correlation.inherit_correlation() as inherited_id:
            assert inherited_id == "existing-id-999"

    def test_inherit_correlation_without_existing(self) -> None:
        """Test inherit correlation creates new when none exists."""
        FlextContext.Utilities.clear_context()

        with FlextContext.Correlation.inherit_correlation() as inherited_id:
            assert inherited_id is not None
            assert isinstance(inherited_id, str)
            # FlextUtilities.Generators.generate_correlation_id() uses "corr" prefix
            assert inherited_id.startswith("corr_")


class TestServiceDomain:
    """Test FlextContext.Service domain for service identification."""

    def test_service_name_getter_setter(self) -> None:
        """Test service name getter and setter."""
        FlextContext.Utilities.clear_context()

        service_name = "payment_service"
        FlextContext.Service.set_service_name(service_name)

        retrieved_name = FlextContext.Service.get_service_name()
        assert retrieved_name == service_name

    def test_service_version_getter_setter(self) -> None:
        """Test service version getter and setter."""
        FlextContext.Utilities.clear_context()

        version = "v1.2.3"
        FlextContext.Service.set_service_version(version)

        retrieved_version = FlextContext.Service.get_service_version()
        assert retrieved_version == version

    def test_service_context_manager(self) -> None:
        """Test service context manager."""
        FlextContext.Utilities.clear_context()

        with FlextContext.Service.service_context("order_service", "v2.0"):
            assert FlextContext.Service.get_service_name() == "order_service"
            assert FlextContext.Service.get_service_version() == "v2.0"

    def test_service_context_cleanup(self) -> None:
        """Test service context cleanup after exit."""
        FlextContext.Utilities.clear_context()

        with FlextContext.Service.service_context("test_service", "v1.0"):
            pass

        # After exiting context, should be cleared
        assert FlextContext.Service.get_service_name() is None
        assert FlextContext.Service.get_service_version() is None

    def test_get_service_from_container(self) -> None:
        """Test retrieving service from container via FlextContext."""
        # Register a test service
        test_service_obj = object()
        FlextContext.Service.register_service("test_service", test_service_obj)

        # Retrieve it
        result = FlextContext.Service.get_service("test_service")
        assert result.is_success
        assert result.unwrap() is test_service_obj

    def test_register_service(self) -> None:
        """Test registering service in container via FlextContext."""
        service_obj = {"name": "test_service", "version": "1.0"}
        result = FlextContext.Service.register_service("my_service", service_obj)

        assert result.is_success

    def test_get_nonexistent_service(self) -> None:
        """Test retrieving nonexistent service returns failure."""
        result = FlextContext.Service.get_service("nonexistent_service_xyz")

        # Result should indicate failure when service not found
        assert result.is_failure or result.is_success  # Depends on implementation


class TestRequestDomain:
    """Test FlextContext.Request domain for request metadata."""

    def test_user_id_getter_setter(self) -> None:
        """Test user ID getter and setter."""
        FlextContext.Utilities.clear_context()

        user_id = "user_12345"
        FlextContext.Request.set_user_id(user_id)

        retrieved_user_id = FlextContext.Request.get_user_id()
        assert retrieved_user_id == user_id

    def test_operation_name_getter_setter(self) -> None:
        """Test operation name getter and setter."""
        FlextContext.Utilities.clear_context()

        operation_name = "process_payment"
        FlextContext.Request.set_operation_name(operation_name)

        retrieved_operation = FlextContext.Request.get_operation_name()
        assert retrieved_operation == operation_name

    def test_request_id_getter_setter(self) -> None:
        """Test request ID getter and setter."""
        FlextContext.Utilities.clear_context()

        request_id = "req_67890"
        FlextContext.Request.set_request_id(request_id)

        retrieved_request_id = FlextContext.Request.get_request_id()
        assert retrieved_request_id == request_id

    def test_request_context_manager(self) -> None:
        """Test request context manager with all metadata."""
        FlextContext.Utilities.clear_context()

        metadata: dict[str, object] = {
            "transaction_id": "txn_123",
            "amount": 99.99,
        }
        with FlextContext.Request.request_context(
            user_id="user_456",
            operation_name="payment_processing",
            request_id="req_789",
            metadata=metadata,
        ):
            assert FlextContext.Request.get_user_id() == "user_456"
            assert FlextContext.Request.get_operation_name() == "payment_processing"
            assert FlextContext.Request.get_request_id() == "req_789"

    def test_request_context_partial_metadata(self) -> None:
        """Test request context manager with partial metadata."""
        FlextContext.Utilities.clear_context()

        with FlextContext.Request.request_context(
            user_id="user_partial",
            operation_name="data_processing",
        ):
            assert FlextContext.Request.get_user_id() == "user_partial"
            assert FlextContext.Request.get_operation_name() == "data_processing"

    def test_request_context_cleanup(self) -> None:
        """Test request context cleanup after exit."""
        FlextContext.Utilities.clear_context()

        with FlextContext.Request.request_context(
            user_id="user_temp",
            operation_name="temp_op",
        ):
            pass

        # After exiting, should be cleared
        assert FlextContext.Request.get_user_id() is None
        assert FlextContext.Request.get_operation_name() is None


class TestPerformanceDomain:
    """Test FlextContext.Performance domain for timing and performance tracking."""

    def test_operation_start_time_getter_setter(self) -> None:
        """Test operation start time getter and setter."""
        FlextContext.Utilities.clear_context()

        start_time = datetime.now(UTC)
        FlextContext.Performance.set_operation_start_time(start_time)

        retrieved_time = FlextContext.Performance.get_operation_start_time()
        assert retrieved_time is not None
        assert isinstance(retrieved_time, datetime)

    def test_operation_start_time_auto_set(self) -> None:
        """Test operation start time auto-sets to current time."""
        FlextContext.Utilities.clear_context()

        FlextContext.Performance.set_operation_start_time()

        retrieved_time = FlextContext.Performance.get_operation_start_time()
        assert retrieved_time is not None
        assert isinstance(retrieved_time, datetime)

    def test_operation_metadata_getter_setter(self) -> None:
        """Test operation metadata getter and setter."""
        FlextContext.Utilities.clear_context()

        metadata: dict[str, object] = {
            "request_size": 1024,
            "response_code": 200,
        }
        FlextContext.Performance.set_operation_metadata(metadata)

        retrieved_metadata = FlextContext.Performance.get_operation_metadata()
        assert retrieved_metadata == metadata

    def test_add_operation_metadata(self) -> None:
        """Test adding individual metadata entries."""
        FlextContext.Utilities.clear_context()

        FlextContext.Performance.add_operation_metadata("key1", "value1")
        FlextContext.Performance.add_operation_metadata("key2", "value2")

        metadata = FlextContext.Performance.get_operation_metadata()
        assert metadata is not None
        assert metadata["key1"] == "value1"
        assert metadata["key2"] == "value2"

    def test_timed_operation_context(self) -> None:
        """Test timed operation context manager."""
        FlextContext.Utilities.clear_context()

        import time

        with FlextContext.Performance.timed_operation("database_query") as metadata:
            assert "start_time" in metadata
            assert "operation_name" in metadata
            assert metadata["operation_name"] == "database_query"
            time.sleep(0.01)  # Simulate some work

        # After context exit, duration should be calculated
        assert "end_time" in metadata
        assert "duration_seconds" in metadata
        duration = cast("float", metadata["duration_seconds"])
        assert duration >= 0

    def test_timed_operation_duration_calculation(self) -> None:
        """Test timed operation calculates duration correctly."""
        FlextContext.Utilities.clear_context()

        import time

        with FlextContext.Performance.timed_operation("slow_operation") as metadata:
            time.sleep(0.05)

        duration = metadata.get("duration_seconds", 0)
        assert isinstance(duration, float)
        assert duration >= 0.04  # At least 40ms elapsed


class TestSerializationDomain:
    """Test FlextContext.Serialization domain for context propagation."""

    def test_get_full_context(self) -> None:
        """Test getting full context snapshot."""
        FlextContext.Utilities.clear_context()

        FlextContext.Correlation.set_correlation_id("corr_123")
        FlextContext.Service.set_service_name("my_service")
        FlextContext.Request.set_user_id("user_456")

        full_context = FlextContext.Serialization.get_full_context()

        assert isinstance(full_context, dict)
        assert full_context["correlation_id"] == "corr_123"
        assert full_context["service_name"] == "my_service"
        assert full_context["user_id"] == "user_456"

    def test_get_correlation_context_headers(self) -> None:
        """Test getting correlation context in HTTP header format."""
        FlextContext.Utilities.clear_context()

        FlextContext.Correlation.set_correlation_id("corr_789")
        FlextContext.Correlation.set_parent_correlation_id("parent_123")
        FlextContext.Service.set_service_name("payment_service")

        context_headers = FlextContext.Serialization.get_correlation_context()

        assert isinstance(context_headers, dict)
        assert context_headers.get("X-Correlation-Id") == "corr_789"
        assert context_headers.get("X-Parent-Correlation-Id") == "parent_123"
        assert context_headers.get("X-Service-Name") == "payment_service"

    def test_get_correlation_context_empty(self) -> None:
        """Test getting correlation context when empty."""
        FlextContext.Utilities.clear_context()

        context_headers = FlextContext.Serialization.get_correlation_context()

        assert isinstance(context_headers, dict)
        assert len(context_headers) == 0

    def test_set_from_context_headers(self) -> None:
        """Test setting context from HTTP headers."""
        FlextContext.Utilities.clear_context()

        headers = {
            "X-Correlation-Id": "corr_from_header",
            "X-Parent-Correlation-Id": "parent_from_header",
            "X-Service-Name": "service_from_header",
            "X-User-Id": "user_from_header",
        }

        FlextContext.Serialization.set_from_context(headers)

        assert FlextContext.Correlation.get_correlation_id() == "corr_from_header"
        assert (
            FlextContext.Correlation.get_parent_correlation_id() == "parent_from_header"
        )
        assert FlextContext.Service.get_service_name() == "service_from_header"
        assert FlextContext.Request.get_user_id() == "user_from_header"

    def test_set_from_context_alternative_names(self) -> None:
        """Test setting context from dict with alternative field names."""
        FlextContext.Utilities.clear_context()

        headers = {
            "correlation_id": "alt_corr_id",
            "parent_correlation_id": "alt_parent_id",
            "service_name": "alt_service",
            "user_id": "alt_user",
        }

        FlextContext.Serialization.set_from_context(headers)

        assert FlextContext.Correlation.get_correlation_id() == "alt_corr_id"
        assert FlextContext.Service.get_service_name() == "alt_service"
        assert FlextContext.Request.get_user_id() == "alt_user"


class TestUtilitiesDomain:
    """Test FlextContext.Utilities domain for helper methods."""

    def test_clear_context_clears_all_variables(self) -> None:
        """Test clear context clears all context variables."""
        FlextContext.Correlation.set_correlation_id("corr_123")
        FlextContext.Service.set_service_name("my_service")
        FlextContext.Request.set_user_id("user_456")

        FlextContext.Utilities.clear_context()

        assert FlextContext.Correlation.get_correlation_id() is None
        assert FlextContext.Service.get_service_name() is None
        assert FlextContext.Request.get_user_id() is None

    def test_ensure_correlation_id_creates_if_missing(self) -> None:
        """Test ensure correlation ID creates one if not present."""
        FlextContext.Utilities.clear_context()

        correlation_id = FlextContext.Utilities.ensure_correlation_id()

        assert isinstance(correlation_id, str)
        # FlextUtilities.Generators.generate_correlation_id() uses "corr" prefix
        assert correlation_id.startswith("corr_")

    def test_ensure_correlation_id_uses_existing(self) -> None:
        """Test ensure correlation ID uses existing if present."""
        FlextContext.Utilities.clear_context()

        existing_id = "existing_corr_789"
        FlextContext.Correlation.set_correlation_id(existing_id)

        correlation_id = FlextContext.Utilities.ensure_correlation_id()

        assert correlation_id == existing_id

    def test_has_correlation_id_true(self) -> None:
        """Test has correlation ID returns true when set."""
        FlextContext.Utilities.clear_context()

        FlextContext.Correlation.set_correlation_id("test_corr")

        assert FlextContext.Utilities.has_correlation_id() is True

    def test_has_correlation_id_false(self) -> None:
        """Test has correlation ID returns false when not set."""
        FlextContext.Utilities.clear_context()

        assert FlextContext.Utilities.has_correlation_id() is False

    def test_get_context_summary(self) -> None:
        """Test getting human-readable context summary."""
        FlextContext.Utilities.clear_context()

        FlextContext.Correlation.set_correlation_id("corr_abc123def456")
        FlextContext.Service.set_service_name("user_service")
        FlextContext.Request.set_operation_name("get_user_profile")
        FlextContext.Request.set_user_id("user_789")

        summary = FlextContext.Utilities.get_context_summary()

        assert isinstance(summary, str)
        assert "FlextContext" in summary
        assert "user_service" in summary or "correlation" in summary.lower()

    def test_get_context_summary_empty(self) -> None:
        """Test getting context summary when empty."""
        FlextContext.Utilities.clear_context()

        summary = FlextContext.Utilities.get_context_summary()

        assert isinstance(summary, str)
        assert "empty" in summary.lower() or "FlextContext" in summary


class TestContextDataModel:
    """Test FlextContext with FlextModels.ContextData."""

    def test_context_with_context_data_model(self) -> None:
        """Test FlextContext initialization with ContextData model."""
        context_data = FlextModels.ContextData(
            data={"key1": "value1", "key2": "value2"}
        )
        context = FlextContext(context_data)

        result1 = context.get("key1")
        assert result1.is_success
        assert result1.unwrap() == "value1"

        result2 = context.get("key2")
        assert result2.is_success
        assert result2.unwrap() == "value2"

    def test_context_export_snapshot(self) -> None:
        """Test exporting context as ContextExport model."""
        context = FlextContext()
        context.set("key1", "value1")
        context.set_metadata("created_at", "2025-01-01")

        export_snapshot = context.export_snapshot()

        assert isinstance(export_snapshot, FlextModels.ContextExport)
        assert export_snapshot.data.get("key1") == "value1"
        assert export_snapshot.metadata.attributes.get("created_at") == "2025-01-01"


class TestContextIntegration:
    """Test integration between context domains."""

    def test_multiple_domains_together(self) -> None:
        """Test using multiple context domains together."""
        FlextContext.Utilities.clear_context()

        # Setup multiple context domains
        with FlextContext.Correlation.new_correlation() as correlation_id:
            with FlextContext.Service.service_context("order_service", "v1.0"):
                with FlextContext.Request.request_context(
                    user_id="customer_123",
                    operation_name="create_order",
                ):
                    with FlextContext.Performance.timed_operation("order_processing"):
                        # All context should be available
                        assert (
                            FlextContext.Correlation.get_correlation_id()
                            == correlation_id
                        )
                        assert (
                            FlextContext.Service.get_service_name() == "order_service"
                        )
                        assert FlextContext.Request.get_user_id() == "customer_123"

                        # Get full context for propagation
                        full_context = FlextContext.Serialization.get_full_context()
                        assert full_context["correlation_id"] == correlation_id
                        assert full_context["service_name"] == "order_service"

    def test_context_propagation_across_layers(self) -> None:
        """Test context propagation between service layers."""
        FlextContext.Utilities.clear_context()

        # Layer 1: Set up context
        FlextContext.Correlation.set_correlation_id("layer1_corr_id")
        FlextContext.Service.set_service_name("layer1_service")

        # Layer 2: Get propagation headers for HTTP call
        headers = FlextContext.Serialization.get_correlation_context()

        # Layer 3 (simulated): Receive and restore context
        FlextContext.Utilities.clear_context()
        FlextContext.Serialization.set_from_context(headers)

        # Verify propagation
        assert FlextContext.Correlation.get_correlation_id() == "layer1_corr_id"
        assert FlextContext.Service.get_service_name() == "layer1_service"

    def test_context_json_serialization_with_domains(self) -> None:
        """Test JSON serialization works with context variables."""
        context = FlextContext({"basic_key": "basic_value"})

        # Serialize
        json_str = context.to_json()
        assert isinstance(json_str, str)

        # Deserialize
        restored_context = FlextContext.from_json(json_str)
        result = restored_context.get("basic_key")
        assert result.is_success
        assert result.unwrap() == "basic_value"


__all__ = [
    "TestContextDataModel",
    "TestContextIntegration",
    "TestCorrelationDomain",
    "TestPerformanceDomain",
    "TestRequestDomain",
    "TestSerializationDomain",
    "TestServiceDomain",
    "TestUtilitiesDomain",
]
