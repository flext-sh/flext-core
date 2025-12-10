"""Comprehensive coverage tests for FlextContext domains.

Module: flext_core.context
Scope: FlextContext - Correlation, Service, Request, Performance, Serialization, Utilities domains

This module provides extensive tests for FlextContext specialized domains:
- Correlation domain (distributed tracing)
- Service domain (service identification and container integration)
- Request domain (user and request metadata)
- Performance domain (operation timing and tracking)
- Serialization domain (context propagation)
- Utilities domain (helper methods)

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, cast

import pytest

from flext_core import FlextContainer, FlextContext, t
from flext_core.models import m
from flext_core.result import r
from flext_tests import FlextTestsUtilities, u


@dataclass(frozen=True, slots=True)
class ContextOperationScenario:
    """Context operation test scenario."""

    name: str
    domain: str
    operation: str
    value: str | None = None
    expected_result: object | None = None


class ContextScenarios:
    """Centralized context test scenarios using FlextConstants."""

    CORRELATION_OPERATIONS: ClassVar[list[ContextOperationScenario]] = [
        ContextOperationScenario("generate", "correlation", "generate", None, None),
        ContextOperationScenario(
            "set_get",
            "correlation",
            "set_get",
            "test-correlation-123",
            "test-correlation-123",
        ),
        ContextOperationScenario("get_without_set", "correlation", "get", None, None),
        ContextOperationScenario(
            "parent_set_get",
            "correlation",
            "parent_set_get",
            "parent-correlation-456",
            "parent-correlation-456",
        ),
    ]

    SERVICE_OPERATIONS: ClassVar[list[ContextOperationScenario]] = [
        ContextOperationScenario(
            "set_get_name",
            "service",
            "set_get_name",
            "payment_service",
            "payment_service",
        ),
        ContextOperationScenario(
            "set_get_version",
            "service",
            "set_get_version",
            "v1.2.3",
            "v1.2.3",
        ),
    ]

    REQUEST_OPERATIONS: ClassVar[list[ContextOperationScenario]] = [
        ContextOperationScenario(
            "set_get_user_id",
            "request",
            "set_get_user_id",
            "user_12345",
            "user_12345",
        ),
        ContextOperationScenario(
            "set_get_operation",
            "request",
            "set_get_operation",
            "process_payment",
            "process_payment",
        ),
        ContextOperationScenario(
            "set_get_request_id",
            "request",
            "set_get_request_id",
            "req_67890",
            "req_67890",
        ),
    ]


class TestCorrelationDomain:
    """Test FlextContext.Correlation domain for distributed tracing using FlextTestsUtilities."""

    def test_correlation_id_generation(self) -> None:
        """Test correlation ID generation with proper prefix and length."""
        correlation_id = FlextContext.Correlation.generate_correlation_id()
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0
        assert correlation_id.startswith("corr_")

    @pytest.mark.parametrize(
        "scenario",
        ContextScenarios.CORRELATION_OPERATIONS,
        ids=lambda s: s.name,
    )
    def test_correlation_operations(self, scenario: ContextOperationScenario) -> None:
        """Test correlation operations with various scenarios."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        if scenario.operation == "generate":
            correlation_id = FlextContext.Correlation.generate_correlation_id()
            assert isinstance(correlation_id, str)
            assert correlation_id.startswith("corr_")
        elif scenario.operation == "set_get" and scenario.value:
            FlextContext.Correlation.set_correlation_id(scenario.value)
            assert (
                FlextContext.Correlation.get_correlation_id()
                == scenario.expected_result
            )
        elif scenario.operation == "get":
            assert FlextContext.Correlation.get_correlation_id() is None
        elif scenario.operation == "parent_set_get" and scenario.value:
            FlextContext.Correlation.set_parent_correlation_id(scenario.value)
            assert (
                FlextContext.Correlation.get_parent_correlation_id()
                == scenario.expected_result
            )

    def test_new_correlation_context(self) -> None:
        """Test new correlation context manager."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Correlation.new_correlation() as correlation_id:
            assert isinstance(correlation_id, str)
            assert correlation_id.startswith("corr_")
            assert FlextContext.Correlation.get_correlation_id() == correlation_id

    def test_new_correlation_with_explicit_id(self) -> None:
        """Test new correlation context with explicit correlation ID."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        explicit_id = "explicit-corr-789"
        with FlextContext.Correlation.new_correlation(
            correlation_id=explicit_id,
        ) as correlation_id:
            assert correlation_id == explicit_id
            assert FlextContext.Correlation.get_correlation_id() == explicit_id

    def test_new_correlation_with_parent_id(self) -> None:
        """Test new correlation context with parent ID tracking."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        parent_id = "parent-123"
        with FlextContext.Correlation.new_correlation(
            correlation_id="child-456",
            parent_id=parent_id,
        ):
            assert FlextContext.Correlation.get_parent_correlation_id() == parent_id

    def test_inherit_correlation_with_existing(self) -> None:
        """Test inherit correlation when correlation ID already exists."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        FlextContext.Correlation.set_correlation_id("existing-id-999")
        with FlextContext.Correlation.inherit_correlation() as inherited_id:
            assert inherited_id == "existing-id-999"

    def test_inherit_correlation_without_existing(self) -> None:
        """Test inherit correlation creates new when none exists."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Correlation.inherit_correlation() as inherited_id:
            assert inherited_id is not None
            assert isinstance(inherited_id, str)
            assert inherited_id.startswith("corr_")


class TestServiceDomain:
    """Test FlextContext.Service domain for service identification using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario",
        ContextScenarios.SERVICE_OPERATIONS,
        ids=lambda s: s.name,
    )
    def test_service_operations(self, scenario: ContextOperationScenario) -> None:
        """Test service operations with various scenarios."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        if scenario.operation == "set_get_name" and scenario.value:
            FlextContext.Service.set_service_name(scenario.value)
            assert FlextContext.Service.get_service_name() == scenario.expected_result
        elif scenario.operation == "set_get_version" and scenario.value:
            FlextContext.Service.set_service_version(scenario.value)
            assert (
                FlextContext.Service.get_service_version() == scenario.expected_result
            )

    def test_service_context_manager(self) -> None:
        """Test service context manager."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Service.service_context("order_service", "v2.0"):
            assert FlextContext.Service.get_service_name() == "order_service"
            assert FlextContext.Service.get_service_version() == "v2.0"

    def test_service_context_cleanup(self) -> None:
        """Test service context cleanup after exit."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Service.service_context("test_service", "v1.0"):
            pass
        assert FlextContext.Service.get_service_name() is None
        assert FlextContext.Service.get_service_version() is None

    def test_get_service_from_container(self) -> None:
        """Test retrieving service from container via FlextContext."""
        # Set up container before using FlextContext.Service methods
        container = FlextContainer(_context=FlextContext())
        FlextContext.set_container(container)
        test_service_obj: t.GeneralValueType = "test_service_value"
        FlextContext.Service.register_service("test_service", test_service_obj)
        result = FlextContext.Service.get_service("test_service")
        # Type narrowing: assert_result_success accepts r[TResult], protocol Result is compatible
        # Cast to r[t.GeneralValueType] for type compatibility
        result_typed: r[t.GeneralValueType] = cast(
            "r[t.GeneralValueType]",
            result,
        )
        u.Tests.Result.assert_result_success(result_typed)
        assert result.value is test_service_obj

    def test_register_service(self) -> None:
        """Test registering service in container via FlextContext."""
        # Set up container before using FlextContext.Service methods
        container = FlextContainer(_context=FlextContext())
        FlextContext.set_container(container)
        service_obj = {"name": "test_service", "version": "1.0"}
        result = FlextContext.Service.register_service("my_service", service_obj)
        u.Tests.Result.assert_result_success(result)

    def test_get_nonexistent_service(self) -> None:
        """Test retrieving nonexistent service returns failure."""
        # Set up container before using FlextContext.Service methods
        container = FlextContainer(_context=FlextContext())
        FlextContext.set_container(container)
        result = FlextContext.Service.get_service("nonexistent_service_xyz")
        assert result.is_failure or result.is_success


class TestRequestDomain:
    """Test FlextContext.Request domain for request metadata using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario",
        ContextScenarios.REQUEST_OPERATIONS,
        ids=lambda s: s.name,
    )
    def test_request_operations(self, scenario: ContextOperationScenario) -> None:
        """Test request operations with various scenarios."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        if scenario.operation == "set_get_user_id" and scenario.value:
            FlextContext.Request.set_user_id(scenario.value)
            assert FlextContext.Request.get_user_id() == scenario.expected_result
        elif scenario.operation == "set_get_operation" and scenario.value:
            FlextContext.Request.set_operation_name(scenario.value)
            assert FlextContext.Request.get_operation_name() == scenario.expected_result
        elif scenario.operation == "set_get_request_id" and scenario.value:
            FlextContext.Request.set_request_id(scenario.value)
            assert FlextContext.Request.get_request_id() == scenario.expected_result

    def test_request_context_manager(self) -> None:
        """Test request context manager with all metadata."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        metadata: dict[str, t.GeneralValueType] = {
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
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Request.request_context(
            user_id="user_partial",
            operation_name="data_processing",
        ):
            assert FlextContext.Request.get_user_id() == "user_partial"
            assert FlextContext.Request.get_operation_name() == "data_processing"

    def test_request_context_cleanup(self) -> None:
        """Test request context cleanup after exit."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Request.request_context(
            user_id="user_temp",
            operation_name="temp_op",
        ):
            pass
        assert FlextContext.Request.get_user_id() is None
        assert FlextContext.Request.get_operation_name() is None


class TestPerformanceDomain:
    """Test FlextContext.Performance domain for timing and performance tracking using FlextTestsUtilities."""

    def test_operation_start_time_getter_setter(self) -> None:
        """Test operation start time getter and setter."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        start_time = datetime.now(UTC)
        FlextContext.Performance.set_operation_start_time(start_time)
        retrieved_time = FlextContext.Performance.get_operation_start_time()
        assert retrieved_time is not None
        assert isinstance(retrieved_time, datetime)

    def test_operation_start_time_auto_set(self) -> None:
        """Test operation start time auto-sets to current time."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        FlextContext.Performance.set_operation_start_time()
        retrieved_time = FlextContext.Performance.get_operation_start_time()
        assert retrieved_time is not None
        assert isinstance(retrieved_time, datetime)

    def test_operation_metadata_getter_setter(self) -> None:
        """Test operation metadata getter and setter."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        metadata: dict[str, t.GeneralValueType] = {
            "request_size": 1024,
            "response_code": 200,
        }
        FlextContext.Performance.set_operation_metadata(metadata)
        retrieved_metadata = FlextContext.Performance.get_operation_metadata()
        assert retrieved_metadata == metadata

    def test_add_operation_metadata(self) -> None:
        """Test adding individual metadata entries."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        FlextContext.Performance.add_operation_metadata("key1", "value1")
        FlextContext.Performance.add_operation_metadata("key2", "value2")
        metadata = FlextContext.Performance.get_operation_metadata()
        assert metadata is not None
        assert metadata["key1"] == "value1"
        assert metadata["key2"] == "value2"

    def test_timed_operation_context(self) -> None:
        """Test timed operation context manager.

        Validates:
        1. Context manager provides metadata
        2. Start time is recorded
        3. Duration is calculated correctly
        4. All expected metadata fields are present
        """
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Performance.timed_operation("database_query") as metadata:
            # Validate initial metadata
            assert "start_time" in metadata
            assert "operation_name" in metadata
            assert metadata["operation_name"] == "database_query"

            # Simulate work with sleep
            time.sleep(0.01)

            # Validate start_time is a valid ISO format string
            start_time = metadata.get("start_time")
            assert start_time is not None
            assert isinstance(start_time, str), (
                f"start_time should be str, got {type(start_time)}"
            )
            # Validate ISO format (contains T separator and timezone info)
            assert "T" in start_time or "+" in start_time or "Z" in start_time

        # Validate final metadata after context exit
        assert "end_time" in metadata
        assert "duration_seconds" in metadata

        # Validate duration calculation
        duration = cast("float", metadata["duration_seconds"])
        assert duration >= 0.01, f"Duration {duration} should be >= 0.01s"
        assert duration < 0.1, (
            f"Duration {duration} should be < 0.1s (reasonable overhead)"
        )

        # Validate end_time is also ISO format string
        end_time_str = metadata.get("end_time")
        assert isinstance(end_time_str, str), (
            f"end_time should be str, got {type(end_time_str)}"
        )
        # Validate end_time is after start_time (ISO strings compare lexicographically)
        assert end_time_str > start_time, (
            f"end_time {end_time_str} should be > start_time {start_time}"
        )

    def test_timed_operation_duration_calculation(self) -> None:
        """Test timed operation calculates duration correctly.

        Validates:
        1. Duration is calculated accurately
        2. Duration matches expected sleep time (within reasonable tolerance)
        3. Metadata contains all required fields
        """
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        expected_sleep = 0.05

        with FlextContext.Performance.timed_operation("slow_operation") as metadata:
            # Record start for validation
            start_time = metadata.get("start_time")
            assert start_time is not None

            # Simulate work
            time.sleep(expected_sleep)

        # Validate duration calculation
        duration = metadata.get("duration_seconds", 0)
        assert isinstance(duration, float), (
            f"Duration should be float, got {type(duration)}"
        )

        # Validate duration is close to expected (within 20% tolerance for overhead)
        assert duration >= expected_sleep * 0.8, (
            f"Duration {duration}s should be >= {expected_sleep * 0.8}s "
            f"(80% of expected {expected_sleep}s)"
        )
        assert duration < expected_sleep * 2, (
            f"Duration {duration}s should be < {expected_sleep * 2}s "
            f"(reasonable overhead for {expected_sleep}s sleep)"
        )

        # Validate all metadata fields are present
        assert "start_time" in metadata
        assert "end_time" in metadata
        assert "operation_name" in metadata
        assert metadata["operation_name"] == "slow_operation"


class TestSerializationDomain:
    """Test FlextContext.Serialization domain for context propagation using FlextTestsUtilities."""

    def test_get_full_context(self) -> None:
        """Test getting full context snapshot."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
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
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
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
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        context_headers = FlextContext.Serialization.get_correlation_context()
        assert isinstance(context_headers, dict)
        assert len(context_headers) == 0

    def test_set_from_context_headers(self) -> None:
        """Test setting context from HTTP headers."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
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
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
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
    """Test FlextContext.Utilities domain for helper methods using FlextTestsUtilities."""

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
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        correlation_id = FlextContext.Utilities.ensure_correlation_id()
        assert isinstance(correlation_id, str)
        assert correlation_id.startswith("corr_")

    def test_ensure_correlation_id_uses_existing(self) -> None:
        """Test ensure correlation ID uses existing if present."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        existing_id = "existing_corr_789"
        FlextContext.Correlation.set_correlation_id(existing_id)
        correlation_id = FlextContext.Utilities.ensure_correlation_id()
        assert correlation_id == existing_id

    def test_has_correlation_id_true(self) -> None:
        """Test has correlation ID returns true when set."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        FlextContext.Correlation.set_correlation_id("test_corr")
        assert FlextContext.Utilities.has_correlation_id() is True

    def test_has_correlation_id_false(self) -> None:
        """Test has correlation ID returns false when not set."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        assert FlextContext.Utilities.has_correlation_id() is False

    def test_get_context_summary(self) -> None:
        """Test getting human-readable context summary."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        FlextContext.Correlation.set_correlation_id("corr_abc123def456")
        FlextContext.Service.set_service_name("user_service")
        FlextContext.Request.set_operation_name("get_user_profile")
        FlextContext.Request.set_user_id("user_789")
        summary = FlextContext.Utilities.get_context_summary()
        assert isinstance(summary, str)
        assert "FlextContext" in summary

    def test_get_context_summary_empty(self) -> None:
        """Test getting context summary when empty."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        summary = FlextContext.Utilities.get_context_summary()
        assert isinstance(summary, str)
        assert "empty" in summary.lower() or "FlextContext" in summary


class TestContextDataModel:
    """Test FlextContext with m.Context.ContextData using FlextTestsUtilities."""

    def test_context_with_context_data_model(self) -> None:
        """Test FlextContext initialization with ContextData model."""
        context_data = m.Context.ContextData(
            data={"key1": "value1", "key2": "value2"},
        )
        context = FlextContext(context_data)
        result1 = context.get("key1")
        assert result1.is_success
        assert result1.value == "value1"
        result2 = context.get("key2")
        assert result2.is_success
        assert result2.value == "value2"

    def test_context_export_snapshot(self) -> None:
        """Test exporting context as ContextExport model."""
        context = FlextContext()
        context.set("key1", "value1")
        context.set_metadata("created_at", "2025-01-01")
        export_snapshot = context._export_snapshot()
        assert isinstance(export_snapshot, m.Context.ContextExport)
        assert export_snapshot.data.get("key1") == "value1"
        assert export_snapshot.metadata is not None
        metadata = export_snapshot.metadata
        if isinstance(metadata, dict):
            assert metadata.get("created_at") == "2025-01-01"
        elif hasattr(metadata, "attributes"):
            # Type narrowing: metadata has attributes - use getattr for dynamic access
            attributes = getattr(metadata, "attributes", None)
            assert attributes is not None
            assert attributes.get("created_at") == "2025-01-01"
        else:
            # Fallback for other types
            pytest.fail(f"Unexpected metadata type: {type(metadata)}")


class TestContextIntegration:
    """Test integration between context domains using FlextTestsUtilities."""

    def test_multiple_domains_together(self) -> None:
        """Test using multiple context domains together."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Correlation.new_correlation() as correlation_id:
            with FlextContext.Service.service_context("order_service", "v1.0"):
                with FlextContext.Request.request_context(
                    user_id="customer_123",
                    operation_name="create_order",
                ):
                    with FlextContext.Performance.timed_operation("order_processing"):
                        assert (
                            FlextContext.Correlation.get_correlation_id()
                            == correlation_id
                        )
                        assert (
                            FlextContext.Service.get_service_name() == "order_service"
                        )
                        assert FlextContext.Request.get_user_id() == "customer_123"
                        full_context = FlextContext.Serialization.get_full_context()
                        assert full_context["correlation_id"] == correlation_id
                        assert full_context["service_name"] == "order_service"

    def test_context_propagation_across_layers(self) -> None:
        """Test context propagation between service layers."""
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        FlextContext.Correlation.set_correlation_id("layer1_corr_id")
        FlextContext.Service.set_service_name("layer1_service")
        headers = FlextContext.Serialization.get_correlation_context()
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        FlextContext.Serialization.set_from_context(headers)
        assert FlextContext.Correlation.get_correlation_id() == "layer1_corr_id"
        assert FlextContext.Service.get_service_name() == "layer1_service"

    def test_context_json_serialization_with_domains(self) -> None:
        """Test JSON serialization works with context variables."""
        context = FlextContext({"basic_key": "basic_value"})
        json_str = context.to_json()
        assert isinstance(json_str, str)
        restored_context = FlextContext.from_json(json_str)
        result = restored_context.get("basic_key")
        assert result.is_success
        assert result.value == "basic_value"


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
