"""Coverage tests for currently supported FlextContext APIs."""

from __future__ import annotations

import time

from flext_core import FlextContainer, FlextContext, m, t
from flext_tests import FlextTestsUtilities, u

from ..test_utils import assertion_helpers


class TestCorrelationDomain:
    """Correlation domain behavior tests."""

    def test_new_correlation_context(self) -> None:
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Correlation.new_correlation() as correlation_id:
            assert isinstance(correlation_id, str)
            assert FlextContext.Correlation.get_correlation_id() == correlation_id

    def test_new_correlation_with_explicit_id(self) -> None:
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        explicit_id = "explicit-corr-789"
        with FlextContext.Correlation.new_correlation(
            correlation_id=explicit_id,
        ) as correlation_id:
            assert correlation_id == explicit_id
            assert FlextContext.Correlation.get_correlation_id() == explicit_id


class TestServiceDomain:
    """Service domain behavior tests."""

    def test_get_service_from_container(self) -> None:
        container = FlextContainer(_context=FlextContext())
        FlextContext.set_container(container)
        test_service_obj: t.GeneralValueType = "test_service_value"
        FlextContext.Service.register_service("test_service", test_service_obj)
        result = FlextContext.Service.get_service("test_service")
        u.Tests.Result.assert_success(result)
        assert result.value is test_service_obj

    def test_register_service(self) -> None:
        container = FlextContainer(_context=FlextContext())
        FlextContext.set_container(container)
        service_obj = {"name": "test_service", "version": "1.0"}
        result = FlextContext.Service.register_service("my_service", service_obj)
        u.Tests.Result.assert_success(result)

    def test_get_nonexistent_service(self) -> None:
        container = FlextContainer(_context=FlextContext())
        FlextContext.set_container(container)
        result = FlextContext.Service.get_service("nonexistent_service_xyz")
        assertion_helpers.assert_flext_result_failure(result)


class TestPerformanceDomain:
    """Performance domain behavior tests."""

    def test_timed_operation_context(self) -> None:
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        with FlextContext.Performance.timed_operation("database_query") as metadata:
            assert "start_time" in metadata
            assert "operation_name" in metadata
            assert metadata["operation_name"] == "database_query"
            time.sleep(0.01)
            start_time = metadata.get("start_time")
            assert start_time is not None
            assert isinstance(start_time, str)

        assert "end_time" in metadata
        assert "duration_seconds" in metadata
        duration_value = metadata["duration_seconds"]
        assert isinstance(duration_value, float)
        duration = duration_value
        assert duration >= 0.01
        assert duration < 0.1

    def test_timed_operation_duration_calculation(self) -> None:
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        expected_sleep = 0.05

        with FlextContext.Performance.timed_operation("slow_operation") as metadata:
            start_time = metadata.get("start_time")
            assert start_time is not None
            time.sleep(expected_sleep)

        duration = metadata.get("duration_seconds", 0)
        assert isinstance(duration, float)
        assert duration >= expected_sleep * 0.8
        assert duration < expected_sleep * 2
        assert "start_time" in metadata
        assert "end_time" in metadata
        assert "operation_name" in metadata
        assert metadata["operation_name"] == "slow_operation"


class TestUtilitiesDomain:
    """Utilities domain behavior tests."""

    def test_ensure_correlation_id_creates_if_missing(self) -> None:
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        correlation_id = FlextContext.Utilities.ensure_correlation_id()
        assert isinstance(correlation_id, str)

    def test_ensure_correlation_id_uses_existing(self) -> None:
        FlextTestsUtilities.Tests.ContextHelpers.clear_context()
        existing_id = "existing_corr_789"
        FlextContext.Correlation.set_correlation_id(existing_id)
        correlation_id = FlextContext.Utilities.ensure_correlation_id()
        assert correlation_id == existing_id


class TestContextDataModel:
    """Context data model initialization tests."""

    def test_context_with_context_data_model(self) -> None:
        context_data = m.ContextData(
            data=t.Dict(root={"key1": "value1", "key2": "value2"}),
        )
        context = FlextContext(context_data)
        result1 = context.get("key1")
        assert result1.is_success
        assert result1.value == "value1"
        result2 = context.get("key2")
        assert result2.is_success
        assert result2.value == "value2"


__all__ = [
    "TestContextDataModel",
    "TestCorrelationDomain",
    "TestPerformanceDomain",
    "TestServiceDomain",
    "TestUtilitiesDomain",
]
