"""Coverage tests for currently supported FlextContext APIs."""

from __future__ import annotations

import time

from flext_core import FlextContainer, FlextContext
from flext_tests import tm
from tests import m, t, u

from ..test_utils import assertion_helpers


class TestCoverageContext:
    """Coverage tests for currently supported FlextContext APIs."""

    def test_new_correlation_context(self) -> None:
        u.Tests.clear_context()
        with FlextContext.Correlation.new_correlation() as correlation_id:
            tm.that(correlation_id, is_=str)
            tm.that(FlextContext.Correlation.get_correlation_id(), eq=correlation_id)

    def test_new_correlation_with_explicit_id(self) -> None:
        u.Tests.clear_context()
        explicit_id = "explicit-corr-789"
        with FlextContext.Correlation.new_correlation(
            correlation_id=explicit_id,
        ) as correlation_id:
            tm.that(correlation_id, eq=explicit_id)
            tm.that(FlextContext.Correlation.get_correlation_id(), eq=explicit_id)

    def test_get_service_from_container(self) -> None:
        container = FlextContainer(_context=FlextContext())
        FlextContext.set_container(container)
        test_service_obj = "test_service_value"
        FlextContext.Service.register_service("test_service", test_service_obj)
        result = FlextContext.Service.get_service("test_service")
        _ = u.Tests.assert_success(result)
        tm.that(result.value is test_service_obj, eq=True)

    def test_register_service(self) -> None:
        container = FlextContainer(_context=FlextContext())
        FlextContext.set_container(container)
        service_obj = {"name": "test_service", "version": "1.0"}
        result = FlextContext.Service.register_service("my_service", service_obj)
        _ = u.Tests.assert_success(result)

    def test_get_nonexistent_service(self) -> None:
        container = FlextContainer(_context=FlextContext())
        FlextContext.set_container(container)
        result = FlextContext.Service.get_service("nonexistent_service_xyz")
        _ = assertion_helpers.assert_flext_result_failure(result)

    def test_timed_operation_context(self) -> None:
        u.Tests.clear_context()
        with FlextContext.Performance.timed_operation("database_query") as metadata:
            tm.that(metadata, has="start_time")
            tm.that(metadata, has="operation_name")
            tm.that(metadata["operation_name"], eq="database_query")
            time.sleep(0.01)
            start_time = metadata.get("start_time")
            tm.that(start_time, none=False)
            tm.that(start_time, is_=str)
        tm.that(metadata, has="end_time")
        tm.that(metadata, has="duration_seconds")
        duration_value = metadata["duration_seconds"]
        tm.that(duration_value, is_=float)
        duration_float: float = (
            float(duration_value) if isinstance(duration_value, (int, float)) else 0.0
        )
        tm.that(duration_float, gte=0.01)
        tm.that(duration_float, lt=0.1)

    def test_timed_operation_duration_calculation(self) -> None:
        u.Tests.clear_context()
        expected_sleep = 0.05
        with FlextContext.Performance.timed_operation("slow_operation") as metadata:
            start_time = metadata.get("start_time")
            tm.that(start_time, none=False)
            time.sleep(expected_sleep)
        duration_value = metadata.get("duration_seconds", 0)
        tm.that(duration_value, is_=float)
        if isinstance(duration_value, (int, float)):
            duration_float: float = float(duration_value)
        else:
            duration_float = 0.0
        tm.that(duration_float, gte=expected_sleep * 0.8)
        tm.that(duration_float, lt=expected_sleep * 2)
        tm.that(metadata, has="start_time")
        tm.that(metadata, has="end_time")
        tm.that(metadata, has="operation_name")
        tm.that(metadata["operation_name"], eq="slow_operation")

    def test_ensure_correlation_id_creates_if_missing(self) -> None:
        u.Tests.clear_context()
        correlation_id = FlextContext.Utilities.ensure_correlation_id()
        tm.that(correlation_id, is_=str)

    def test_ensure_correlation_id_uses_existing(self) -> None:
        u.Tests.clear_context()
        existing_id = "existing_corr_789"
        FlextContext.Correlation.set_correlation_id(existing_id)
        correlation_id = FlextContext.Utilities.ensure_correlation_id()
        tm.that(correlation_id, eq=existing_id)

    def test_context_with_context_data_model(self) -> None:
        context_data = m.ContextData(
            data=t.Dict(root={"key1": "value1", "key2": "value2"}),
        )
        context = FlextContext(initial_data=context_data)
        result1 = context.get("key1")
        tm.ok(result1)
        tm.that(result1.value, eq="value1")
        result2 = context.get("key2")
        tm.ok(result2)
        tm.that(result2.value, eq="value2")


__all__ = ["TestCoverageContext"]
