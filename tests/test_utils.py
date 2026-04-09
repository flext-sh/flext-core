"""Reusable test utilities to eliminate code duplication.

Provides highly automated testing patterns following strict
type-system-architecture.md rules with zero duplication.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import override

from pydantic import BaseModel

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextDispatcher,
    FlextHandlers,
    FlextLogger,
    FlextRegistry,
    FlextSettings,
    d,
    e,
    h,
    r,
    s,
    x,
)
from tests import m, t, u


class TestUtils:
    class TestDataFactory:
        """Factory for creating standardized test data."""

        @staticmethod
        def create_entity_data(
            unique_id: str,
            name: str,
            **kwargs: t.Scalar,
        ) -> t.ConfigurationMapping:
            """Create standardized entity test data."""
            return {"unique_id": unique_id, "name": name, **kwargs}

        @staticmethod
        def create_value_object_data(
            value: t.Scalar,
            **kwargs: t.Scalar,
        ) -> t.ConfigurationMapping:
            """Create standardized value t.NormalizedValue test data."""
            return {"value": value, **kwargs}

        @staticmethod
        def create_operation_test_case(
            operation: str,
            description: str,
            input_data: t.ContainerValueMapping,
            expected_result: t.ContainerValue,
            *,
            expected_success: bool = True,
            error_contains: str | None = None,
        ) -> m.Core.StandardTestCaseModel:
            """Create standardized operation test case."""
            return m.Core.StandardTestCaseModel(
                description=description,
                input_data={"operation": operation, **input_data},
                expected_result=expected_result,
                expected_success=expected_success,
                error_contains=error_contains,
            )

    class AssertionHelpers:
        """Reusable assertion helpers to eliminate duplication."""

        @staticmethod
        def assert_flext_result_success[TResult](
            result: r[TResult],
            context: str = "",
            expected_type: type | None = None,
        ) -> TResult:
            """Assert r success with optional type checking."""
            assert result.is_success, (
                f"{context}: Expected success, got: {result.error}"
            )
            value = result.value
            if expected_type:
                assert isinstance(value, expected_type), (
                    f"{context}: Expected {expected_type.__name__}, got {type(value).__name__}"
                )
            return value

        @staticmethod
        def assert_flext_result_failure[TResult](
            result: r[TResult],
            context: str = "",
            error_contains: str | None = None,
        ) -> str:
            """Assert r failure with optional error checking."""
            assert result.is_failure, (
                f"{context}: Expected failure, got success: {result.value}"
            )
            error_str = str(result.error)
            if error_contains:
                assert error_contains in error_str, (
                    f"{context}: Expected error to contain '{error_contains}', got: {error_str}"
                )
            return error_str

        @staticmethod
        def assert_entity_properties(
            entity: BaseModel,
            expected_props: t.ContainerValueMapping,
            context: str = "",
        ) -> None:
            """Assert entity has expected properties."""
            for prop, expected_value in expected_props.items():
                assert hasattr(entity, prop), (
                    f"{context}: Entity missing property '{prop}'"
                )
                actual_value = getattr(entity, prop)
                assert actual_value == expected_value, (
                    f"{context}: Property '{prop}' expected {expected_value}, got {actual_value}"
                )

        @staticmethod
        def assert_operation_result(
            operation_func: Callable[[], r[t.Container]],
            test_case: m.Core.StandardTestCaseModel,
            context: str = "",
        ) -> t.Container:
            """Execute operation and assert result matches test case."""
            try:
                result = operation_func()
                if test_case.expected_success:
                    actual_result = (
                        TestUtils.AssertionHelpers.assert_flext_result_success(
                            result,
                            f"{context} - {test_case.description}",
                        )
                    )
                    assert actual_result == test_case.expected_result, (
                        f"{context}: Expected {test_case.expected_result}, got {actual_result}"
                    )
                    return actual_result
                return TestUtils.AssertionHelpers.assert_flext_result_failure(
                    result,
                    f"{context} - {test_case.description}",
                    test_case.error_contains,
                )
            except Exception as e:
                if not test_case.expected_success:
                    return str(e)
                raise AssertionError(f"{context}: Unexpected error: {e}") from e

    class TestFixtureFactory:
        """Factory for creating reusable test fixtures."""

        @staticmethod
        def create_test_entity(
            unique_id: str = "test-123",
            name: str = "Test Entity",
        ) -> m.Core.UtilityEntityModel:
            """Create test entity fixture."""
            return m.Core.UtilityEntityModel(
                unique_id=unique_id,
                name=name,
                value=name,
            )

        @staticmethod
        def create_test_value_object(
            value: t.ContainerValue = "test_value",
        ) -> m.Core.UtilityValueModel:
            """Create test value t.NormalizedValue fixture."""
            return m.Core.UtilityValueModel(value=value)

        @staticmethod
        def create_test_container_instance() -> FlextContainer:
            """Create test container fixture."""
            return FlextContainer()

        @staticmethod
        def create_test_context_instance() -> FlextContext:
            """Create test context fixture."""
            return FlextContext()

        @staticmethod
        def create_test_decorators_instance() -> type[d]:
            """Create test decorators fixture."""
            return d

        @staticmethod
        def create_test_dispatcher_instance() -> FlextDispatcher:
            """Create test dispatcher fixture."""
            return FlextDispatcher()

        @staticmethod
        def create_test_exceptions_instance() -> type[e]:
            """Create test exceptions fixture."""
            return e

        @staticmethod
        def create_test_handlers_instance() -> type[
            FlextHandlers[t.ValueOrModel, t.ValueOrModel]
        ]:
            """Create test handlers fixture."""
            return h

        @staticmethod
        def create_test_loggings_instance() -> type[FlextLogger]:
            """Create test loggings fixture."""
            return FlextLogger

        @staticmethod
        def create_test_mixins_instance() -> type[x]:
            """Create test mixins fixture."""
            return x

        @staticmethod
        def create_test_registry_instance() -> FlextRegistry:
            """Create test registry fixture."""
            return FlextRegistry()

        @staticmethod
        def create_test_result_instance() -> type[r[t.Container]]:
            """Create test result fixture."""
            return r

        @staticmethod
        def create_test_runtime_instance() -> type[u]:
            """Create test runtime fixture."""
            return u

        @staticmethod
        def create_test_service_instance() -> s[t.ConfigMap]:
            """Create test service fixture."""

            class TestFlextService(s[t.ConfigMap]):
                """Concrete test service implementation."""

                @override
                def execute(self) -> r[t.ConfigMap]:
                    """Execute test service operation."""
                    return r[t.ConfigMap].ok(
                        t.ConfigMap(root={"result": "test_service_executed"}),
                    )

            return TestFlextService()

        @staticmethod
        def create_test_settings_instance() -> type[FlextSettings]:
            """Create test settings fixture."""
            return FlextSettings

        @staticmethod
        def create_test_utilities_instance() -> type[u]:
            """Create test utilities fixture."""
            return u

        @staticmethod
        def create_test_service_result(
            *,
            success: bool = True,
            value: str | None = None,
            error: str = "Test error",
        ) -> r[str]:
            """Create test service result fixture."""
            if success:
                resolved_value = str(value) if value is not None else "test_value"
                return r[str].ok(resolved_value)
            return r[str].fail(error)


test_data_factory = TestUtils.TestDataFactory()
assertion_helpers = TestUtils.AssertionHelpers()
fixture_factory = TestUtils.TestFixtureFactory()
