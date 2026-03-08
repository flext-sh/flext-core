"""Reusable test utilities to eliminate code duplication.

Provides highly automated testing patterns following strict
type-system-architecture.md rules with zero duplication.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, override

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextDecorators,
    FlextDispatcher,
    FlextExceptions,
    FlextHandlers,
    FlextLogger,
    FlextMixins,
    FlextRegistry,
    FlextResult,
    FlextRuntime,
    FlextService,
    FlextSettings,
    FlextUtilities,
    m,
    r,
    t,
)

from .models import StandardTestCaseModel, UtilityEntityModel, UtilityValueModel

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
TestResult = FlextResult[T]
TestResultCo = FlextResult[T_co]
type StandardTestCase = StandardTestCaseModel


class TestDataFactory:
    """Factory for creating standardized test data."""

    @staticmethod
    def create_entity_data(
        unique_id: str, name: str, **kwargs: t.ContainerValue
    ) -> t.ConfigurationMapping:
        """Create standardized entity test data."""
        return {"unique_id": unique_id, "name": name, **kwargs}

    @staticmethod
    def create_value_object_data(
        value: t.ContainerValue, **kwargs: t.ContainerValue
    ) -> t.ConfigurationMapping:
        """Create standardized value object test data."""
        return {"value": value, **kwargs}

    @staticmethod
    def create_operation_test_case(
        operation: str,
        description: str,
        input_data: t.ConfigurationMapping,
        expected_result: t.ContainerValue,
        *,
        expected_success: bool = True,
        error_contains: str | None = None,
    ) -> StandardTestCaseModel:
        """Create standardized operation test case."""
        return StandardTestCaseModel(
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
        result: FlextResult[TResult],
        context: str = "",
        expected_type: type | None = None,
    ) -> TResult:
        """Assert FlextResult success with optional type checking."""
        assert result.is_success, f"{context}: Expected success, got: {result.error}"
        value = result.value
        if expected_type:
            assert isinstance(value, expected_type), (
                f"{context}: Expected {expected_type.__name__}, got {type(value).__name__}"
            )
        return value

    @staticmethod
    def assert_flext_result_failure[TResult](
        result: FlextResult[TResult],
        context: str = "",
        error_contains: str | None = None,
    ) -> str:
        """Assert FlextResult failure with optional error checking."""
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
        entity: m.Entity, expected_props: t.ConfigurationMapping, context: str = ""
    ) -> None:
        """Assert entity has expected properties."""
        for prop, expected_value in expected_props.items():
            assert hasattr(entity, prop), f"{context}: Entity missing property '{prop}'"
            actual_value = getattr(entity, prop)
            assert actual_value == expected_value, (
                f"{context}: Property '{prop}' expected {expected_value}, got {actual_value}"
            )

    @staticmethod
    def assert_operation_result(
        operation_func: Callable[[], FlextResult[t.ContainerValue]],
        test_case: StandardTestCase,
        context: str = "",
    ) -> t.ContainerValue:
        """Execute operation and assert result matches test case."""
        try:
            result = operation_func()
            if test_case.expected_success:
                actual_result = AssertionHelpers.assert_flext_result_success(
                    result, f"{context} - {test_case.description}"
                )
                assert actual_result == test_case.expected_result, (
                    f"{context}: Expected {test_case.expected_result}, got {actual_result}"
                )
                return actual_result
            return AssertionHelpers.assert_flext_result_failure(
                result, f"{context} - {test_case.description}", test_case.error_contains
            )
        except Exception as e:
            if not test_case.expected_success:
                return str(e)
            raise AssertionError(f"{context}: Unexpected error: {e}") from e


class TestFixtureFactory:
    """Factory for creating reusable test fixtures."""

    @staticmethod
    def create_test_entity(
        unique_id: str = "test-123", name: str = "Test Entity"
    ) -> UtilityEntityModel:
        """Create test entity fixture."""
        return UtilityEntityModel(unique_id=unique_id, name=name, value=name)

    @staticmethod
    def create_test_value_object(
        value: t.ContainerValue = "test_value",
    ) -> UtilityValueModel:
        """Create test value object fixture."""
        return UtilityValueModel(value=value)

    @staticmethod
    def create_test_container_instance() -> FlextContainer:
        """Create test container fixture."""
        return FlextContainer()

    @staticmethod
    def create_test_context_instance() -> FlextContext:
        """Create test context fixture."""
        return FlextContext()

    @staticmethod
    def create_test_decorators_instance() -> type[FlextDecorators]:
        """Create test decorators fixture."""
        return FlextDecorators

    @staticmethod
    def create_test_dispatcher_instance() -> FlextDispatcher:
        """Create test dispatcher fixture."""
        return FlextDispatcher()

    @staticmethod
    def create_test_exceptions_instance() -> type[FlextExceptions]:
        """Create test exceptions fixture."""
        return FlextExceptions

    @staticmethod
    def create_test_handlers_instance() -> type[
        FlextHandlers[t.ContainerValue, t.ContainerValue]
    ]:
        """Create test handlers fixture."""
        return FlextHandlers

    @staticmethod
    def create_test_loggings_instance() -> type[FlextLogger]:
        """Create test loggings fixture."""
        return FlextLogger

    @staticmethod
    def create_test_mixins_instance() -> type[FlextMixins]:
        """Create test mixins fixture."""
        return FlextMixins

    @staticmethod
    def create_test_registry_instance() -> FlextRegistry:
        """Create test registry fixture."""
        return FlextRegistry()

    @staticmethod
    def create_test_result_instance() -> type[FlextResult[t.ContainerValue]]:
        """Create test result fixture."""
        return FlextResult

    @staticmethod
    def create_test_runtime_instance() -> type[FlextRuntime]:
        """Create test runtime fixture."""
        return FlextRuntime

    @staticmethod
    def create_test_service_instance() -> FlextService[m.ConfigMap]:
        """Create test service fixture."""

        class TestFlextService(FlextService[m.ConfigMap]):
            """Concrete test service implementation."""

            @override
            def execute(self) -> r[m.ConfigMap]:
                """Execute test service operation."""
                return r[m.ConfigMap].ok(
                    m.ConfigMap(root={"result": "test_service_executed"})
                )

        return TestFlextService()

    @staticmethod
    def create_test_settings_instance() -> type[FlextSettings]:
        """Create test settings fixture."""
        return FlextSettings

    @staticmethod
    def create_test_utilities_instance() -> type[FlextUtilities]:
        """Create test utilities fixture."""
        return FlextUtilities

    @staticmethod
    def create_test_service_result(
        *,
        success: bool = True,
        value: t.ContainerValue | None = None,
        error: str = "Test error",
    ) -> TestResult[t.ContainerValue]:
        """Create test service result fixture."""
        if success:
            return r[t.ContainerValue].ok(value if value is not None else "test_value")
        return r[t.ContainerValue].fail(error)


test_data_factory = TestDataFactory()
assertion_helpers = AssertionHelpers()
fixture_factory = TestFixtureFactory()
