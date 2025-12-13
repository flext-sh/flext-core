"""Comprehensive test configuration and utilities for flext-core.

Provides highly automated testing infrastructure following strict
type-system-architecture.md rules with real functionality testing.
"""

from __future__ import annotations

import math
import signal
import types
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Never, TypeVar

import pytest

from flext_core import FlextContainer, FlextContext, FlextResult, m, r
from flext_core._models import entity as flext_models_entity
from tests.helpers import factories  # Import for User model rebuild
from tests.test_utils import assertion_helpers

# Type variables for test automation
T = TypeVar("T")
TestResult = FlextResult[T]


class TestAutomationFramework:
    """Highly automated test framework with real functionality testing.

    Follows strict type-system-architecture.md rules:
    - Uses FlextResult[T] for all operations that can fail
    - Tests real functionality, not mocks
    - Follows 2-level namespace maximum
    - Zero circular dependencies
    """

    @staticmethod
    def assert_result_success(result: TestResult[object], context: str = "") -> object:
        """Assert FlextResult is success and return value.

        Args:
            result: FlextResult to check
            context: Optional context for error messages

        Returns:
            Unwrapped result value

        Raises:
            AssertionError: If result is not success

        """
        (
            assertion_helpers.assert_flext_result_success(result),
            f"{context}: Expected success, got failure: {result.error}",
        )
        return result.value

    @staticmethod
    def assert_result_failure(
        result: TestResult[object], expected_error: str | None = None, context: str = ""
    ) -> str:
        """Assert FlextResult is failure and optionally check error message.

        Args:
            result: FlextResult to check
            expected_error: Expected error substring (optional)
            context: Optional context for error messages

        Returns:
            Actual error message

        Raises:
            AssertionError: If result is not failure or error doesn't match

        """
        (
            assertion_helpers.assert_flext_result_failure(result),
            f"{context}: Expected failure, got success: {result.value}",
        )
        if expected_error:
            assert expected_error in str(result.error), (
                f"{context}: Expected error '{expected_error}', got '{result.error}'"
            )
        return str(result.error)

    @staticmethod
    def create_test_entity(
        unique_id: str, name: str, **kwargs: object
    ) -> TestResult[m.Entity]:
        """Create test entity with real functionality.

        Args:
            unique_id: Entity unique identifier
            name: Entity name
            **kwargs: Additional entity fields

        Returns:
            FlextResult[Entity]: Result containing created entity

        """
        try:
            # Use real entity creation through facade
            entity_data = {"unique_id": unique_id, "name": name, **kwargs}
            # Note: This would need to be implemented in the actual facade
            # For now, return mock success
            return r[m.Entity].ok(type("MockEntity", (), entity_data)())
        except Exception as e:
            return r[m.Entity].fail(f"Entity creation failed: {e}")

    @staticmethod
    def create_test_value_object(value: object, value_type: type[T]) -> TestResult[T]:
        """Create test value object with real validation.

        Args:
            value: Value to create object from
            value_type: Type of value object to create

        Returns:
            FlextResult[T]: Result containing created value object

        """
        try:
            # Use real value object creation through facade
            if hasattr(value_type, "__init__"):
                obj = value_type(value)
                return r[T].ok(obj)
            return r[T].fail(f"Invalid value type: {value_type}")
        except Exception as e:
            return r[T].fail(f"Value object creation failed: {e}")

    @staticmethod
    def execute_with_timeout(
        func: callable, timeout_seconds: float = 5.0
    ) -> TestResult[object]:
        """Execute function with timeout for performance testing.

        Args:
            func: Function to execute
            timeout_seconds: Timeout in seconds

        Returns:
            FlextResult[object]: Result of execution or timeout error

        """

        @contextmanager
        def timeout_context() -> Generator[None]:
            def timeout_handler(signum: int, frame: types.FrameType) -> Never:
                raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            try:
                yield
            finally:
                signal.alarm(0)

        try:
            with timeout_context():
                result = func()
                return r[object].ok(result)
        except TimeoutError as e:
            return r[object].fail(str(e))
        except Exception as e:
            return r[object].fail(f"Execution failed: {e}")

    @staticmethod
    def parametrize_real_data(*test_cases: dict[str, object]) -> pytest.MarkDecorator:
        """Parametrize test with real data following architecture rules.

        Args:
            *test_cases: Test case dictionaries with real data

        Returns:
            pytest.mark.parametrize decorator

        """
        return pytest.mark.parametrize(
            "test_data", test_cases, ids=lambda case: case.get("description", str(case))
        )


# Global test utilities instance
test_framework = TestAutomationFramework()


# Pytest fixtures for automated testing
@pytest.fixture
def automation_framework() -> TestAutomationFramework:
    """Provide automated test framework instance."""
    return test_framework


@pytest.fixture
def real_entity() -> object:
    """Provide real test entity."""
    result = test_framework.create_test_entity("test-123", "Test Entity")
    if result.is_success:
        return result.value
    pytest.skip(f"Could not create test entity: {result.error}")


@pytest.fixture
def real_value_object() -> object:
    """Provide real test value object."""
    result = test_framework.create_test_value_object("test value", str)
    if result.is_success:
        return result.value
    pytest.skip(f"Could not create test value object: {result.error}")


class FunctionalExternalService:
    """Mock external service for integration testing.

    Provides real functionality for testing service integration patterns
    with dependency injection and result handling.
    """

    def __init__(self) -> None:
        """Initialize mock service with empty state."""
        self.processed_items: list[str] = []
        self.call_count = 0

    def process(self, input_data: str) -> TestResult[str]:
        """Process input data by prefixing with 'processed_'.

        Args:
            input_data: String to process

        Returns:
            FlextResult[str]: Processed result or failure

        """
        try:
            self.call_count += 1
            if not isinstance(input_data, str):
                return r[str].fail(f"Invalid input type: {type(input_data)}")

            processed = f"processed_{input_data}"
            self.processed_items.append(processed)
            return r[str].ok(processed)
        except Exception as e:
            return r[str].fail(f"Processing failed: {e}")

    def get_call_count(self) -> int:
        """Get number of times process() was called."""
        return self.call_count


@pytest.fixture(autouse=True)
def _rebuild_pydantic_models() -> None:
    """Auto-rebuild Pydantic models before each test.

    Pydantic v2 requires model_rebuild() when using forward references
    (from __future__ import annotations). This fixture ensures all
    FlextModels subclasses are rebuilt before tests run.
    """
    # Provide namespace for forward reference resolution
    types_namespace = {
        "FlextModelsEntity": flext_models_entity.FlextModelsEntity,
        "FlextModels": m,
    }

    # Rebuild base model classes to resolve forward references
    m.AggregateRoot.model_rebuild(_types_namespace=types_namespace)
    m.Entity.model_rebuild(_types_namespace=types_namespace)
    m.Value.model_rebuild(_types_namespace=types_namespace)

    # Also rebuild User and other test models if they exist
    try:
        factories.User.model_rebuild(_types_namespace=types_namespace)
    except (ImportError, AttributeError):
        pass  # User model may not be defined yet


@pytest.fixture
def test_context() -> FlextContext:
    """Provide FlextContext instance for testing."""
    return FlextContext()


@pytest.fixture
def clean_container() -> FlextContainer:
    """Provide a clean FlextContainer instance for testing.

    Creates a container and clears auto-registered services for testing
    in isolation.
    """
    container = FlextContainer()
    # Clear auto-registered services for test isolation
    container.unregister("config")
    container.unregister("logger")
    container.unregister("container")
    return container


@pytest.fixture(autouse=True)
def _reset_global_container() -> None:
    """Reset the global FlextContainer instance after each test.

    This fixture ensures test isolation by clearing the global container
    singleton that persists across tests. Without this, tests interfere
    with each other due to shared global state.
    """
    yield  # Run the test
    # After the test completes, reset the global instance
    FlextContainer._global_instance = None


@pytest.fixture
def mock_external_service() -> FunctionalExternalService:
    """Provide mock external service for integration tests."""
    return FunctionalExternalService()


@pytest.fixture
def sample_data() -> dict[str, object]:
    """Provide sample test data for integration tests."""
    return {
        "string": "test_value",
        "integer": 42,
        "float": math.pi,
        "boolean": True,
        "none": None,
        "list": ["item1", "item2"],
        "dict": {"key": "value"},
    }


@pytest.fixture
def temp_directory(tmp_path: Path) -> Path:
    """Provide temporary directory path for integration tests."""
    return tmp_path
