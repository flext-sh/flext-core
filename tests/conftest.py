# mypy: disable-error-code=misc
"""Comprehensive test configuration and utilities for flext-core.

Provides highly automated testing infrastructure following strict
type-system-architecture.md rules with real functionality testing.
"""

from __future__ import annotations

import math
import queue
import tempfile
import threading
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TypeVar, cast

import pytest
from flext_core import (
    FlextContainer,
    FlextContext,
    FlextResult,
    FlextSettings,
    m,
    r,
    t,
)
from pydantic import ConfigDict

from tests.helpers.scenarios import (
    ParserScenario,
    ParserScenarios,
    ReliabilityScenario,
    ReliabilityScenarios,
    ValidationScenario,
    ValidationScenarios,
)
from tests.test_utils import assertion_helpers

# Type variables for test automation
T = TypeVar("T")
TestResult = FlextResult[T]


class FlextTestAutomationFramework:
    """Highly automated test framework with real functionality testing.

    Follows strict type-system-architecture.md rules:
    - Uses FlextResult[T] for all operations that can fail
    - Tests real functionality, not mocks
    - Follows 2-level namespace maximum
    - Zero circular dependencies
    """

    @staticmethod
    def assert_result_success[TResult](
        result: FlextResult[TResult],
        context: str = "",
    ) -> TResult:
        """Assert FlextResult is success and return value.

        Args:
            result: FlextResult to check
            context: Optional context for error messages

        Returns:
            Unwrapped result value

        Raises:
            AssertionError: If result is not success

        """
        assertion_helpers.assert_flext_result_success(
            result,
            f"{context}: Expected success, got failure: {result.error}",
        )
        return result.value

    @staticmethod
    def assert_result_failure[TResult](
        result: FlextResult[TResult],
        expected_error: str | None = None,
        context: str = "",
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
        assertion_helpers.assert_flext_result_failure(
            result,
            f"{context}: Expected failure, got success: {result.value}",
            error_contains=expected_error,
        )
        if expected_error:
            assert expected_error in str(result.error), (
                f"{context}: Expected error '{expected_error}', got '{result.error}'"
            )
        return str(result.error)

    @staticmethod
    def create_test_entity(
        unique_id: str,
        name: str,
        **kwargs: t.GeneralValueType,
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

            class TestEntity(m.Entity):
                model_config = ConfigDict(extra="allow")
                name: str

            entity = TestEntity.model_validate({
                "unique_id": unique_id,
                "name": name,
                **kwargs,
            })
            return r[m.Entity].ok(entity)
        except Exception as e:
            return r[m.Entity].fail(f"Entity creation failed: {e}")

    @staticmethod
    def create_test_value_object(
        value: object,
        value_type: Callable[[object], T],
    ) -> TestResult[T]:
        """Create test value object with real validation.

        Args:
            value: Value to create object from
            value_type: Type of value object to create

        Returns:
            FlextResult[T]: Result containing created value object

        """
        try:
            obj = value_type(value)
            return r[T].ok(obj)
        except Exception as e:
            return r[T].fail(f"Value object creation failed: {e}")

    @staticmethod
    def execute_with_timeout(
        func: Callable[..., object],
        timeout_seconds: float = 5.0,
    ) -> TestResult[object]:
        """Execute function with timeout for performance testing.

        Args:
            func: Function to execute
            timeout_seconds: Timeout in seconds

        Returns:
            FlextResult[object]: Result of execution or timeout error

        """
        result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def _target() -> None:
            try:
                result_queue.put((True, func()))
            except Exception as error:
                result_queue.put((False, error))

        worker = threading.Thread(target=_target, daemon=True)
        try:
            worker.start()
            worker.join(timeout=timeout_seconds)
            if worker.is_alive():
                return r[object].fail(f"Operation timed out after {timeout_seconds}s")

            succeeded, payload = result_queue.get_nowait()
            if succeeded:
                return r[object].ok(payload)

            return r[object].fail(f"Execution failed: {payload}")
        except queue.Empty:
            return r[object].fail("Execution failed: no result produced")

    @staticmethod
    def parametrize_real_data(
        *test_cases: dict[str, t.GeneralValueType],
    ) -> pytest.MarkDecorator:
        """Parametrize test with real data following architecture rules.

        Args:
            *test_cases: Test case dictionaries with real data

        Returns:
            pytest.mark.parametrize decorator

        """
        return pytest.mark.parametrize(
            "test_data",
            test_cases,
            ids=lambda case: case.get("description", str(case)),
        )


# Global test utilities instance
test_framework = FlextTestAutomationFramework()


class FlextScenarioRunner:
    """Helper for executing parametrized scenario tests.

    Provides common pattern for validation scenario execution:
    1. Get scenario from parametrize
    2. Execute validator with scenario parameters
    3. Assert result matches expected outcome
    """

    @staticmethod
    def execute_validation_scenario(
        validator_func: Callable[..., FlextResult[object]],
        scenario: ValidationScenario,
    ) -> FlextResult[object]:
        """Execute validation scenario and return result.

        Args:
            validator_func: The validation function to call
            scenario: ValidationScenario dataclass instance

        Returns:
            FlextResult with validator output

        """
        try:
            if hasattr(scenario, "input_params") and scenario.input_params:
                result = validator_func(scenario.input_value, **scenario.input_params)
            else:
                result = validator_func(scenario.input_value)
            return result
        except Exception as e:
            return FlextResult.fail(str(e))

    @staticmethod
    def execute_parser_scenario(
        parser_func: Callable[[str], FlextResult[object]],
        scenario: ParserScenario,
    ) -> FlextResult[object]:
        """Execute parser scenario and return result.

        Args:
            parser_func: The parser function to call
            scenario: ParserScenario dataclass instance

        Returns:
            FlextResult with parser output

        """
        try:
            return parser_func(scenario.input_data)
        except Exception as e:
            return FlextResult.fail(str(e))

    @staticmethod
    def assert_scenario_result(
        result: FlextResult[object],
        scenario: ValidationScenario | ParserScenario | ReliabilityScenario,
        context: str = "",
    ) -> None:
        """Assert scenario result matches expected outcome.

        Args:
            result: FlextResult from scenario execution
            scenario: Scenario dataclass with expected values
            context: Optional context for error messages

        """
        expected_value = getattr(scenario, "expected_value", None)
        expected_error_contains = getattr(
            scenario,
            "expected_error_contains",
            getattr(scenario, "error_contains", None),
        )

        if scenario.should_succeed:
            assert result.is_success, (
                f"{context}: Expected success, got failure: {result.error}"
            )
            if expected_value is not None:
                assert result.value == expected_value, (
                    f"{context}: Expected {expected_value}, got {result.value}"
                )
        else:
            assert result.is_failure, (
                f"{context}: Expected failure, got success: {result.value}"
            )
            if expected_error_contains:
                assert expected_error_contains in str(result.error), (
                    f"{context}: Expected error containing "
                    f"'{expected_error_contains}', "
                    f"got '{result.error}'"
                )


class FlextResultAssertionHelper:
    """Helper for common FlextResult assertions.

    Provides pattern for asserting result success/failure with
    optional error message matching.
    """

    @staticmethod
    def assert_success(
        result: FlextResult[object],
        expected_value: object = None,
        context: str = "",
    ) -> object:
        """Assert result is success.

        Args:
            result: FlextResult to check
            expected_value: Optional expected value (checked if provided)
            context: Optional context for error messages

        """
        assert result.is_success, (
            f"{context}: Expected success, got failure: {result.error}"
        )
        if expected_value is not None:
            assert result.value == expected_value, (
                f"{context}: Expected {expected_value}, got {result.value}"
            )
        return result.value

    @staticmethod
    def assert_failure(
        result: FlextResult[object],
        expected_error: str | None = None,
        context: str = "",
    ) -> str:
        """Assert result is failure.

        Args:
            result: FlextResult to check
            expected_error: Optional expected error substring
            context: Optional context for error messages

        """
        assert result.is_failure, (
            f"{context}: Expected failure, got success: {result.value}"
        )
        if expected_error:
            assert expected_error in str(result.error), (
                f"{context}: Expected error containing "
                f"'{expected_error}', got '{result.error}'"
            )
        return str(result.error)


class FlextConsolidationContext:
    """Context for test consolidation operations.

    Provides state tracking and validation for consolidation workflows:
    - Tracking parametrized scenario counts
    - Validating test deduplication
    - Monitoring coverage improvement
    """

    def __init__(self) -> None:
        """Initialize consolidation context with empty state."""
        self.scenarios_tested: dict[str, int] = {}
        self.tests_executed = 0
        self.unique_code_paths: set[str] = set()

    def record_scenario(self, scenario_name: str, scenario_type: str) -> None:
        """Record a scenario test execution."""
        key = f"{scenario_type}:{scenario_name}"
        if key not in self.scenarios_tested:
            self.scenarios_tested[key] = 0
        self.scenarios_tested[key] += 1

    def record_test_execution(self) -> None:
        """Record a test execution."""
        self.tests_executed += 1

    def record_code_path(self, code_path: str) -> None:
        """Record a unique code path executed."""
        self.unique_code_paths.add(code_path)

    def get_deduplication_stats(self) -> dict[str, int]:
        """Get statistics on test deduplication.

        Returns:
            dict with test count and unique scenario count

        """
        return {
            "total_scenarios": len(self.scenarios_tested),
            "total_tests_executed": self.tests_executed,
            "unique_code_paths": len(self.unique_code_paths),
        }


# Pytest fixtures for automated testing
@pytest.fixture
def automation_framework() -> FlextTestAutomationFramework:
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
        """Initialize external service with empty state."""
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


# NOTE: Forward references resolved through proper imports instead of model_rebuild
# @pytest.fixture(autouse=True)
# def _rebuild_pydantic_models() -> None:
#     """Auto-rebuild Pydantic models before each test.
#
#     Pydantic v2 requires model_rebuild() when using forward references
#     (from __future__ import annotations). This fixture ensures all
#     FlextModels subclasses are rebuilt before tests run.
#     """
#     # Provide namespace for forward reference resolution
#     types_namespace = {
#         "FlextModelsEntity": flext_models_entity.FlextModelsEntity,
#         "FlextModels": m,
#     }
#
#     # NOTE: model_rebuild calls removed to comply with FLEXT standards
#     # Forward references should be resolved through proper imports instead


@pytest.fixture
def test_context() -> FlextContext:
    """Provide FlextContext instance for testing."""
    return FlextContext()


@pytest.fixture
def clean_container() -> FlextContainer:
    """Provide a clean FlextContainer instance for testing.

    Creates a container and clears all registered services for testing
    in isolation regardless of what other tests may have registered.
    """
    container = FlextContainer()
    container.clear_all()
    return container


@pytest.fixture(autouse=True)
def _reset_global_container() -> Generator[None]:
    """Reset the global FlextContainer and FlextSettings instances after each test.

    This fixture ensures test isolation by clearing the global singletons
    that persist across tests. Without this, tests interfere with each other
    due to shared global state.
    """
    yield  # Run the test
    # After the test completes, reset the global instances
    FlextContainer._global_instance = cast("FlextContainer | None", None)
    FlextSettings.reset_global_instance()


@pytest.fixture
def mock_external_service() -> FunctionalExternalService:
    """Provide mock external service for integration tests."""
    return FunctionalExternalService()


@pytest.fixture
def sample_data() -> dict[str, t.GeneralValueType]:
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


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Temporary directory fixture available to all FLEXT projects."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Temporary file fixture available to all FLEXT projects."""
    return temp_dir / "test_file.txt"


@pytest.fixture
def flext_result_success() -> r[dict[str, object]]:
    """Successful FlextResult fixture available to all FLEXT projects."""
    return r[dict[str, object]].ok({"success": True})


@pytest.fixture
def flext_result_failure() -> r[object]:
    """Failed FlextResult fixture available to all FLEXT projects."""
    return r[object].fail("Test error")


# =========================================================================
# Advanced Fixtures for Test Consolidation and Scenario Testing
# =========================================================================


@pytest.fixture
def validation_scenarios() -> type[ValidationScenarios]:
    """Access to all centralized validation scenarios."""
    return ValidationScenarios


@pytest.fixture
def parser_scenarios() -> type[ParserScenarios]:
    """Access to all centralized parser scenarios."""
    return ParserScenarios


@pytest.fixture
def reliability_scenarios() -> type[ReliabilityScenarios]:
    """Access to all centralized reliability scenarios."""
    return ReliabilityScenarios


@pytest.fixture
def scenario_runner() -> FlextScenarioRunner:
    """Helper for executing parametrized scenario tests."""
    return FlextScenarioRunner()


@pytest.fixture
def result_assertion_helper() -> FlextResultAssertionHelper:
    """Helper for common FlextResult assertions."""
    return FlextResultAssertionHelper()


@pytest.fixture
def consolidation_context() -> FlextConsolidationContext:
    """Context for test consolidation operations."""
    return FlextConsolidationContext()


# =========================================================================
# PYDANTIC V2 VALIDATION FIXTURES - Test Data for Annotated Types
# =========================================================================
# Comprehensive test fixtures for validating Pydantic v2 patterns.
# Covers: PortNumber, UriString, HostnameStr, and string/numeric validators.


@pytest.fixture
def valid_port_numbers() -> list[int]:
    """Valid port numbers for PortNumber validation (1-65535)."""
    return [
        1,  # Minimum valid port
        80,  # HTTP
        443,  # HTTPS
        8080,  # Common development port
        3306,  # MySQL
        5432,  # PostgreSQL
        27017,  # MongoDB
        65535,  # Maximum valid port
    ]


@pytest.fixture
def invalid_port_numbers() -> list[int]:
    """Invalid port numbers for PortNumber validation."""
    return [
        0,  # Below minimum
        -1,  # Negative
        -8080,  # Negative port
        65536,  # Above maximum
        100000,  # Way above maximum
    ]


@pytest.fixture
def valid_uris() -> list[str]:
    """Valid URIs for UriString validation."""
    return [
        "http://localhost",
        "https://example.com",
        "https://example.com:8080",
        "https://example.com/path",
        "https://example.com/path?query=value",
        "https://user:pass@example.com",
        "ftp://files.example.com",
        "grpc://service:50051",
        "postgresql://localhost:5432/db",
        "mongodb://localhost:27017/db",
    ]


@pytest.fixture
def invalid_uris() -> list[str]:
    """Invalid URIs for UriString validation."""
    return [
        "",  # Empty string
        "   ",  # Whitespace only
        "localhost",  # Missing scheme
        "example.com",  # Missing scheme
        "://example.com",  # Missing scheme name
        "http://",  # Missing host
        "http://:8080",  # Missing host
    ]


@pytest.fixture
def valid_hostnames() -> list[str]:
    """Valid hostnames for HostnameStr validation."""
    return [
        "localhost",
        "example.com",
        "sub.example.com",
        "my-server",
        "server-01",
        "api-gateway-v2",
        "db.prod.internal",
        "a",  # Single character
        "a.b",  # Minimal subdomain
    ]


@pytest.fixture
def invalid_hostnames() -> list[str]:
    """Invalid hostnames for HostnameStr validation."""
    return [
        "",  # Empty string
        "   ",  # Whitespace only
        "-invalid",  # Starts with hyphen
        "invalid-",  # Ends with hyphen
        "invalid..com",  # Double dots
        "invalid .com",  # Contains space
        "invalid@com",  # Contains invalid character
        "invalid_com",  # Contains underscore
    ]


@pytest.fixture
def valid_strings() -> list[str]:
    """Valid non-empty strings for string validation."""
    return [
        "a",  # Single character
        "hello",
        "Hello World",
        "test-value",
        "test_value",
        "test.value",
        "123",
        "value with spaces",
        "UPPERCASE",
        "MixedCase",
    ]


@pytest.fixture
def empty_strings() -> list[str]:
    """Empty strings for validation."""
    return [
        "",  # Empty string
    ]


@pytest.fixture
def whitespace_strings() -> list[str]:
    """Whitespace-only strings for validation."""
    return [
        " ",  # Single space
        "   ",  # Multiple spaces
        "\t",  # Tab
        "\n",  # Newline
        "  \t  \n  ",  # Mixed whitespace
    ]


@pytest.fixture
def valid_ranges() -> list[tuple[int, int, int]]:
    """Valid numeric ranges (value, min, max) for range validation."""
    return [
        (0, 0, 10),  # At minimum
        (5, 0, 10),  # In middle
        (10, 0, 10),  # At maximum
        (100, 0, 1000),  # Large range
        (-5, -10, 0),  # Negative range
        (-5, -10, 10),  # Range crossing zero
    ]


@pytest.fixture
def out_of_range() -> list[tuple[int, int, int]]:
    """Out-of-range numeric values (value, min, max) for range validation."""
    return [
        (-1, 0, 10),  # Below minimum
        (11, 0, 10),  # Above maximum
        (100, 0, 50),  # Way above maximum
        (-100, 0, 10),  # Way below minimum
    ]


# =========================================================================
# VALIDATION HELPER FUNCTIONS
# =========================================================================


def assert_validates(
    model_class: type,
    field_name: str,
    value: object,
) -> object:
    """Validate a value against a model field and return the validated value.

    Args:
        model_class: Pydantic model class to validate against
        field_name: Name of the field to validate
        value: Value to validate

    Returns:
        The validated value

    Raises:
        AssertionError: If validation fails

    """
    try:
        # Create instance with the field value
        instance = model_class(**{field_name: value})
        return getattr(instance, field_name)
    except Exception as e:
        pytest.fail(f"Validation failed for {field_name}={value}: {e}")


def assert_rejects(
    model_class: type,
    field_name: str,
    value: object,
    error_type: type[Exception] | None = None,
) -> str:
    """Assert that a value is rejected during validation.

    Args:
        model_class: Pydantic model class to validate against
        field_name: Name of the field to validate
        value: Value that should be rejected
        error_type: Expected exception type (optional)

    Returns:
        The error message from validation

    Raises:
        AssertionError: If validation succeeds when it should fail

    """
    try:
        instance = model_class(**{field_name: value})
        pytest.fail(
            (
                f"Expected validation to fail for {field_name}={value}, "
                f"but got: {getattr(instance, field_name)}"
            ),
            pytrace=False,
        )
    except Exception as e:
        error_msg = str(e)
        if error_type and not isinstance(e, error_type):
            pytest.fail(
                (
                    f"Expected {error_type.__name__}, "
                    f"but got {type(e).__name__}: {error_msg}"
                ),
                pytrace=False,
            )
        return error_msg
