"""Comprehensive test configuration and utilities for flext-core.

Provides highly automated testing infrastructure following strict
type-system-architecture.md rules with real functionality testing.
"""

from __future__ import annotations

import builtins
import math
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from flext_tests import t

from flext_core import FlextContainer, FlextContext, FlextSettings, r, t as core_t

setattr(builtins, "t", core_t)

from .helpers.scenarios import (  # noqa: E402 — must run after builtins.t assignment above
    TestHelperScenarios,
)


class FunctionalExternalService:
    """Mock external service for integration testing.

    Provides real functionality for testing service integration patterns
    with dependency injection and result handling.
    """

    def __init__(self) -> None:
        """Initialize external service with empty state."""
        self.processed_items: list[str] = []
        self.call_count = 0

    def process(self, input_data: str) -> r[str]:
        """Process input data by prefixing with 'processed_'.

        Args:
            input_data: String to process

        Returns:
            r[str]: Processed result or failure

        """
        try:
            self.call_count += 1
            processed = f"processed_{input_data}"
            self.processed_items.append(processed)
            return r[str].ok(processed)
        except Exception as e:
            return r[str].fail(f"Processing failed: {e}")

    def get_call_count(self) -> int:
        """Get number of times process() was called."""
        return self.call_count


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
def reset_global_container() -> Generator[None]:
    """Reset the global FlextContainer and FlextSettings instances after each test.

    This fixture ensures test isolation by clearing the global singletons
    that persist across tests. Without this, tests interfere with each other
    due to shared global state.
    """
    yield
    FlextContainer.reset_for_testing()
    FlextSettings.reset_for_testing()


@pytest.fixture
def mock_external_service() -> FunctionalExternalService:
    """Provide mock external service for integration tests."""
    return FunctionalExternalService()


@pytest.fixture
def sample_data() -> dict[str, t.NormalizedValue]:
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
def flext_result_success() -> r[dict[str, t.NormalizedValue]]:
    """Successful r fixture available to all FLEXT projects."""
    return r[dict[str, t.NormalizedValue]].ok({"success": True})


@pytest.fixture
def flext_result_failure() -> r[str]:
    """Failed r fixture available to all FLEXT projects."""
    return r[str].fail("Test error")


@pytest.fixture
def validation_scenarios() -> type:
    """Access to all centralized validation scenarios."""
    return TestHelperScenarios.ValidationScenarios


@pytest.fixture
def parser_scenarios() -> type:
    """Access to all centralized parser scenarios."""
    return TestHelperScenarios.ParserScenarios


@pytest.fixture
def reliability_scenarios() -> type:
    """Access to all centralized reliability scenarios."""
    return TestHelperScenarios.ReliabilityScenarios


@pytest.fixture
def valid_port_numbers() -> list[int]:
    """Valid port numbers for PortNumber validation (1-65535)."""
    return [1, 80, 443, 8080, 3306, 5432, 27017, 65535]


@pytest.fixture
def invalid_port_numbers() -> list[int]:
    """Invalid port numbers for PortNumber validation."""
    return [0, -1, -8080, 65536, 100000]


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
        "",
        "   ",
        "localhost",
        "example.com",
        "://example.com",
        "http://",
        "http://:8080",
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
        "a",
        "a.b",
    ]


@pytest.fixture
def invalid_hostnames() -> list[str]:
    """Invalid hostnames for HostnameStr validation."""
    return [
        "",
        "   ",
        "-invalid",
        "invalid-",
        "invalid..com",
        "invalid .com",
        "invalid@com",
        "invalid_com",
    ]


@pytest.fixture
def valid_strings() -> list[str]:
    """Valid non-empty strings for string validation."""
    return [
        "a",
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
    return [""]


@pytest.fixture
def whitespace_strings() -> list[str]:
    """Whitespace-only strings for validation."""
    return [" ", "   ", "\t", "\n", "  \t  \n  "]


@pytest.fixture
def valid_ranges() -> list[tuple[int, int, int]]:
    """Valid numeric ranges (value, min, max) for range validation."""
    return [
        (0, 0, 10),
        (5, 0, 10),
        (10, 0, 10),
        (100, 0, 1000),
        (-5, -10, 0),
        (-5, -10, 10),
    ]


@pytest.fixture
def out_of_range() -> list[tuple[int, int, int]]:
    """Out-of-range numeric values (value, min, max) for range validation."""
    return [(-1, 0, 10), (11, 0, 10), (100, 0, 50), (-100, 0, 10)]


def assert_validates(
    model_class: type, field_name: str, value: t.NormalizedValue
) -> t.NormalizedValue:
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
        instance = model_class(**{field_name: value})
        return getattr(instance, field_name)
    except Exception as e:
        pytest.fail(f"Validation failed for {field_name}={value}: {e}")


def assert_rejects(
    model_class: type,
    field_name: str,
    value: t.NormalizedValue,
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
            f"Expected validation to fail for {field_name}={value}, but got: {getattr(instance, field_name)}",
            pytrace=False,
        )
    except Exception as e:
        error_msg = str(e)
        if error_type and (not isinstance(e, error_type)):
            pytest.fail(
                f"Expected {error_type.__name__}, but got {type(e).__name__}: {error_msg}",
                pytrace=False,
            )
        return error_msg
