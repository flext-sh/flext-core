"""Comprehensive test configuration and utilities for flext-core.

Provides highly automated testing infrastructure following strict
type-system-architecture.md rules with real functionality testing.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import (
    Generator,
    Sequence,
)
from pathlib import Path

import pytest

from flext_core import FlextContainer, FlextContext
from tests import c, p, r, u


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
    container.clear()
    return container


@pytest.fixture
def mock_external_service() -> u.Core.Tests.FunctionalExternalService:
    """Provide mock external service for integration tests."""
    return u.Core.Tests.FunctionalExternalService()


@pytest.fixture
def sample_data() -> dict[
    str, str | int | float | bool | list[str] | dict[str, str] | None
]:
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
def flext_result_success() -> p.Result[dict[str, bool]]:
    """Successful r fixture available to all FLEXT projects."""
    return r[dict[str, bool]].ok({"success": True})


@pytest.fixture
def flext_result_failure() -> p.Result[str]:
    """Failed r fixture available to all FLEXT projects."""
    return r[str].fail(c.Core.Tests.TestErrors.TEST_ERROR)


@pytest.fixture
def valid_port_numbers() -> Sequence[int]:
    """Valid port numbers for PortNumber validation (1-65535)."""
    return [1, 80, 443, 8080, 3306, 5432, 27017, 65535]


@pytest.fixture
def invalid_port_numbers() -> Sequence[int]:
    """Invalid port numbers for PortNumber validation."""
    return [0, -1, -8080, 65536, 100000]


@pytest.fixture
def valid_uris() -> Sequence[str]:
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
def invalid_uris() -> Sequence[str]:
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
def valid_hostnames() -> Sequence[str]:
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
def invalid_hostnames() -> Sequence[str]:
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
def valid_strings() -> Sequence[str]:
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
def empty_strings() -> Sequence[str]:
    """Empty strings for validation."""
    return [""]


@pytest.fixture
def whitespace_strings() -> Sequence[str]:
    """Whitespace-only strings for validation."""
    return [" ", "   ", "\t", "\n", "  \t  "]


@pytest.fixture
def valid_ranges() -> Sequence[tuple[int, int, int]]:
    """Valid numeric ranges (value, min, max) for range validation."""
    return [
        (5, 0, 10),
        (0, 0, 10),
        (10, 0, 10),
        (100, 50, 150),
        (-5, -10, 0),
    ]


@pytest.fixture
def invalid_ranges() -> Sequence[tuple[int, int, int]]:
    """Invalid numeric ranges (value, min, max) for range validation."""
    return [
        (-1, 0, 10),
        (11, 0, 10),
        (200, 50, 150),
        (-15, -10, 0),
    ]


@pytest.fixture
def valid_percentages() -> Sequence[float]:
    """Valid percentages (0.0 to 1.0) for percentage validation."""
    return [0.0, 0.5, 0.99, 1.0]


@pytest.fixture
def invalid_percentages() -> Sequence[float]:
    """Invalid percentages for validation."""
    return [-0.1, 1.1, 2.0, -1.0]
