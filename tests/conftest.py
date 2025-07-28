"""Pytest configuration for FLEXT Core test suite.

Provides shared fixtures and configuration for all test types:
unit, integration, and e2e tests.
"""

import math
import os
from typing import Any

import pytest

from flext_core.container import FlextContainer


# Pytest configuration for test markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers for organized test execution."""
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests (deselect with '-m \"not unit\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "pep8: marks tests for PEP8 compliance validation",
    )
    config.addinivalue_line(
        "markers",
        "core: marks tests for core framework functionality",
    )
    config.addinivalue_line(
        "markers",
        "architecture: marks tests for architectural patterns",
    )
    config.addinivalue_line(
        "markers",
        "ddd: marks tests for domain-driven design",
    )


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables that might interfere with tests."""
    # Clear any FLEXT environment variables that might interfere
    for key in list(os.environ.keys()):
        if key.startswith("FLEXT_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Provide sample data for tests."""
    return {
        "string": "test_string",
        "integer": 42,
        "float": math.pi,
        "boolean": True,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "none": None,
    }


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Provide sample metadata for tests."""
    return {
        "source": "test",
        "timestamp": 1234567890,
        "version": "1.0.0",
        "tags": ["unit", "test"],
    }


@pytest.fixture
def error_context() -> dict[str, Any]:
    """Provide sample error context for tests."""
    return {
        "error_code": "TEST_ERROR",
        "severity": "medium",
        "component": "test_module",
        "user_id": "test_user",
    }


@pytest.fixture
def clean_container() -> FlextContainer:
    """Provide a clean FlextContainer instance for each test."""
    return FlextContainer()
