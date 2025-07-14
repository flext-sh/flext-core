"""Pytest configuration and fixtures for flext-core tests."""

import os
from typing import Any, Generator

import pytest

# Basic pytest configuration
pytest_plugins: list[str] = []


@pytest.fixture(autouse=True)
def set_test_environment(request) -> Generator[None, None, None]:
    """Automatically set environment to 'test' for all tests except those testing real defaults."""
    # Skip setting test environment for tests that explicitly test defaults
    test_name = request.node.name
    parent_name = request.node.parent.name.lower() if request.node.parent else ""

    # Tests that intentionally test real defaults without interference
    skip_tests = [
        ("test_base_settings_creation", "comprehensive"),
        ("test_base_settings_defaults", ""),  # Any parent name for this test
    ]

    for test_pattern, parent_pattern in skip_tests:
        if test_pattern in test_name and (
            not parent_pattern or parent_pattern in parent_name
        ):
            yield
            return

    # Store original value to restore after test
    original_env = os.environ.get("FLEXT_ENVIRONMENT")

    # Set test environment (using FLEXT_ prefix as defined in ConfigDefaults)
    os.environ["FLEXT_ENVIRONMENT"] = "test"

    # Yield control back to test
    yield

    # Restore original environment after test
    if original_env is not None:
        os.environ["FLEXT_ENVIRONMENT"] = original_env
    else:
        os.environ.pop("FLEXT_ENVIRONMENT", None)


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "environment": "test",
        "debug": True,
        "log_level": "DEBUG",
    }


@pytest.fixture
def sample_pipeline_data() -> dict[str, Any]:
    """Sample pipeline data for testing."""
    return {
        "id": "test-pipeline-001",
        "name": "Test Pipeline",
        "description": "A test pipeline for unit tests",
        "status": "pending",
    }
