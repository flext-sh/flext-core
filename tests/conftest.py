"""Pytest configuration for flext-core tests.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Set test environment
os.environ["FLEXT_ENV"] = "testing"
os.environ["FLEXT_DEBUG"] = "true"
os.environ["FLEXT_ENVIRONMENT"] = "test"


@pytest.fixture(autouse=True)
def set_test_environment(
    request: pytest.FixtureRequest,
) -> Generator[None]:
    """Automatically set environment to 'test' for all tests except those testing real defaults.

    This fixture ensures tests run in test environment unless explicitly testing defaults.
    """
    # Skip setting test environment for tests that explicitly test defaults
    test_name = request.node.name
    parent_name = request.node.parent.name.lower() if request.node.parent else ""

    # Tests that intentionally test real defaults without interference
    skip_tests = [
        ("test_base_settings_creation", "comprehensive"),
        ("test_base_settings_defaults", ""),  # Any parent name for this test
        ("test_default_configuration", ""),  # Tests development defaults
        ("test_load_from_file", ""),  # Tests file-based configuration loading
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


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


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


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test (fast, isolated)",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (may require external services)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )
    config.addinivalue_line(
        "markers",
        "docker: mark test as requiring Docker",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to all tests in unit directory
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to all tests in integration directory
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        # Add e2e marker to all tests in e2e directory
        elif "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
