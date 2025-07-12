"""Pytest configuration and fixtures for flext-core tests."""

from typing import Any

import pytest

# Basic pytest configuration
pytest_plugins: list[str] = []


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
