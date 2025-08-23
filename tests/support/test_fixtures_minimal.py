"""Minimal test fixtures for flext-core tests with strict typing.

This module provides essential fixtures without factory_boy complications,
focusing on real implementations and strict typing compliance.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from flext_core import FlextContainer, FlextResult


@pytest.fixture
def sample_entity_id() -> str:
    """Provide a sample entity ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_timestamp() -> datetime:
    """Provide a sample timestamp for testing."""
    return datetime.now(UTC)


@pytest.fixture
def success_result() -> FlextResult[str]:
    """Provide a successful FlextResult for testing."""
    return FlextResult[str].ok("test_value")


@pytest.fixture
def failure_result() -> FlextResult[str]:
    """Provide a failed FlextResult for testing."""
    return FlextResult[str].fail("test_error")


@pytest.fixture
def clean_container() -> FlextContainer:
    """Provide a clean FlextContainer for testing."""
    return FlextContainer()


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Provide sample data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "name": "test_item",
        "value": 42,
        "active": True,
        "created_at": datetime.now(UTC).isoformat(),
    }


# Helper functions for creating test objects
def create_test_entity_data(**kwargs: Any) -> dict[str, Any]:
    """Create test entity data with defaults."""
    defaults = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "version": 1,
    }
    defaults.update(kwargs)
    return defaults


def create_test_result_success(data: Any = None) -> FlextResult[Any]:
    """Create a successful test result."""
    return FlextResult[Any].ok(data or "test_data")


def create_test_result_failure(error: str = "test_error") -> FlextResult[Any]:
    """Create a failed test result."""
    return FlextResult[Any].fail(error)
