"""Basic test configuration for flext-core."""

from __future__ import annotations

import pytest

from tests.support.domain_factories import UserDataFactory


@pytest.fixture
def test_scenario() -> dict[str, str]:
    """Basic test scenario fixture."""
    return {"status": "test"}


@pytest.fixture
def user_data_factory() -> type[UserDataFactory]:
    """User data factory fixture."""
    return UserDataFactory


@pytest.fixture
def assert_helpers():
    """Simple assert helpers for test compatibility."""
    class AssertHelpers:
        def assert_result_success(self, result) -> None:
            """Assert that a FlextResult is successful."""
            assert result.success, f"Expected success but got failure: {result.error}"

        def assert_result_failure(self, result) -> None:
            """Assert that a FlextResult is a failure."""
            assert result.failure, f"Expected failure but got success: {result.value}"

    return AssertHelpers()


class TestScenario:
    """Basic test scenario class."""

    def __init__(self, name: str = "test") -> None:
        self.name = name
        self.status = "active"


@pytest.fixture
def clean_container():
    """Create a clean container for testing."""
    from flext_core import FlextContainer
    
    # Create a new container instance
    container = FlextContainer()
    yield container
    # No cleanup needed as each test gets a fresh container


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for testing."""
    yield tmp_path
