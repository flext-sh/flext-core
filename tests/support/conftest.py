# ruff: noqa: PLC0415
"""Comprehensive test configuration with factory_boy and pytest ecosystem integration.

This module provides:
- Factory Boy configuration and registration
- Pytest plugins integration
- Custom markers and hooks
- Test environment setup

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Generator

import pytest

from tests.support.factories import (
    BatchFactory,
    FlextResultFactory,
    TestEntityFactory,
    UserDataFactory,
)
from tests.support.fixtures_advanced import *  # noqa: F403,F401 # Import all fixtures
from tests.support.matchers import FlextMatchers, PerformanceMatchers

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "fast: marks tests as fast running")
    config.addinivalue_line("markers", "factory: marks tests using factory_boy")
    config.addinivalue_line("markers", "benchmark: marks performance benchmark tests")
    config.addinivalue_line(
        "markers", "flext_result: marks tests for FlextResult patterns"
    )
    config.addinivalue_line(
        "markers", "flext_model: marks tests for FlextModel patterns"
    )
    config.addinivalue_line("markers", "container: marks tests for DI container")
    config.addinivalue_line("markers", "domain: marks tests for domain logic")
    config.addinivalue_line("markers", "infrastructure: marks tests for infrastructure")
    config.addinivalue_line("markers", "application: marks tests for application layer")


# =============================================================================
# FACTORY BOY INTEGRATION
# =============================================================================


@pytest.fixture(scope="session")
def factory_registry() -> dict[str, type]:
    """Provide registry of all available factories."""
    return {
        "user": UserDataFactory,
        "entity": TestEntityFactory,
        "batch": BatchFactory,
        "result": FlextResultFactory,
    }


@pytest.fixture
def create_users(
    factory_registry: dict[str, type],
) -> Callable[[int], list[dict[str, object]]]:
    """Factory function for creating test users."""

    def _create_users(count: int = 1, **kwargs: object) -> list[dict[str, object]]:
        return factory_registry["batch"].create_user_batch(count, **kwargs)

    return _create_users


@pytest.fixture
def create_entities(factory_registry: dict[str, type]) -> Callable[[int], list[object]]:
    """Factory function for creating test entities."""

    def _create_entities(count: int = 1, **kwargs: object) -> list[object]:
        return factory_registry["batch"].create_entity_batch(count, **kwargs)

    return _create_entities


@pytest.fixture
def create_results(
    factory_registry: dict[str, type],
) -> Callable[[int, int], list[object]]:
    """Factory function for creating FlextResult objects."""

    def _create_results(success_count: int = 1, failure_count: int = 0) -> list[object]:
        return factory_registry["batch"].create_mixed_results(
            success_count, failure_count
        )

    return _create_results


# =============================================================================
# MATCHER INTEGRATION
# =============================================================================


@pytest.fixture(scope="session")
def matchers() -> type[FlextMatchers]:
    """Provide FlextMatchers for test assertions."""
    return FlextMatchers


@pytest.fixture(scope="session")
def performance_matchers() -> type[PerformanceMatchers]:
    """Provide PerformanceMatchers for benchmark tests."""
    return PerformanceMatchers


# =============================================================================
# PYTEST HOOKS
# =============================================================================


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],  # noqa: ARG001
) -> None:
    """Modify test collection to add markers based on test patterns."""
    for item in items:
        # Auto-mark factory tests
        if "factory" in item.nodeid.lower() or any(
            keyword in item.name.lower()
            for keyword in ["factory", "create_", "build_", "generate_"]
        ):
            item.add_marker(pytest.mark.factory)

        # Auto-mark benchmark tests
        if "benchmark" in item.nodeid.lower() or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.benchmark)
            item.add_marker(pytest.mark.slow)

        # Auto-mark FlextResult tests
        if any(
            keyword in item.name.lower() for keyword in ["result", "success", "failure"]
        ):
            item.add_marker(pytest.mark.flext_result)

        # Auto-mark by test location
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Setup hook before each test runs."""
    # Skip slow tests if --fast flag is used
    if item.config.getoption("--fast", default=False):
        slow_marker = item.get_closest_marker("slow")
        if slow_marker:
            pytest.skip("skipping slow test in fast mode")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="run only fast tests, skip slow ones",
    )
    parser.addoption(
        "--benchmark-only",
        action="store_true",
        default=False,
        help="run only benchmark tests",
    )
    parser.addoption(
        "--factory-tests",
        action="store_true",
        default=False,
        help="run only factory-related tests",
    )


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:  # noqa: ARG001
    """Hook to customize test reports."""
    # Add factory information to test reports if applicable
    if hasattr(item, "factory_used"):
        item.user_properties.append(("factory", item.factory_used))


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment() -> None:
    """Configure test environment settings."""
    import os

    # Set test-specific environment variables
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("PYTEST_RUNNING", "true")


# =============================================================================
# FACTORY BOY SEQUENCE RESET
# =============================================================================


@pytest.fixture(autouse=True)
def reset_factory_sequences() -> None:
    """Reset factory sequences between tests for consistency."""
    # Note: factory-boy doesn't have global sequence reset
    # Individual factories should handle sequence isolation if needed


# =============================================================================
# TEST DATA ISOLATION
# =============================================================================


@pytest.fixture(autouse=True)
def isolate_test_data() -> Generator[None]:
    """Ensure test data isolation between tests."""
    # Clear any global state that might affect tests
    yield

    # Cleanup after test
    import gc

    gc.collect()
