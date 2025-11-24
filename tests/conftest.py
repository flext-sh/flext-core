"""Comprehensive test configuration for flext-core with advanced pytest features.

Provides centralized fixtures, test utilities, and configuration for all flext-core tests
using the consolidated tests/support/ infrastructure for maximum testing efficiency.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
from pathlib import Path

import pytest

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextLogger,
    FlextResult,
)
from flext_core.runtime import FlextRuntime
from flext_tests.docker import FlextTestDocker

from .fixtures import (
    get_benchmark_data,
    get_error_context,
    get_performance_threshold,
    get_sample_data,
    get_test_constants,
    get_test_contexts,
    get_test_error_scenarios,
    get_test_payloads,
    get_test_user_data,
)

# Suppress pkg_resources deprecation warning from fs package
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
)


# Docker Test Infrastructure
@pytest.fixture(scope="session", autouse=True)
def test_docker_containers() -> Generator[FlextTestDocker | None]:
    """Session-scoped fixture for Docker test containers.

    Starts test containers at the beginning of the test session and keeps them
    running throughout all tests. Containers are only recreated if there are
    real infrastructure failures inside them.

    Yields:
        FlextTestDocker or None: Configured docker controller instance or None if docker unavailable

    Note:
        Containers remain running after test completion for debugging and
        iterative development. They are only stopped on explicit teardown
        or system shutdown.

    """
    try:
        docker_controller = FlextTestDocker()
        # Start containers - they will stay running
        docker_controller.start_all()
        yield docker_controller
    except Exception:
        # Docker not available, yield None
        yield None
    # Note: No teardown - containers stay running as per requirements


# Core Fixtures
@pytest.fixture(autouse=True)
def reset_container_singleton() -> Generator[None]:
    """Reset FlextContainer, FlextConfig, FlextRuntime, and FlextLogger between tests.

    This autouse fixture ensures that every test starts with clean singleton states,
    preventing test contamination from shared state. Critical for test idempotency
    and parallel execution.
    """
    # Clear singletons before test
    FlextContainer._global_instance = None
    FlextConfig.reset_global_instance()
    FlextConfig._di_config_provider = None

    # Reset FlextRuntime structlog configuration state
    FlextRuntime._structlog_configured = False

    # Clear FlextLogger global context (contextvars)
    FlextLogger.clear_global_context()
    # Clear all scoped contexts
    FlextLogger._scoped_contexts.clear()
    # Clear all level contexts
    FlextLogger._level_contexts.clear()

    yield

    # Clear singletons after test
    FlextContainer._global_instance = None
    FlextConfig.reset_global_instance()
    FlextConfig._di_config_provider = None

    # Reset FlextRuntime state after test
    FlextRuntime._structlog_configured = False

    # Clear FlextLogger state after test
    FlextLogger.clear_global_context()
    FlextLogger._scoped_contexts.clear()
    FlextLogger._level_contexts.clear()


@pytest.fixture
def test_scenario() -> dict[str, str]:
    """Basic test scenario fixture.

    Returns:
        dict[str, str]: Test scenario data with status and environment.

    """
    return {"status": "test", "environment": "test"}


@pytest.fixture
def clean_container() -> Generator[object]:
    """Provide isolated FlextContainer for dependency injection testing.

    Enterprise-grade DI container fixture ensuring complete test isolation.
    Each test receives a fresh container with no service registrations.

    Yields:
      FlextContainer instance with proper cleanup

    """
    # Clear any existing singleton state before test
    FlextContainer._global_instance = None
    container = FlextContainer()
    container.clear()  # Ensure it starts clean
    yield container
    # Cleanup: Clear the container and reset singleton after test
    container.clear()
    FlextContainer._global_instance = None


@pytest.fixture
def sample_data() -> dict[str, object]:
    """Provide deterministic sample data for tests."""
    return get_sample_data()


@pytest.fixture
def test_user_data() -> dict[str, object] | list[str] | None:
    """Provide consistent user data for domain testing."""
    return get_test_user_data()


@pytest.fixture
def temp_directory() -> Generator[Path]:
    """Provide temporary directory for file-based testing.

    Temporary directory fixture for testing file operations,
    configuration loading, and data persistence patterns.

    Yields:
      Path to temporary directory with automatic cleanup

    """
    temp_dir = Path(tempfile.mkdtemp(prefix="flext_test_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_external_service() -> object:
    """Provide mock external service for integration testing.

    Simple mock service that can be used across different test modules
    for consistent external service simulation.

    Returns:
        Mock external service object

    """

    class MockExternalService:
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0
            self.processed_items: list[object] = []
            self._should_fail = False
            self._failure_message = ""

        def process(self, data: object) -> object:
            """Process data through mock external service.

            Returns:
                Result: Success with processed data or failure with error message.

            """
            self.call_count += 1
            self.processed_items.append(data)

            if self._should_fail:
                return FlextResult[str].fail(self._failure_message)

            return FlextResult[str].ok(f"processed_{data}")

        def get_call_count(self) -> int:
            """Get number of times process was called.

            Returns:
                int: Number of times the process method was called.

            """
            return self.call_count

        def set_failure_mode(
            self,
            *,
            should_fail: bool = True,
            message: str = "Mock service failure",
        ) -> None:
            """Configure mock to simulate failures."""
            self._should_fail = should_fail
            self._failure_message = message

    return MockExternalService()


@pytest.fixture
def configured_container(
    clean_container: FlextContainer,
    mock_external_service: object,
) -> FlextContainer:
    """Provide pre-configured container for integration testing.

    Container factory with common service registrations for testing
    service integration patterns and dependency resolution.

    Args:
      clean_container: Fresh container instance
      mock_external_service: External service for functional testing

    Returns:
      FlextContainer with standard test services registered

    """
    clean_container.with_service("external_service", mock_external_service)
    clean_container.with_service("config", {"test_mode": True})
    clean_container.with_service("logger", "test_logger")
    return clean_container


@pytest.fixture
def error_context() -> dict[str, str | None]:
    """Provide structured error context for testing."""
    return get_error_context()


# Test Data Constants - Centralized test data and constants
@pytest.fixture
def test_constants() -> dict[str, object]:
    """Provide centralized test constants for all tests."""
    return get_test_constants()


@pytest.fixture
def test_contexts() -> dict[str, object]:
    """Provide common test contexts for various scenarios."""
    return get_test_contexts()


@pytest.fixture
def test_payloads() -> object:
    """Provide common test payloads for different operations."""
    return get_test_payloads()


@pytest.fixture
def test_error_scenarios() -> object:
    """Provide common error scenarios for testing."""
    return get_test_error_scenarios()


@pytest.fixture
def performance_threshold() -> object:
    """Provide performance thresholds for testing."""
    return get_performance_threshold()


@pytest.fixture
def benchmark_data() -> object:
    """Provide standardized data for performance testing."""
    return get_benchmark_data()


# Factory Fixtures - COMMENTED OUT: Use FlextTestsMatchers directly instead of aliases
# @pytest.fixture
# def user_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def advanced_user_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def config_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers


# COMMENTED OUT: Use FlextTestsMatchers directly instead of aliases
# @pytest.fixture
# def advanced_config_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def result_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def service_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers
#
# @pytest.fixture
# def payload_factory() -> type:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers


# -----------------------------------------------------------------------------
# Test isolation for flext-core singleton state
# -----------------------------------------------------------------------------


# Temporarily disabled - causing test hangs, needs investigation
# @pytest.fixture(autouse=True)
# def _isolate_flext_core_state() -> None:
#     """Clear core specialized configs before each test.
#
#     Prevents cross-test leakage of database/security/logging configs that could
#     make property-default tests flaky or order-dependent.
#     """
#     try:
#         # Individual components isolation - Flext facade was removed
#         # Reset container singleton state if needed
#         manager = FlextContainer.get_global().clear()()
#         container = manager.get_or_create()
#         container.clear()
#     except Exception:
#         # If container isolation fails, continue with test
#         # This prevents test hangs while still attempting isolation
#         pass


@pytest.fixture
def logging_test_env() -> Generator[None]:
    """Set up test environment for logging tests.

    Ensures logging tests get consistent log level by temporarily overriding
    the FLEXT_LOG_LEVEL environment variable to WARNING as expected by tests.
    """
    # Save original value
    original_log_level = os.environ.get("FLEXT_LOG_LEVEL")

    try:
        # Clear both config and logger singleton states
        FlextConfig.reset_global_instance()

        # Reset logger singleton state via runtime
        FlextRuntime._structlog_configured = False

        # Set the expected log level for logging tests
        os.environ["FLEXT_LOG_LEVEL"] = "WARNING"
        yield
    finally:
        # Clear both singleton states again and restore original value
        FlextConfig.reset_global_instance()
        FlextRuntime._structlog_configured = False

        if original_log_level is not None:
            os.environ["FLEXT_LOG_LEVEL"] = original_log_level
        elif "FLEXT_LOG_LEVEL" in os.environ:
            del os.environ["FLEXT_LOG_LEVEL"]


# Performance Testing Fixtures
# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# COMMENTED OUT: # def benchmark_utils() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# COMMENTED OUT: # def memory_profiler() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# COMMENTED OUT: # def performance_profiler() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# HTTP Testing Fixtures
# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# COMMENTED OUT: # def http_test_utils() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# COMMENTED OUT: # def api_test_client() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#


#
# @pytest.fixture
# COMMENTED OUT: # def test_builders() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# COMMENTED OUT: # def flext_matchers() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# Shared test configuration
@pytest.fixture(autouse=True)
def setup_test_environment() -> None:
    """Automatically set up test environment for all tests."""
    # object global test setup can go here
    return
    # object global test teardown can go here


# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# COMMENTED OUT: # def user_repository() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# def email_service() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# def audit_service() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()
#
# @pytest.fixture
# def failing_repository() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# COMMENTED OUT: Use FlextTestsMatchers directly
# @pytest.fixture
# def real_services() -> FlextTestsMatchers:
#     """Use FlextTestsMatchers directly."""
#     return FlextTestsMatchers()


# Mark configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest marks."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "core: Core framework tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "smoke: Smoke tests")
