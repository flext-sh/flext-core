"""Comprehensive test configuration for flext-core with advanced pytest features.

Provides centralized fixtures, test utilities, and configuration for all flext-core tests
using consolidated infrastructure for maximum testing efficiency and reduced code duplication.
Includes Docker integration, singleton management, and advanced pytest patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import cast

import pytest

from flext_core.config import FlextConfig
from flext_core.container import FlextContainer
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

# FlextLogger will be imported lazily in functions that need it
FlextLogger = None  # Placeholder to avoid NameError
# Import fixtures - disabled due to flext_tests dependency issues
# from .fixtures import (
#     get_benchmark_data,
#     get_error_context,
#     get_performance_threshold,
#     get_sample_data,
#     get_test_constants,
#     get_test_contexts,
#     get_test_error_scenarios,
#     get_test_payloads,
#     get_test_user_data,
# )

# Suppress pkg_resources deprecation warning from fs package
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
    category=UserWarning,
)


# Docker Test Infrastructure - Disabled due to missing dependency
# @pytest.fixture(scope="session", autouse=True)
# def test_docker_containers() -> Generator[FlextTestDocker | None]:
#     """Session-scoped fixture for Docker test containers.
#     Disabled due to missing python_on_whales dependency.
#     """


# Core Fixtures
@pytest.fixture(autouse=True)
def reset_container_singleton() -> Generator[None]:
    """Reset FlextContainer, FlextConfig, and FlextRuntime between tests.

    This autouse fixture ensures that every test starts with clean singleton states,
    preventing test contamination from shared state. Critical for test idempotency
    and parallel execution.

    Note: FlextLogger state is managed through FlextRuntime and context cleanup.
    """
    # Clear singletons before test
    FlextContainer._global_instance = None  # type: ignore[assignment,misc]
    FlextConfig.reset_global_instance()

    # Reset FlextRuntime structlog configuration state
    FlextRuntime._structlog_configured = False  # type: ignore[assignment,misc]

    yield

    # Clear singletons after test
    FlextContainer._global_instance = None  # type: ignore[assignment,misc]
    FlextConfig.reset_global_instance()

    # Reset FlextRuntime state after test
    FlextRuntime._structlog_configured = False  # type: ignore[assignment,misc]


@pytest.fixture
def test_scenario() -> dict[str, str]:
    """Basic test scenario fixture.

    Returns:
        dict[str, str]: Test scenario data with status and environment.

    """
    return {"status": "test", "environment": "test"}


@pytest.fixture
def clean_container() -> Generator[FlextContainer]:
    """Provide isolated FlextContainer for dependency injection testing.

    Enterprise-grade DI container fixture ensuring complete test isolation.
    Each test receives a fresh container with no service registrations.

    Yields:
      FlextContainer instance with proper cleanup

    """
    # Clear any existing singleton state before test
    # Reset singleton for clean test state
    FlextContainer._global_instance = None  # type: ignore[assignment,misc]
    container = FlextContainer()
    container.clear_all()  # Ensure it starts clean
    yield container
    # Cleanup: Clear the container and reset singleton after test
    container.clear_all()
    FlextContainer._global_instance = None  # type: ignore[assignment,misc]


@pytest.fixture
def sample_data() -> Mapping[str, FlextTypes.GeneralValueType]:
    """Provide deterministic sample data for tests."""
    # Basic sample data since fixtures are disabled
    return {"id": 1, "name": "test", "string": "sample_value", "value": "sample"}


@pytest.fixture
def test_user_data() -> Mapping[str, FlextTypes.GeneralValueType] | list[str] | None:
    """Provide consistent user data for domain testing."""
    # Basic user data since fixtures are disabled
    return {"user_id": 1, "username": "testuser", "email": "test@example.com"}


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


class FunctionalExternalService:
    """Functional external service for integration testing - real implementation.

    This is a real service implementation that can be used for testing
    without mocks. It provides functional behavior that can be configured
    for success/failure scenarios.
    """

    def __init__(self) -> None:
        """Initialize functional external service with processing state."""
        self.call_count = 0
        self.processed_items: list[FlextTypes.GeneralValueType] = []
        self._should_fail = False
        self._failure_message = "Service unavailable"

    def process(
        self,
        data: FlextTypes.GeneralValueType | None = None,
    ) -> FlextResult[str]:
        """Process data through functional external service.

        Returns:
            FlextResult[str]: Success with processed data or failure with error message.

        """
        self.call_count += 1
        processed_data = data or "processed"
        self.processed_items.append(processed_data)

        if self._should_fail:
            return FlextResult[str].fail(self._failure_message)

        return FlextResult[str].ok(f"processed_{processed_data}")

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
        message: str = "Service unavailable",
    ) -> None:
        """Configure service to fail for testing error scenarios."""
        self._should_fail = should_fail
        self._failure_message = message

    def reset(self) -> None:
        """Reset service state for test isolation."""
        self.call_count = 0
        self.processed_items.clear()
        self._should_fail = False
        self._failure_message = "Service unavailable"


@pytest.fixture
def mock_external_service() -> FunctionalExternalService:
    """Provide functional external service for integration testing.

    Real service implementation (not a mock) that can be configured
    for various test scenarios. Uses proper types from FlextTypes.

    Returns:
        FunctionalExternalService: A configured external service instance.

    """
    return FunctionalExternalService()


@pytest.fixture
def configured_container(
    clean_container: FlextContainer,
    mock_external_service: FunctionalExternalService,
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
    _ = clean_container.with_service(
        "external_service",
        cast("FlextTypes.GeneralValueType", mock_external_service),
    )
    _ = clean_container.with_service("config", {"test_mode": True})
    _ = clean_container.with_service("logger", "test_logger")
    return clean_container


@pytest.fixture
def error_context() -> dict[str, str | None]:
    """Provide structured error context for testing."""
    return {"error_type": "test_error", "message": "Test error occurred"}


# Test Data Constants - Centralized test data and constants
@pytest.fixture
def test_constants() -> Mapping[str, FlextTypes.GeneralValueType]:
    """Provide centralized test constants for all tests."""
    return {"timeout": 30, "max_retries": 3, "batch_size": 100}


@pytest.fixture
def test_contexts() -> Mapping[str, FlextTypes.GeneralValueType]:
    """Provide common test contexts for various scenarios."""
    return {"environment": "test", "service": "test_service"}


@pytest.fixture
def test_payloads() -> FlextTypes.GeneralValueType:
    """Provide common test payloads for different operations."""
    return {"operation": "test", "data": {"key": "value"}}


@pytest.fixture
def test_error_scenarios() -> FlextTypes.GeneralValueType:
    """Provide common error scenarios for testing."""
    return {"scenario": "network_error", "expected_code": 500}


@pytest.fixture
def performance_threshold() -> Mapping[str, float]:
    """Provide performance thresholds for testing."""
    return {"max_response_time": 1.0, "min_throughput": 100.0}


@pytest.fixture
def benchmark_data() -> FlextTypes.GeneralValueType:
    """Provide standardized data for performance testing."""
    return {"iterations": 1000, "data_size": 1024}


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
        FlextConfig._instances.clear()

        # Reset logger singleton state via runtime
        FlextRuntime._structlog_configured = False

        # Set the expected log level for logging tests
        os.environ["FLEXT_LOG_LEVEL"] = "WARNING"
        yield
    finally:
        # Clear both singleton states again and restore original value
        FlextConfig._instances.clear()
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
    # Global test setup can go here
    return
    # Global test teardown can go here


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
