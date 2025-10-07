"""Comprehensive test configuration for flext-core with advanced pytest features.

Provides centralized fixtures, test utilities, and configuration for all flext-core tests
using the consolidated tests/support/ infrastructure for maximum testing efficiency.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextLogger,
    FlextResult,
    FlextTypes,
)


# Core Fixtures
@pytest.fixture(autouse=True)
def reset_container_singleton() -> Generator[None]:
    """Reset FlextContainer singleton between tests for isolation.

    This autouse fixture ensures that every test starts with a clean
    FlextContainer singleton state, preventing test contamination.
    """
    # Clear singleton before test
    FlextContainer._global_instance = None
    yield
    # Clear singleton after test
    FlextContainer._global_instance = None


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
    FlextContainer._global_instance = None
    container = FlextContainer()
    container.clear()  # Ensure it starts clean
    yield container
    # Cleanup: Clear the container and reset singleton after test
    container.clear()
    FlextContainer._global_instance = None


@pytest.fixture
def sample_data() -> FlextTypes.Dict:
    """Provide deterministic sample data for tests.

    Enterprise-grade test data factory providing consistent, typed sample data
    for testing data transformation and validation patterns across the ecosystem.

    Returns:
      Dict containing various data types for comprehensive testing

    """
    return {
        "string": "test_string",
        "integer": 42,
        "float": math.pi,
        "boolean": True,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "none": None,
        "timestamp": datetime.now(UTC).isoformat(),
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
    }


@pytest.fixture
def test_user_data() -> FlextTypes.Dict | FlextTypes.StringList | None:
    """Provide consistent user data for domain testing.

    User data factory aligned with shared domain patterns
    for testing DDD entities and value objects.

    Returns:
      Dict containing user data for testing

    """
    return {
        "id": "user-12345",
        "name": "Test User",
        "email": "test.user@flext.example.com",
        "age": 30,
        "is_active": True,
        "created_at": datetime.now(UTC).isoformat(),
        "roles": ["user", "tester"],
    }


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
            self.processed_items: FlextTypes.List = []
            self._should_fail = False
            self._failure_message = ""

        def process(self, data: object) -> FlextResult[str]:
            """Process data through mock external service.

            Returns:
                FlextResult[str]: Success with processed data or failure with error message.

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
    clean_container.register("external_service", mock_external_service)
    clean_container.register("config", {"test_mode": True})
    clean_container.register("logger", "test_logger")
    return clean_container


@pytest.fixture
def error_context() -> dict[str, str | None]:
    """Provide structured error context for testing.

    Error context factory for testing FlextResult error handling,
    logging, and observability patterns.

    Returns:
      Dict containing error context fields for testing

    """
    return {
        "error_code": "TEST_ERROR_001",
        "severity": "medium",
        "component": "test_module",
        "user_id": "test_user_123",
        "request_id": "test-request-456",
        "timestamp": datetime.now(UTC).isoformat(),
        "stack_trace": None,  # Populated by actual error handling
    }


# Test Data Constants - Centralized test data and constants
@pytest.fixture
def test_constants() -> FlextTypes.Dict:
    """Provide centralized test constants for all tests.

    Centralized constants used across multiple test files to ensure
    consistency and reduce duplication.

    Returns:
        Dict containing test constants and data

    """
    return {
        # Common test identifiers
        "test_user_id": "test_user_123",
        "test_session_id": "test_session_123",
        "test_service_name": "test_service",
        "test_operation_id": "test_operation",
        "test_request_id": "test-request-456",
        "test_correlation_id": "test-corr-123",
        # Test module and component names
        "test_module_name": "test_module",
        "test_handler_name": "test_handler",
        "test_chain_name": "test_chain",
        "test_command_type": "test_command",
        "test_query_type": "test_query",
        # Test error codes and messages
        "test_error_code": "TEST_ERROR_001",
        "test_validation_error": "test_error",
        "test_operation_error": "Op failed",
        "test_config_error": "Config failed",
        "test_timeout_error": "Operation timeout",
        # Test field names and values
        "test_field_name": "test_field",
        "test_config_key": "test_key",
        "test_username": "test_user",
        "test_email": "test@example.com",
        "test_password": "test_pass",
        # Test data values
        "test_string_value": "test_value",
        "test_input_data": "test_input",
        "test_request_data": "test_request",
        "test_result_data": "test_result",
        "test_message": "test_message",
        # Test service and component identifiers
        "test_logger_name": "test_logger",
        "test_app_name": "test-app",
        "test_validation_app": "validation-test",
        "test_source_service": "test_service",
        # Test patterns and formats
        "test_slug_input": "Test_String",
        "test_slug_expected": "test_string",
        "test_uuid_format": "550e8400-e29b-41d4-a716-446655440000",
        # Test port and numeric values
        "test_port": 8080,
        "test_timeout": 30,
        "test_retry_count": 3,
        "test_batch_size": 100,
    }


@pytest.fixture
def test_contexts() -> FlextTypes.NestedDict:
    """Provide common test contexts for various scenarios.

    Pre-defined contexts for testing different scenarios like
    user operations, service calls, validation, etc.

    Returns:
        Dict containing various test contexts

    """
    return {
        "user_context": {
            "user_id": "test_user_123",
            "username": "test_user",
            "email": "test@example.com",
            "roles": ["user", "tester"],
        },
        "service_context": {
            "service_name": "test_service",
            "version": "1.0.0",
            "environment": "test",
            "port": 8080,
        },
        "operation_context": {
            "operation_id": "test_operation",
            "module": "test_module",
            "function": "test_func",
            "correlation_id": "test-corr-123",
        },
        "error_context": {
            "error_code": "TEST_ERROR_001",
            "severity": "medium",
            "component": "test_module",
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "validation_context": {
            "field": "test_field",
            "rule": "required",
            "validator": "test_validator",
            "message": "Validation failed",
        },
        "request_context": {
            "request_id": "test-request-456",
            "method": "POST",
            "path": "/api/test",
            "headers": {"Content-Type": "application/json"},
        },
    }


@pytest.fixture
def test_payloads() -> FlextTypes.NestedDict:
    """Provide common test payloads for different operations.

    Standardized payloads for testing commands, queries, events,
    and other data structures across the system.

    Returns:
        Dict containing various test payloads

    """
    return {
        "command_payload": {
            "command_type": "test_command",
            "data": {"action": "create", "entity": "user"},
            "timestamp": datetime.now(UTC).isoformat(),
            "correlation_id": "test-corr-123",
        },
        "query_payload": {
            "query_type": "test_query",
            "filters": {"status": "active", "type": "test"},
            "pagination": {"page": 1, "size": 10},
            "sort": {"field": "created_at", "order": "desc"},
        },
        "event_payload": {
            "event_type": "test_event",
            "source": "test_service",
            "data": {"entity_id": "123", "action": "created"},
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "user_creation_payload": {
            "username": "test_user",
            "email": "test@example.com",
            "password": "test_pass",
            "profile": {"first_name": "Test", "last_name": "User"},
        },
        "service_config_payload": {
            "name": "test_service",
            "port": 8080,
            "timeout": 30,
            "retries": 3,
            "endpoints": ["/health", "/metrics"],
        },
        "validation_payload": {
            "field": "email",
            "value": "test@example.com",
            "rules": ["required", "email_format"],
            "context": {"form": "user_registration"},
        },
    }


@pytest.fixture
def test_error_scenarios() -> FlextTypes.NestedDict:
    """Provide common error scenarios for testing.

    Pre-defined error scenarios for testing error handling,
    validation failures, timeouts, and other edge cases.

    Returns:
        Dict containing various error scenarios

    """
    return {
        "validation_error": {
            "type": "ValidationError",
            "message": "Invalid input data",
            "field": "test_field",
            "code": "VAL_001",
            "context": {"input": "invalid_data"},
        },
        "configuration_error": {
            "type": "ConfigurationError",
            "message": "Missing required configuration",
            "config_key": "database_url",
            "code": "CFG_001",
            "context": {"section": "database"},
        },
        "connection_error": {
            "type": "ConnectionError",
            "message": "Failed to connect to service",
            "service": "test_service",
            "code": "CONN_001",
            "context": {"host": "localhost", "port": 8080},
        },
        "timeout_error": {
            "type": "TimeoutError",
            "message": "Operation timed out",
            "operation": "test_operation",
            "code": "TIMEOUT_001",
            "context": {"timeout": 30, "elapsed": 35},
        },
        "processing_error": {
            "type": "ProcessingError",
            "message": "Failed to process request",
            "handler": "test_handler",
            "code": "PROC_001",
            "context": {"stage": "validation", "input_size": 1024},
        },
        "authentication_error": {
            "type": "AuthenticationError",
            "message": "Authentication failed",
            "user": "test_user",
            "code": "AUTH_001",
            "context": {"method": "token", "reason": "expired"},
        },
    }


@pytest.fixture
def performance_threshold() -> FlextTypes.FloatDict:
    """Provide performance thresholds for testing.

    Performance threshold configuration for validating
    that core operations meet enterprise performance standards.

    Returns:
      Dict containing performance thresholds in seconds

    """
    return {
        "result_creation": 0.001,  # 1ms for FlextResult creation
        "container_registration": 0.005,  # 5ms for service registration
        "container_retrieval": 0.001,  # 1ms for service retrieval
        "validation": 0.01,  # 10ms for validation operations
        "serialization": 0.05,  # 50ms for serialization
    }


@pytest.fixture
def benchmark_data() -> FlextTypes.Dict:
    """Provide standardized data for performance testing.

    Benchmark data factory for testing performance characteristics
    of core patterns and algorithms.

    Returns:
      Dict containing benchmark data sets

    """
    return {
        "small_dataset": list(range(100)),
        "medium_dataset": list(range(1000)),
        "large_dataset": list(range(10000)),
        "complex_dict": {f"key_{i}": f"value_{i}" for i in range(1000)},
        "nested_structure": {
            "level1": {"level2": {"level3": {"data": list(range(100))}}},
        },
    }


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

        # Reset logger singleton state
        FlextLogger._structlog_configured = False

        # Set the expected log level for logging tests
        os.environ["FLEXT_LOG_LEVEL"] = "WARNING"
        yield
    finally:
        # Clear both singleton states again and restore original value
        FlextConfig.reset_global_instance()
        FlextLogger._structlog_configured = False

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
