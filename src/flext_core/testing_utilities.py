"""FLEXT Core Testing Utilities - Ecosystem Testing Support.

Common testing fixtures, utilities, and configuration factories that eliminate
code duplication across all 32 projects in the FLEXT ecosystem. Provides
standardized test data, connection configurations, and testing patterns that
ensure consistent testing practices throughout the entire platform.

Module Role in Architecture:
    Testing Support Layer → Test Fixtures & Utilities → All Test Suites

    This testing module enables:
    - Consistent test configurations across ALGAR migration and OUD projects
    - Standardized LDAP connection testing in flext-ldap and related projects
    - Common API response mocking for flext-api and service integration tests
    - Shared test data factories for Singer taps, targets, and DBT projects
    - Configuration testing patterns for all infrastructure libraries

Testing Patterns:
    Configuration Factories: Pre-configured test environments for consistency
    Connection Mocking: Standardized connection simulation for integration tests
    Data Fixtures: Common test data sets used across multiple projects
    Response Mocking: API response simulation for service testing
    Validation Testing: Common validation scenarios for business logic testing

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Connection configurations, basic test utilities
    🔄 Enhancement: Advanced test fixtures for enterprise scenarios
    📋 TODO Integration: Plugin testing utilities (Plugin Priority 3)

Core Testing Utilities:
    create_oud_connection_config(): Oracle OUD connection for ALGAR testing
    create_ldap_test_config(): Generic LDAP connection for directory testing
    create_api_test_response(): Mock API responses for service integration
    create_database_test_config(): Database connection for repository testing
    create_singer_test_config(): Singer protocol testing configuration

Ecosystem Usage Patterns:
    # ALGAR OUD migration testing
    def test_oud_connection():
        config = testing_utilities.create_oud_connection_config()
        connection = OUDConnection(config)
        assert connection.test_connection().is_success

    # Singer tap testing across projects
    def test_tap_configuration():
        config = testing_utilities.create_singer_test_config("oracle")
        tap = OracleTap(config)
        assert tap.validate_config().is_success

    # API service integration testing
    def test_user_api():
        mock_response = testing_utilities.create_api_test_response(
            {"user_id": "123", "name": "Test User"}
        )
        # Use in service tests

Testing Philosophy:
    - Consistent test data prevents environment-specific test failures
    - Standardized configurations ensure reproducible test results
    - Shared utilities reduce maintenance overhead across 32 projects
    - Realistic test data improves integration test quality
    - Common patterns improve developer productivity

Quality Standards:
    - All test utilities must work across different project types
    - Configuration factories must match production patterns
    - Test data must be realistic but not contain sensitive information
    - Utilities must support both unit and integration testing scenarios
    - Mock responses must accurately reflect real API behavior

Enterprise Testing Requirements:
    - Security: No real credentials or sensitive data in test utilities
    - Performance: Test utilities must not introduce significant overhead
    - Reliability: Consistent behavior across different testing environments
    - Maintainability: Changes to test utilities must not break existing tests

See Also:
    tests/conftest.py: FLEXT Core test configuration and fixtures
    tests/shared_test_domain.py: Shared domain models for testing
    docs/python-module-organization.md: Testing patterns and practices

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations


def create_oud_connection_config() -> dict[str, str]:
    """Create standardized OUD connection configuration for testing.

    Provides consistent OUD connection parameters across all FLEXT tests
    to eliminate code duplication and ensure compatibility.

    Returns:
        Dictionary containing OUD connection parameters compatible with
        ALGAR and other FLEXT ecosystem requirements.

    """
    return {
        "host": "localhost",
        "port": "3389",
        "bind_dn": "cn=orcladmin",
        "bind_password": "Welcome1",
        "base_dn": "dc=ctbc,dc=com",
        "use_ssl": "false",
        "timeout": "30",
    }


def create_ldap_test_config() -> dict[str, object]:
    """Create standardized LDAP test configuration.

    Returns:
        Dictionary with LDAP connection parameters for testing.

    """
    return {
        "host": "localhost",
        "port": 389,
        "bind_dn": "cn=admin,dc=test,dc=com",
        "bind_password": "testpass",
        "base_dn": "dc=test,dc=com",
        "use_ssl": False,
        "timeout": 30,
    }


def create_api_test_response(
    *,
    success: bool = True,
    data: object = None,
) -> dict[str, object]:
    """Create standardized API test response.

    Args:
        success: Whether the response represents success.
        data: Response data payload.

    Returns:
        Standardized API response dictionary.

    """
    if success:
        return {
            "success": True,
            "data": data or {"id": "test_123", "status": "active"},
            "timestamp": "2025-01-20T12:00:00Z",
        }
    return {
        "success": False,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Invalid input data",
            "details": {"field": "name", "error": "required"},
        },
        "timestamp": "2025-01-20T12:00:00Z",
    }
