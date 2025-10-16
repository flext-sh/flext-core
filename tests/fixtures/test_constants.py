"""Test constants fixtures for flext-core tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import FlextTypes


def get_test_constants() -> FlextTypes.Dict:
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
