"""Error scenarios fixtures for flext-core tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations


def get_test_error_scenarios() -> dict[str, dict[str, object]]:
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
