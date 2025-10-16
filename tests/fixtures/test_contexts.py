"""Test contexts fixtures for flext-core tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from datetime import UTC, datetime

from flext_core import FlextTypes


def get_test_contexts() -> FlextTypes.NestedDict:
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
