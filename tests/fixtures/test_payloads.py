"""Test payloads fixtures for flext-core tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from datetime import UTC, datetime


def get_test_payloads() -> dict[str, object]:
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
