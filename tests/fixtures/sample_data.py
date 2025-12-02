"""Sample data fixtures using advanced Python 3.13 patterns.

Provides comprehensive sample data factories for testing data transformation,
validation patterns, and domain objects across the flext-core ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from flext_core.typings import FlextTypes

from ..helpers.constants import TestConstants


@dataclass(frozen=True, slots=True)
class TestUserData:
    """Factory for test user data structures."""

    user_id: str = TestConstants.Strings.USER_ID_VALID
    name: str = "Test User"
    email: str = TestConstants.Strings.VALID_EMAIL
    age: int = 30
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    roles: list[str] = field(default_factory=lambda: ["user", "tester"])

    def to_dict(self) -> dict[str, FlextTypes.GeneralValueType]:
        """Convert to dictionary format for compatibility."""
        return {
            "id": self.user_id,
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "roles": self.roles.copy(),
        }


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """Factory for error context data structures."""

    error_code: str = "TEST_ERROR_001"
    severity: str = "medium"
    component: str = "test_module"
    user_id: str = "test_user_123"
    request_id: str = "test-request-456"
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    stack_trace: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary format for compatibility."""
        return {
            "error_code": self.error_code,
            "severity": self.severity,
            "component": self.component,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace,
        }


@dataclass(frozen=True, slots=True)
class SampleDataSet:
    """Factory for comprehensive sample data sets."""

    string_data: str = TestConstants.Strings.BASIC_WORD
    integer_data: int = 42
    float_data: float = math.pi
    boolean_data: bool = True
    list_data: list[int] = field(default_factory=lambda: [1, 2, 3])
    dict_data: dict[str, str] = field(default_factory=lambda: {"key": "value"})
    none_data: None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    uuid_data: str = "550e8400-e29b-41d4-a716-446655440000"

    def to_dict(self) -> dict[str, FlextTypes.GeneralValueType]:
        """Convert to dictionary format for compatibility."""
        return {
            "string": self.string_data,
            "integer": self.integer_data,
            "float": self.float_data,
            "boolean": self.boolean_data,
            "list": self.list_data.copy(),
            "dict": self.dict_data.copy(),
            "none": self.none_data,
            "timestamp": self.timestamp,
            "uuid": self.uuid_data,
        }


class SampleDataFactories:
    """Centralized factories for sample data."""

    @staticmethod
    def create_basic_user() -> TestUserData:
        """Create basic test user."""
        return TestUserData()

    @staticmethod
    def create_admin_user() -> TestUserData:
        """Create admin test user."""
        return TestUserData(
            user_id="admin-123",
            name="Admin User",
            email="admin@flext.example.com",
            roles=["admin", "user", "tester"],
        )

    @staticmethod
    def create_inactive_user() -> TestUserData:
        """Create inactive test user."""
        return TestUserData(
            user_id="inactive-123",
            name="Inactive User",
            is_active=False,
        )

    @staticmethod
    def create_error_context(
        error_code: str = "TEST_ERROR_001",
        severity: str = "medium",
    ) -> ErrorContext:
        """Create error context for testing."""
        return ErrorContext(error_code=error_code, severity=severity)

    @staticmethod
    def create_validation_error_context() -> ErrorContext:
        """Create validation error context."""
        return ErrorContext(
            error_code="VAL_001",
            severity="low",
            component="validation_module",
        )

    @staticmethod
    def create_comprehensive_sample_data() -> SampleDataSet:
        """Create comprehensive sample data set."""
        return SampleDataSet()

    @staticmethod
    def create_edge_case_sample_data() -> SampleDataSet:
        """Create edge case sample data."""
        return SampleDataSet(
            string_data=TestConstants.Strings.EMPTY,
            integer_data=0,
            float_data=0.0,
            boolean_data=False,
            list_data=[],
            dict_data={},
        )


# Backward compatibility functions
def get_sample_data() -> FlextTypes.Types.ConfigurationMapping:
    """Provide deterministic sample data for tests (backward compatibility).

    Returns:
        Dict containing various data types for comprehensive testing

    """
    return SampleDataFactories.create_comprehensive_sample_data().to_dict()


def get_test_user_data() -> FlextTypes.Types.ConfigurationMapping | list[str] | None:
    """Provide consistent user data for domain testing (backward compatibility).

    Returns:
        Dict containing user data for testing

    """
    return SampleDataFactories.create_basic_user().to_dict()


def get_error_context() -> dict[str, str | None]:
    """Provide structured error context for testing (backward compatibility).

    Returns:
        Dict containing error context fields for testing

    """
    return SampleDataFactories.create_error_context().to_dict()
