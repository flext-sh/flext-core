"""Test constants fixtures using advanced Python 3.13 patterns.

Provides comprehensive test constant factories organized by domain,
ensuring consistency and reducing duplication across all flext-core tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from flext_core.typings import FlextTypes

from ..helpers.constants import TestConstants


@dataclass(frozen=True, slots=True)
class FixtureIdentifiers:
    """Test identifiers and IDs."""

    user_id: str = "test_user_123"
    session_id: str = "test_session_123"
    service_name: str = "test_service"
    operation_id: str = "test_operation"
    request_id: str = "test-request-456"
    correlation_id: str = "test-corr-123"


@dataclass(frozen=True, slots=True)
class FixtureNames:
    """Test module and component names."""

    module_name: str = "test_module"
    handler_name: str = "test_handler"
    chain_name: str = "test_chain"
    command_type: str = "test_command"
    query_type: str = "test_query"
    logger_name: str = "test_logger"
    app_name: str = "test-app"
    validation_app: str = "validation-test"
    source_service: str = "test_service"


@dataclass(frozen=True, slots=True)
class FixtureErrors:
    """Test error codes and messages."""

    error_code: str = "TEST_ERROR_001"
    validation_error: str = "test_error"
    operation_error: str = "Op failed"
    config_error: str = "Config failed"
    timeout_error: str = "Operation timeout"


@dataclass(frozen=True, slots=True)
class FixtureData:
    """Test field names and data values."""

    field_name: str = "test_field"
    config_key: str = "test_key"
    username: str = "test_user"
    email: str = TestConstants.Strings.VALID_EMAIL
    password: str = "test_pass"
    string_value: str = "test_value"
    input_data: str = "test_input"
    request_data: str = "test_request"
    result_data: str = "test_result"
    message: str = "test_message"


@dataclass(frozen=True, slots=True)
class FixturePatterns:
    """Test patterns and formats."""

    slug_input: str = "Test_String"
    slug_expected: str = "test_string"
    uuid_format: str = "550e8400-e29b-41d4-a716-446655440000"


@dataclass(frozen=True, slots=True)
class FixtureNumericValues:
    """Test port and numeric values."""

    port: int = 8080
    timeout: int = 30
    retry_count: int = 3
    batch_size: int = 100


@dataclass(frozen=True, slots=True)
class FixtureConstantsCollection:
    """Complete collection of all test constants."""

    identifiers: FixtureIdentifiers = FixtureIdentifiers()
    names: FixtureNames = FixtureNames()
    errors: FixtureErrors = FixtureErrors()
    data: FixtureData = FixtureData()
    patterns: FixturePatterns = FixturePatterns()
    numeric: FixtureNumericValues = FixtureNumericValues()

    def to_dict(self) -> FlextTypes.Types.ConfigurationMapping:
        """Convert to flat dictionary for backward compatibility."""
        result: dict[str, FlextTypes.GeneralValueType] = {}

        # Identifiers
        result.update(
            {
                "test_user_id": self.identifiers.user_id,
                "test_session_id": self.identifiers.session_id,
                "test_service_name": self.identifiers.service_name,
                "test_operation_id": self.identifiers.operation_id,
                "test_request_id": self.identifiers.request_id,
                "test_correlation_id": self.identifiers.correlation_id,
            },
        )

        # Names
        result.update(
            {
                "test_module_name": self.names.module_name,
                "test_handler_name": self.names.handler_name,
                "test_chain_name": self.names.chain_name,
                "test_command_type": self.names.command_type,
                "test_query_type": self.names.query_type,
                "test_logger_name": self.names.logger_name,
                "test_app_name": self.names.app_name,
                "test_validation_app": self.names.validation_app,
                "test_source_service": self.names.source_service,
            },
        )

        # Errors
        result.update(
            {
                "test_error_code": self.errors.error_code,
                "test_validation_error": self.errors.validation_error,
                "test_operation_error": self.errors.operation_error,
                "test_config_error": self.errors.config_error,
                "test_timeout_error": self.errors.timeout_error,
            },
        )

        # Data
        result.update(
            {
                "test_field_name": self.data.field_name,
                "test_config_key": self.data.config_key,
                "test_username": self.data.username,
                "test_email": self.data.email,
                "test_password": self.data.password,
                "test_string_value": self.data.string_value,
                "test_input_data": self.data.input_data,
                "test_request_data": self.data.request_data,
                "test_result_data": self.data.result_data,
                "test_message": self.data.message,
            },
        )

        # Patterns
        result.update(
            {
                "test_slug_input": self.patterns.slug_input,
                "test_slug_expected": self.patterns.slug_expected,
                "test_uuid_format": self.patterns.uuid_format,
            },
        )

        # Numeric
        result.update(
            {
                "test_port": self.numeric.port,
                "test_timeout": self.numeric.timeout,
                "test_retry_count": self.numeric.retry_count,
                "test_batch_size": self.numeric.batch_size,
            },
        )

        return result


class FixtureConstantsFactories:
    """Factories for test constants collections."""

    @staticmethod
    def create_basic_constants() -> FixtureConstantsCollection:
        """Create basic test constants."""
        return FixtureConstantsCollection()

    @staticmethod
    def create_integration_constants() -> FixtureConstantsCollection:
        """Create constants for integration testing."""
        return FixtureConstantsCollection(
            identifiers=FixtureIdentifiers(
                request_id="int-request-789",
                correlation_id="int-corr-456",
            ),
            names=FixtureNames(
                app_name="integration-app",
                module_name="integration_module",
            ),
        )

    @staticmethod
    def create_performance_constants() -> FixtureConstantsCollection:
        """Create constants for performance testing."""
        return FixtureConstantsCollection(
            numeric=FixtureNumericValues(batch_size=1000, timeout=60),
        )


# Backward compatibility function
def get_test_constants() -> FlextTypes.Types.ConfigurationMapping:
    """Provide centralized test constants for all tests (backward compatibility).

    Returns:
        Dict containing test constants and data

    """
    return FixtureConstantsFactories.create_basic_constants().to_dict()
