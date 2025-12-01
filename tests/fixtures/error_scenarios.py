"""Error scenarios fixtures for flext-core tests using advanced patterns.

Provides comprehensive error scenario factories for testing error handling,
validation failures, timeouts, and edge cases using Python 3.13 dataclasses.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flext_core.typings import FlextTypes

from ..helpers.constants import TestConstants


@dataclass(frozen=True, slots=True)
class ErrorScenario:
    """Factory for error scenario data structures."""

    type: str
    message: str
    code: str
    context: FlextTypes.Types.ConfigurationMapping = field(default_factory=dict)
    field: str | None = None
    config_key: str | None = None
    service: str | None = None
    operation: str | None = None
    handler: str | None = None
    user: str | None = None

    def to_dict(self) -> FlextTypes.Types.ConfigurationMapping:
        """Convert to dictionary format for compatibility."""
        result: FlextTypes.Types.ConfigurationMapping = {
            "type": self.type,
            "message": self.message,
            "code": self.code,
            "context": dict(self.context),
        }

        # Add optional fields if present
        if self.field is not None:
            result["field"] = self.field
        if self.config_key is not None:
            result["config_key"] = self.config_key
        if self.service is not None:
            result["service"] = self.service
        if self.operation is not None:
            result["operation"] = self.operation
        if self.handler is not None:
            result["handler"] = self.handler
        if self.user is not None:
            result["user"] = self.user

        return result


class ErrorScenarioFactories:
    """Centralized factories for error scenarios."""

    @staticmethod
    def validation_error(
        field: str = TestConstants.Strings.BASIC_WORD,
        input_data: str = TestConstants.Strings.INVALID_EMAIL,
    ) -> ErrorScenario:
        """Create validation error scenario."""
        return ErrorScenario(
            type="ValidationError",
            message="Invalid input data",
            code="VAL_001",
            field=field,
            context={"input": input_data},
        )

    @staticmethod
    def configuration_error(
        config_key: str = "database_url",
        section: str = "database",
    ) -> ErrorScenario:
        """Create configuration error scenario."""
        return ErrorScenario(
            type="ConfigurationError",
            message="Missing required configuration",
            code="CFG_001",
            config_key=config_key,
            context={"section": section},
        )

    @staticmethod
    def connection_error(
        service: str = "test_service",
        host: str = "localhost",
        port: int = 8080,
    ) -> ErrorScenario:
        """Create connection error scenario."""
        return ErrorScenario(
            type="ConnectionError",
            message="Failed to connect to service",
            code="CONN_001",
            service=service,
            context={"host": host, "port": port},
        )

    @staticmethod
    def timeout_error(
        operation: str = "test_operation",
        timeout: int = 30,
        elapsed: int = 35,
    ) -> ErrorScenario:
        """Create timeout error scenario."""
        return ErrorScenario(
            type="TimeoutError",
            message="Operation timed out",
            code="TIMEOUT_001",
            operation=operation,
            context={"timeout": timeout, "elapsed": elapsed},
        )

    @staticmethod
    def processing_error(
        handler: str = "test_handler",
        stage: str = "validation",
        input_size: int = 1024,
    ) -> ErrorScenario:
        """Create processing error scenario."""
        return ErrorScenario(
            type="ProcessingError",
            message="Failed to process request",
            code="PROC_001",
            handler=handler,
            context={"stage": stage, "input_size": input_size},
        )

    @staticmethod
    def authentication_error(
        user: str = "test_user",
        method: str = "token",
        reason: str = "expired",
    ) -> ErrorScenario:
        """Create authentication error scenario."""
        return ErrorScenario(
            type="AuthenticationError",
            message="Authentication failed",
            code="AUTH_001",
            user=user,
            context={"method": method, "reason": reason},
        )

    @staticmethod
    def get_all_scenarios() -> dict[str, ErrorScenario]:
        """Get all predefined error scenarios."""
        return {
            "validation_error": ErrorScenarioFactories.validation_error(),
            "configuration_error": ErrorScenarioFactories.configuration_error(),
            "connection_error": ErrorScenarioFactories.connection_error(),
            "timeout_error": ErrorScenarioFactories.timeout_error(),
            "processing_error": ErrorScenarioFactories.processing_error(),
            "authentication_error": ErrorScenarioFactories.authentication_error(),
        }

    @staticmethod
    def get_all_scenarios_dict() -> dict[str, FlextTypes.Types.ConfigurationMapping]:
        """Get all scenarios in dict format for backward compatibility."""
        return {
            key: scenario.to_dict()
            for key, scenario in ErrorScenarioFactories.get_all_scenarios().items()
        }


# Backward compatibility function
def get_test_error_scenarios() -> dict[str, FlextTypes.Types.ConfigurationMapping]:
    """Provide common error scenarios for testing (backward compatibility).

    Returns:
        Dict containing various error scenarios in legacy format

    """
    return ErrorScenarioFactories.get_all_scenarios_dict()
