"""FLEXT Core Exceptions - Complete Enterprise Exception Hierarchy.

This module provides the complete exception hierarchy for the entire FLEXT
ecosystem. All FLEXT modules MUST use these exceptions instead of creating
their own, ensuring consistency and proper error handling across the ecosystem.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# BASE EXCEPTIONS - Root of all FLEXT exceptions
# =============================================================================


class FlextError(Exception):
    """Base exception for all FLEXT-related errors.

    This is the root exception class for the entire FLEXT ecosystem.
    All other FLEXT exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlextError.

        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class FlextCriticalError(FlextError):
    """Critical errors that require immediate attention."""

    def __init__(self, message: str, **kwargs: object) -> None:
        error_code = kwargs.pop("error_code", "FLEXT_CRITICAL")
        details = kwargs.pop("details", None)
        super().__init__(
            message,
            str(error_code),
            details if isinstance(details, dict) else None,
        )


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================


class FlextConfigError(FlextError):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs: object) -> None:
        error_code = kwargs.pop("error_code", "FLEXT_CONFIG")
        details = kwargs.pop("details", None)
        super().__init__(
            message,
            str(error_code),
            details if isinstance(details, dict) else None,
        )


class FlextConfigurationError(FlextError):
    """Configuration-related errors.

    Raised when there are issues with configuration files, environment
    variables, or settings.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlextConfigurationError.

        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_value: Configuration value that caused the error
            details: Optional additional error details
        """
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key
        self.config_value = config_value

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "config_key": self.config_key,
                "config_value": self.config_value,
            },
        )
        return base_dict


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================


class FlextValidationError(FlextError):
    """Validation-related errors.

    Raised when data validation fails, such as invalid input parameters,
    malformed data, or business rule violations.
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlextValidationError.

        Args:
            message: Error message
            field: Field that failed validation
            value: Value that failed validation
            details: Optional additional error details
        """
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "field": self.field,
                "value": self.value,
            },
        )
        return base_dict


# =============================================================================
# PROCESSING EXCEPTIONS
# =============================================================================


class FlextProcessingError(FlextError):
    """Processing-related errors.

    Raised when data processing fails, such as transformation errors,
    parsing failures, or processing pipeline issues.
    """

    def __init__(
        self,
        message: str,
        entry_dn: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlextProcessingError.

        Args:
            message: Error message
            entry_dn: DN of the entry being processed
            operation: Operation that failed
            details: Optional additional error details
        """
        super().__init__(message, "PROCESSING_ERROR", details)
        self.entry_dn = entry_dn
        self.operation = operation

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "entry_dn": self.entry_dn,
                "operation": self.operation,
            },
        )
        return base_dict


# =============================================================================
# CONNECTION EXCEPTIONS
# =============================================================================


class FlextConnectionError(FlextError):
    """Connection-related errors.

    Raised when there are issues with network connections, database
    connections, or external service connections.
    """

    def __init__(
        self,
        message: str,
        host: str | None = None,
        port: int | None = None,
        service: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlextConnectionError.

        Args:
            message: Error message
            host: Host that failed to connect
            port: Port that failed to connect
            service: Service that failed to connect
            details: Optional additional error details
        """
        super().__init__(message, "CONNECTION_ERROR", details)
        self.host = host
        self.port = port
        self.service = service

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "host": self.host,
                "port": self.port,
                "service": self.service,
            },
        )
        return base_dict


# =============================================================================
# AUTHENTICATION EXCEPTIONS
# =============================================================================


class FlextAuthenticationError(FlextError):
    """Authentication-related errors.

    Raised when authentication fails, such as invalid credentials,
    expired tokens, or permission issues.
    """

    def __init__(
        self,
        message: str,
        username: str | None = None,
        service: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlextAuthenticationError.

        Args:
            message: Error message
            username: Username that failed authentication
            service: Service that failed authentication
            details: Optional additional error details
        """
        super().__init__(message, "AUTHENTICATION_ERROR", details)
        self.username = username
        self.service = service

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "username": self.username,
                "service": self.service,
            },
        )
        return base_dict


# =============================================================================
# MIGRATION EXCEPTIONS
# =============================================================================


class FlextMigrationError(FlextError):
    """Migration-related errors.

    Raised when data migration fails, such as schema conversion errors,
    data transformation failures, or migration pipeline issues.
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        source: str | None = None,
        target: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlextMigrationError.

        Args:
            message: Error message
            operation: Migration operation that failed
            source: Source system or format
            target: Target system or format
            details: Optional additional error details
        """
        super().__init__(message, "MIGRATION_ERROR", details)
        self.operation = operation
        self.source = source
        self.target = target

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "operation": self.operation,
                "source": self.source,
                "target": self.target,
            },
        )
        return base_dict


# =============================================================================
# SCHEMA EXCEPTIONS
# =============================================================================


class FlextSchemaError(FlextError):
    """Schema-related errors.

    Raised when there are issues with data schemas, such as schema
    validation failures, schema conversion errors, or schema
    compatibility issues.
    """

    def __init__(
        self,
        message: str,
        schema_type: str | None = None,
        schema_version: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlextSchemaError.

        Args:
            message: Error message
            schema_type: Type of schema that caused the error
            schema_version: Version of schema that caused the error
            details: Optional additional error details
        """
        super().__init__(message, "SCHEMA_ERROR", details)
        self.schema_type = schema_type
        self.schema_version = schema_version

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "schema_type": self.schema_type,
                "schema_version": self.schema_version,
            },
        )
        return base_dict


# =============================================================================
# TIMEOUT EXCEPTIONS
# =============================================================================


class FlextTimeoutError(FlextError):
    """Timeout-related errors.

    Raised when operations exceed their timeout limits, such as
    connection timeouts, operation timeouts, or processing timeouts.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FlextTimeoutError.

        Args:
            message: Error message
            timeout_seconds: Timeout duration in seconds
            operation: Operation that timed out
            details: Optional additional error details
        """
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "timeout_seconds": self.timeout_seconds,
                "operation": self.operation,
            },
        )
        return base_dict
