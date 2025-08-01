"""Singer Protocol Base Classes - DRY implementation for all Singer projects.

This module provides base exception classes and utilities for Singer taps and targets
to eliminate code duplication across flext-tap-* and flext-target-* projects.

Follows SOLID principles and DRY methodology to centralize common Singer patterns.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.exceptions import (
    FlextAuthenticationError,
    FlextConfigurationError,
    FlextConnectionError,
    FlextError,
    FlextProcessingError,
    FlextValidationError,
)


class FlextSingerError(FlextError):
    """Base exception for all Singer operations (taps and targets)."""

    def __init__(
        self,
        message: str = "Singer operation error",
        component_type: str | None = None,
        stream_name: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize Singer error with context.

        Args:
            message: Error message
            component_type: Type of Singer component (tap/target/transform)
            stream_name: Name of the stream being processed
            **kwargs: Additional context

        """
        context = kwargs.copy()
        if component_type is not None:
            context["component_type"] = component_type
        if stream_name is not None:
            context["stream_name"] = stream_name

        super().__init__(message, error_code="SINGER_ERROR", context=context)


class FlextSingerConnectionError(FlextConnectionError):
    """Singer connection errors for taps and targets."""

    def __init__(
        self,
        message: str = "Singer connection failed",
        host: str | None = None,
        port: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize Singer connection error with context."""
        context = kwargs.copy()
        if host is not None:
            context["host"] = host
        if port is not None:
            context["port"] = port

        super().__init__(f"Singer connection: {message}", **context)


class FlextSingerAuthenticationError(FlextAuthenticationError):
    """Singer authentication errors for taps and targets."""

    def __init__(
        self,
        message: str = "Singer authentication failed",
        username: str | None = None,
        auth_method: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize Singer authentication error with context."""
        context = kwargs.copy()
        if username is not None:
            context["username"] = username
        if auth_method is not None:
            context["auth_method"] = auth_method

        super().__init__(f"Singer auth: {message}", **context)


class FlextSingerValidationError(FlextValidationError):
    """Singer validation errors for taps and targets."""

    def __init__(
        self,
        message: str = "Singer validation failed",
        field: str | None = None,
        value: object = None,
        record_id: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize Singer validation error with context."""
        validation_details = {}
        if field is not None:
            validation_details["field"] = field
        if value is not None:
            validation_details["value"] = str(value)[:100]  # Truncate long values

        context = kwargs.copy()
        if record_id is not None:
            context["record_id"] = record_id

        super().__init__(
            f"Singer validation: {message}",
            validation_details=validation_details,
            context=context,
        )


class FlextSingerConfigurationError(FlextConfigurationError):
    """Singer configuration errors for taps and targets."""

    def __init__(
        self,
        message: str = "Singer configuration error",
        config_key: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize Singer configuration error with context."""
        context = kwargs.copy()
        if config_key is not None:
            context["config_key"] = config_key

        super().__init__(f"Singer config: {message}", **context)


class FlextSingerProcessingError(FlextProcessingError):
    """Singer processing errors for taps and targets."""

    def __init__(
        self,
        message: str = "Singer processing failed",
        operation: str | None = None,
        record_count: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize Singer processing error with context."""
        context = kwargs.copy()
        if operation is not None:
            context["operation"] = operation
        if record_count is not None:
            context["record_count"] = record_count

        super().__init__(f"Singer processing: {message}", **context)


# Base classes for specific Singer component types


class FlextTapError(FlextSingerError):
    """Base exception for Singer tap operations."""

    def __init__(
        self,
        message: str = "Tap operation error",
        source_system: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize tap error with context."""
        context = kwargs.copy()
        if source_system is not None:
            context["source_system"] = source_system

        super().__init__(message, component_type="tap", **context)


class FlextTargetError(FlextSingerError):
    """Base exception for Singer target operations."""

    def __init__(
        self,
        message: str = "Target operation error",
        destination_system: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize target error with context."""
        context = kwargs.copy()
        if destination_system is not None:
            context["destination_system"] = destination_system

        super().__init__(message, component_type="target", **context)


class FlextTransformError(FlextSingerError):
    """Base exception for Singer transform operations."""

    def __init__(
        self,
        message: str = "Transform operation error",
        transform_name: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize transform error with context."""
        context = kwargs.copy()
        if transform_name is not None:
            context["transform_name"] = transform_name

        super().__init__(message, component_type="transform", **context)


# Factory functions removed - use direct exception class inheritance instead


__all__ = [
    "FlextSingerAuthenticationError",
    "FlextSingerConfigurationError",
    "FlextSingerConnectionError",
    # Base Singer exceptions
    "FlextSingerError",
    "FlextSingerProcessingError",
    "FlextSingerValidationError",
    # Component-specific base exceptions
    "FlextTapError",
    "FlextTargetError",
    "FlextTransformError",
]
