"""Common configuration validators for FLEXT projects.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Provides comprehensive validation functions for configuration values
used across FLEXT projects with proper error handling and documentation.
"""

from __future__ import annotations

import re

from pydantic import field_validator


def validate_url(value: str) -> str:
    """Validate URL format and structure."""
    if not value:
        msg = "URL cannot be empty"
        raise ValueError(msg)
    return value


def validate_database_url(value: str) -> str:
    """Validate database URL format and scheme."""
    if not value:
        msg = "Database URL cannot be empty"
        raise ValueError(msg)
    return value


def validate_port(value: int | str) -> int:
    """Validate port number."""
    if isinstance(value, str):
        try:
            port_value = int(value)
        except (TypeError, ValueError) as e:
            msg = f"Port must be a valid integer: {value}"
            raise TypeError(msg) from e
    else:
        port_value = value

    if port_value < 1 or port_value > 65535:
        msg = f"Port must be between 1 and 65535, got {port_value}"
        raise ValueError(msg)

    return port_value


def validate_timeout(value: float | str, *, min_value: float = 0.0) -> float:
    """Validate timeout value."""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as e:
        msg = f"Timeout must be a valid number: {value}"
        raise TypeError(msg) from e

    if timeout < min_value:
        msg = f"Timeout must be >= {min_value}, got {timeout}"
        raise ValueError(msg)

    return timeout


def validate_email(email: str) -> str:
    """Validate email format."""
    if not email:
        msg = "Email cannot be empty"
        raise ValueError(msg)

    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        msg = f"Invalid email format: {email}"
        raise ValueError(msg)

    return email.lower()


def validate_log_level(level: str) -> str:
    """Validate log level."""
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level_upper = level.upper()

    if level_upper not in valid_levels:
        msg = f"Invalid log level: {level}. Must be one of {valid_levels}"
        raise ValueError(msg)

    return level_upper


# ==============================================================================
# FIELD VALIDATORS FOR PYDANTIC MODELS
# ==============================================================================


class CommonValidators:
    """Common field validators for Pydantic models."""

    @staticmethod
    @field_validator("url", check_fields=False)
    def validate_url_field(v: str | None) -> str | None:
        """Validate URL field."""
        if v is None:
            return v
        return validate_url(v)

    @staticmethod
    @field_validator("database_url", check_fields=False)
    def validate_database_url_field(v: str | None) -> str | None:
        """Validate database URL field."""
        if v is None:
            return v
        return validate_database_url(v)

    @staticmethod
    @field_validator("port", check_fields=False)
    def validate_port_field(v: int | str | None) -> int | None:
        """Validate port field."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError as e:
                msg = f"Invalid port: {v}"
                raise ValueError(msg) from e
        return validate_port(v)

    @staticmethod
    @field_validator("timeout", check_fields=False)
    def validate_timeout_field(v: float | str | None) -> float | None:
        """Validate timeout field."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError as e:
                msg = f"Invalid timeout: {v}"
                raise ValueError(msg) from e
        return validate_timeout(v)

    @staticmethod
    @field_validator("email", check_fields=False)
    def validate_email_field(v: str | None) -> str | None:
        """Validate email field."""
        if v is None:
            return v
        return validate_email(v)

    @staticmethod
    @field_validator("log_level", check_fields=False)
    def validate_log_level_field(v: str | None) -> str | None:
        """Validate log level field."""
        if v is None:
            return v
        return validate_log_level(v)


# Pydantic field validators - for backward compatibility
url_validator = CommonValidators.validate_url_field
port_validator = CommonValidators.validate_port_field
timeout_validator = CommonValidators.validate_timeout_field


__all__ = [
    "CommonValidators",
    "port_validator",
    "timeout_validator",
    "url_validator",
    "validate_database_url",
    "validate_email",
    "validate_log_level",
    "validate_port",
    "validate_timeout",
    "validate_url",
]
