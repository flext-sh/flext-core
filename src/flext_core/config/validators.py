"""Common configuration validators for FLEXT projects.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Provides comprehensive validation functions for configuration values
used across FLEXT projects with proper error handling and documentation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from urllib.parse import urlparse

from pydantic import field_validator

if TYPE_CHECKING:
    from collections.abc import Callable


def validate_url(value: Any, *, require_tld: bool = True) -> str:
    """Validate URL format and structure."""
    if not isinstance(value, str):
        msg = f"URL must be a string, got {type(value).__name__}"
        raise TypeError(msg)

    if not value:
        msg = "URL cannot be empty"
        raise ValueError(msg)

    # Basic URL validation
    if not value.startswith(("http://", "https://", "ftp://", "ftps://")):
        msg = "Invalid URL format"
        raise ValueError(msg)

    try:
        # Parse URL using urlparse
        parsed = urlparse(value)

        if not parsed.netloc:
            msg = "Invalid URL format"
            raise ValueError(msg)

        # Check TLD requirement
        # Extract hostname from netloc (remove user:pass@ and :port)
        netloc = parsed.netloc
        if "@" in netloc:
            # Remove user:pass@ part
            netloc = netloc.split("@")[-1]
        domain = netloc.split(":")[0]  # Remove port if present

        if require_tld and domain in ("localhost", "127.0.0.1"):
            msg = "URL must contain a valid domain"
            raise ValueError(msg)
        if require_tld and "." not in domain:
            msg = "URL must contain a valid domain"
            raise ValueError(msg)

    except ValueError:
        # Re-raise our own ValueError exceptions
        raise
    except Exception as e:
        # Catch other exceptions (like urlparse errors) and wrap them
        msg = f"Invalid URL: {e}"
        raise ValueError(msg) from e

    return value


def validate_database_url(value: Any) -> str:
    """Validate database URL format and scheme."""
    if not isinstance(value, str):
        msg = f"Database URL must be a string, got {type(value).__name__}"
        raise TypeError(msg)

    if not value:
        msg = "Database URL cannot be empty"
        raise ValueError(msg)

    # Supported database schemes
    supported_schemes = [
        "postgresql",
        "postgres",  # PostgreSQL alias
        "mysql",
        "sqlite",
        "oracle",
        "mongodb",
        "redis",
        "postgresql+asyncpg",
        "mysql+aiomysql",
        "sqlite+aiosqlite",
    ]

    try:
        # Parse URL using urlparse
        parsed = urlparse(value)

        if not parsed.scheme:
            msg = "Database URL must include a scheme"
            raise ValueError(msg)

        if parsed.scheme not in supported_schemes:
            msg = "Invalid database URL. Must start with a valid database scheme"
            raise ValueError(msg)

        # For non-file based databases, ensure we have a netloc (host)
        if parsed.scheme != "sqlite" and not parsed.netloc:
            msg = "Database URL must include a host"
            raise ValueError(msg)

    except ValueError:
        # Re-raise our own ValueError exceptions
        raise
    except Exception as e:
        # Catch other exceptions (like urlparse errors) and wrap them
        msg = f"Invalid database URL: {e}"
        raise ValueError(msg) from e

    return value


def validate_port(value: int | str) -> int:
    """Validate port number."""
    if isinstance(value, str):
        try:
            port_value = int(value)
        except (TypeError, ValueError) as e:
            msg = "Port must be an integer"
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
        msg = "Timeout must be a number"
        raise TypeError(msg) from e

    if timeout < min_value:
        msg = f"Timeout must be at least {min_value} seconds"
        raise ValueError(msg)

    return timeout


def validate_email(email: Any) -> str:
    """Validate email format."""
    if not isinstance(email, str):
        msg = f"Email must be a string, got {type(email).__name__}"
        raise TypeError(msg)

    if not email:
        msg = "Email cannot be empty"
        raise ValueError(msg)

    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, email):
        raise ValueError(f"Invalid email format: {email}")

    return email.lower()


def validate_log_level(level: Any) -> str:
    """Validate log level."""
    if not isinstance(level, str):
        msg = f"Log level must be a string, got {type(level).__name__}"
        raise TypeError(msg)

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

    @classmethod
    @field_validator("url", check_fields=False)
    def validate_url_field(cls, v: str | None) -> str | None:
        """Validate URL field."""
        if v is None:
            return v
        return validate_url(v)

    @classmethod
    @field_validator("database_url", check_fields=False)
    def validate_database_url_field(cls, v: str | None) -> str | None:
        """Validate database URL field."""
        if v is None:
            return v
        return validate_database_url(v)

    @classmethod
    @field_validator("port", check_fields=False)
    def validate_port_field(cls, v: int | str | None) -> int | None:
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

    @classmethod
    @field_validator("timeout", check_fields=False)
    def validate_timeout_field(cls, v: float | str | None) -> float | None:
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

    @classmethod
    @field_validator("email", check_fields=False)
    def validate_email_field(cls, v: str | None) -> str | None:
        """Validate email field."""
        if v is None:
            return v
        return validate_email(v)

    @classmethod
    @field_validator("log_level", check_fields=False)
    def validate_log_level_field(cls, v: str | None) -> str | None:
        """Validate log level field."""
        if v is None:
            return v
        return validate_log_level(v)


# ==============================================================================
# DECORATOR FUNCTIONS FOR PYDANTIC MODELS
# ==============================================================================


def url_validator(*, require_tld: bool = True) -> Callable[[Any], str | None]:
    """Create a URL field validator decorator for Pydantic models."""

    def validator_func(v: str | None) -> str | None:
        if v is None:
            return v
        return validate_url(v, require_tld=require_tld)

    return field_validator("url", check_fields=False)(validator_func)


def port_validator() -> Callable[[Any], int | None]:
    """Create a port field validator decorator for Pydantic models."""

    def validator_func(v: int | str | None) -> int | None:
        if v is None:
            return v
        return validate_port(v)

    return field_validator("port", check_fields=False)(validator_func)


def timeout_validator(*, min_value: float = 0.0) -> Callable[[Any], float | None]:
    """Create a timeout field validator decorator for Pydantic models."""

    def validator_func(v: float | str | None) -> float | None:
        if v is None:
            return v
        return validate_timeout(v, min_value=min_value)

    return field_validator("timeout", check_fields=False)(validator_func)


# Static methods for direct testing - create callable versions without decorators
def validate_url_field(v: str | None) -> str | None:
    """Validate URL field - callable version for testing."""
    if v is None:
        return v
    if not v:
        msg = "URL cannot be empty"
        raise ValueError(msg)
    return validate_url(v)

def validate_database_url_field(v: str | None) -> str | None:
    """Validate database URL field - callable version for testing."""
    if v is None:
        return v
    if not v:
        msg = "Database URL cannot be empty"
        raise ValueError(msg)
    return validate_database_url(v)

def validate_port_field(v: int | str | None) -> int | None:
    """Validate port field - callable version for testing."""
    if v is None:
        return v
    if isinstance(v, str):
        try:
            v = int(v)
        except ValueError as e:
            msg = f"Invalid port: {v}"
            raise ValueError(msg) from e
    return validate_port(v)

def validate_timeout_field(v: float | str | None) -> float | None:
    """Validate timeout field - callable version for testing."""
    if v is None:
        return v
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError as e:
            msg = f"Invalid timeout: {v}"
            raise ValueError(msg) from e
    return validate_timeout(v)

def validate_email_field(v: str | None) -> str | None:
    """Validate email field - callable version for testing."""
    if v is None:
        return v
    if not v:
        msg = "Invalid email format"
        raise ValueError(msg)
    return validate_email(v)

# Direct access functions available as standalone functions above
# These provide the same functionality without dynamic class modification

# Pydantic field validators - for backward compatibility
url_validator_field = validate_url_field
port_validator_field = validate_port_field
timeout_validator_field = validate_timeout_field


__all__ = [
    "CommonValidators",
    "port_validator",
    "port_validator_field",
    "timeout_validator",
    "timeout_validator_field",
    "url_validator",
    "url_validator_field",
    "validate_database_url",
    "validate_database_url_field",
    "validate_email",
    "validate_email_field",
    "validate_log_level",
    "validate_port",
    "validate_port_field",
    "validate_timeout",
    "validate_timeout_field",
    "validate_url",
    "validate_url_field",
]
