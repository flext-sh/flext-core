"""Common configuration validators for FLEXT projects.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Provides comprehensive validation functions for configuration values
used across FLEXT projects with proper error handling and documentation.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import field_validator

if TYPE_CHECKING:
    from collections.abc import Callable


def _raise_url_error(message: str) -> None:
    """Raise URL validation error.

    Args:
        message: Error message to include

    Raises:
        ValueError: Always raised with the provided message

    """
    raise ValueError(message)


def _raise_database_error(message: str) -> None:
    """Raise database URL validation error.

    Args:
        message: Error message to include

    Raises:
        ValueError: Always raised with the provided message

    """
    raise ValueError(message)


def validate_url(value: str, *, require_tld: bool = True) -> str:
    """Validate URL format and structure.

    Args:
        value: URL string to validate
        require_tld: Whether to require top-level domain

    Returns:
        Validated URL string

    Raises:
        ValueError: If URL is invalid

    """
    if not value:
        _raise_url_error("URL cannot be empty")

    try:
        result = urlparse(value)
    except Exception as e:
        msg = f"Invalid URL: {e}"
        raise ValueError(msg) from e
    else:
        if not all([result.scheme, result.netloc]):
            _raise_url_error("Invalid URL format")

        if require_tld and "." not in result.netloc:
            _raise_url_error("URL must contain a valid domain")

        return value


def validate_database_url(value: str) -> str:
    """Validate database URL format and scheme.

    Args:
        value: Database URL string to validate

    Returns:
        Validated database URL string

    Raises:
        ValueError: If database URL is invalid

    """
    if not value:
        _raise_database_error("Database URL cannot be empty")

    # Common database URL patterns
    db_patterns = [
        r"^postgresql://",
        r"^postgres://",
        r"^mysql://",
        r"^sqlite://",
        r"^oracle://",
        r"^mssql://",
    ]

    if not any(re.match(pattern, value, re.IGNORECASE) for pattern in db_patterns):
        _raise_database_error(
            "Invalid database URL. Must start with a valid database scheme "
            "(postgresql://, mysql://, sqlite://, etc.)",
        )

    # Basic URL validation
    try:
        result = urlparse(value)
    except Exception as e:
        msg = f"Invalid database URL: {e}"
        raise ValueError(msg) from e
    else:
        if not result.scheme:
            _raise_database_error("Database URL must include a scheme")

        # SQLite is special case - doesn't need host
        if not result.scheme.lower().startswith("sqlite") and not result.netloc:
            _raise_database_error("Database URL must include a host")

        return value


def validate_port(value: int | str) -> int:
    """Validate port number.

    Args:
        value: Port number as int or string

    Returns:
        Validated port number as integer

    Raises:
        TypeError: If value is not convertible to int
        ValueError: If port is out of valid range

    """
    if not isinstance(value, int):
        try:
            port_value = int(value)
        except (TypeError, ValueError) as e:
            msg = "Port must be an integer"
            raise TypeError(msg) from e
    else:
        port_value = value

    if port_value < 1 or port_value > 65535:
        msg = "Port must be between 1 and 65535"
        raise ValueError(msg)

    return port_value


def validate_timeout(value: float | str, *, min_value: float = 0.0) -> float:
    """Validate timeout value.

    Args:
        value: Timeout value as float or string
        min_value: Minimum allowed timeout value

    Returns:
        Validated timeout as float

    Raises:
        TypeError: If value is not convertible to float
        ValueError: If timeout is below minimum value

    """
    try:
        timeout = float(value)
    except (TypeError, ValueError) as e:
        msg = "Timeout must be a number"
        raise TypeError(msg) from e

    if timeout < min_value:
        msg = f"Timeout must be at least {min_value} seconds"
        raise ValueError(msg)

    return timeout


# Pydantic field validators for common use cases
class CommonValidators:
    """Collection of common field validators for Pydantic models."""

    @field_validator("url", "api_url", "webhook_url")
    @classmethod
    def validate_url_field(cls, v: str | None) -> str | None:
        """Validate URL fields.

        Args:
            v: URL value to validate

        Returns:
            Validated URL or None

        """
        if v is None:
            return v
        return validate_url(v)

    @field_validator("database_url", "db_url", "connection_string")
    @classmethod
    def validate_database_url_field(cls, v: str | None) -> str | None:
        """Validate database URL fields.

        Args:
            v: Database URL value to validate

        Returns:
            Validated database URL or None

        """
        if v is None:
            return v
        return validate_database_url(v)

    @field_validator("port", "server_port", "api_port")
    @classmethod
    def validate_port_field(cls, v: int | str | None) -> int | None:
        """Validate port fields.

        Args:
            v: Port value to validate

        Returns:
            Validated port or None

        """
        if v is None:
            return v
        return validate_port(v)

    @field_validator("timeout", "read_timeout", "write_timeout", "connect_timeout")
    @classmethod
    def validate_timeout_field(cls, v: float | str | None) -> float | None:
        """Validate timeout fields.

        Args:
            v: Timeout value to validate

        Returns:
            Validated timeout or None

        """
        if v is None:
            return v
        return validate_timeout(v)

    @field_validator("email", "user_email", "REDACTED_LDAP_BIND_PASSWORD_email")
    @classmethod
    def validate_email_field(cls, v: str | None) -> str | None:
        """Validate email fields.

        Args:
            v: Email value to validate

        Returns:
            Validated email in lowercase or None

        Raises:
            ValueError: If email format is invalid

        """
        if v is None:
            return v

        email_regex = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )

        if not email_regex.match(v):
            msg = "Invalid email format"
            raise ValueError(msg)

        return v.lower()

    @field_validator("log_level", "logging_level")
    @classmethod
    def validate_log_level_field(cls, v: str) -> str:
        """Validate log level fields.

        Args:
            v: Log level value to validate

        Returns:
            Validated log level in uppercase

        Raises:
            ValueError: If log level is invalid

        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()

        if v_upper not in valid_levels:
            msg = f"Invalid log level. Must be one of: {', '.join(valid_levels)}"
            raise ValueError(msg)

        return v_upper


# Validator decorators for custom use
def url_validator(*, require_tld: bool = True) -> Callable[[str], str]:
    """Create URL validator decorator.

    Args:
        require_tld: Whether to require top-level domain

    Returns:
        Validator function decorator

    """

    def validator(value: str) -> str:
        """Validate URL value.

        Args:
            value: URL string to validate

        Returns:
            Validated URL string

        """
        return validate_url(value, require_tld=require_tld)

    return field_validator("url", mode="after")(validator)


def port_validator() -> Callable[[int | str], int]:
    """Create port validator decorator.

    Returns:
        Port validator function decorator

    """
    return field_validator("port", mode="after")(validate_port)


def timeout_validator(*, min_value: float = 0.0) -> Callable[[float | str], float]:
    """Create timeout validator decorator.

    Args:
        min_value: Minimum allowed timeout value

    Returns:
        Timeout validator function decorator

    """

    def validator(value: float | str) -> float:
        """Validate timeout value.

        Args:
            value: Timeout value to validate

        Returns:
            Validated timeout as float

        """
        return validate_timeout(value, min_value=min_value)

    return field_validator("timeout", mode="after")(validator)


__all__ = [
    "CommonValidators",
    "port_validator",
    "timeout_validator",
    "url_validator",
    "validate_database_url",
    "validate_port",
    "validate_timeout",
    "validate_url",
]
