"""Common validation patterns - consolidated from multiple projects.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module consolidates validation patterns found across all FLEXT projects.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any

from pydantic import field_validator

if TYPE_CHECKING:
    from collections.abc import Callable


class CommonValidators:
    """Common field validators consolidated from across projects."""

    @field_validator("port", check_fields=False)
    @classmethod
    def validate_port(cls: type[Any], v: int) -> int:
        """Validate port number range - used in 8+ projects."""
        if not 1 <= v <= 65535:
            msg = "Port must be between 1 and 65535"
            raise ValueError(msg)
        return v

    @field_validator("base_url", "url", "server_url", check_fields=False)
    @classmethod
    def validate_url(cls: type[Any], v: Any) -> str:
        """Validate URL format - used in 6+ projects."""
        if not isinstance(v, str):
            msg = "URL must be a string"
            raise TypeError(msg)
        if not v.startswith(("http://", "https://")):
            msg = "URL must start with http:// or https://"
            raise ValueError(msg)
        return v.rstrip("/")

    @field_validator(
        "timeout",
        "connection_timeout",
        "request_timeout",
        check_fields=False,
    )
    @classmethod
    def validate_timeout(cls: type[Any], v: Any) -> int | float:
        """Validate timeout values - used in 10+ projects."""
        if not isinstance(v, int | float) or v <= 0:
            msg = "Timeout must be a positive number"
            raise ValueError(msg)
        if v > 300:
            msg = "Timeout cannot exceed 300 seconds"
            raise ValueError(msg)
        return v

    @field_validator("auth_method", check_fields=False)
    @classmethod
    def validate_auth_method(cls: type[Any], v: Any) -> str:
        """Validate authentication method - used in 4+ projects."""
        allowed = {"basic", "token", "oauth", "bearer", "digest", "api_key"}
        if v.lower() not in allowed:
            msg = f"Auth method must be one of: {allowed}"
            raise ValueError(msg)
        return str(v.lower())

    @field_validator("batch_size", "page_size", check_fields=False)
    @classmethod
    def validate_batch_size(cls: type[Any], v: int) -> int:
        """Validate batch/page size - used in 7+ projects."""
        if not isinstance(v, int) or v <= 0:
            msg = "Batch size must be a positive integer"
            raise ValueError(msg)
        if v > 50000:
            msg = "Batch size cannot exceed 50000"
            raise ValueError(msg)
        return v

    @field_validator("email", check_fields=False)
    @classmethod
    def validate_email(cls: type[Any], v: Any) -> str:
        """Validate email format - used in 3+ projects."""
        if not isinstance(v, str):
            msg = "Email must be a string"
            raise TypeError(msg)
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            msg = "Invalid email format"
            raise ValueError(msg)
        return v.lower()

    @field_validator("host", "hostname", check_fields=False)
    @classmethod
    def validate_host(cls: type[Any], v: Any) -> str:
        """Validate hostname/IP format - used in 5+ projects."""
        if not isinstance(v, str):
            msg = "Host must be a string"
            raise TypeError(msg)
        if not v.strip():
            msg = "Host cannot be empty"
            raise ValueError(msg)

        # Basic validation for hostname or IP
        # Hostname must contain at least one letter (not all numeric like IP)
        hostname_pattern = (
            r"^(?=.*[a-zA-Z])[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?"
            r"(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$"
        )
        ip_pattern = (
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        )

        if not (
            re.match(hostname_pattern, v) or re.match(ip_pattern, v) or v == "localhost"
        ):
            msg = "Invalid hostname or IP address format"
            raise ValueError(msg)
        return v

    @field_validator("log_level", check_fields=False)
    @classmethod
    def validate_log_level(cls: type[Any], v: Any) -> str:
        """Validate log level - used in 6+ projects."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"}
        if v.upper() not in allowed_levels:
            msg = f"Log level must be one of: {allowed_levels}"
            raise ValueError(msg)
        return v.upper()

    @field_validator("environment", "env", check_fields=False)
    @classmethod
    def validate_environment(cls: type[Any], v: Any) -> str:
        """Validate environment name - used in 4+ projects."""
        allowed_envs = {
            "development",
            "dev",
            "testing",
            "test",
            "staging",
            "stage",
            "production",
            "prod",
        }
        if v.lower() not in allowed_envs:
            msg = f"Environment must be one of: {allowed_envs}"
            raise ValueError(msg)
        return v.lower()

    @field_validator("database_url", check_fields=False)
    @classmethod
    def validate_database_url(cls: type[Any], v: Any) -> str:
        """Validate database URL - used in 6+ projects."""
        if not isinstance(v, str):
            msg = "Database URL must be a string"
            raise TypeError(msg)

        supported_schemes = [
            "postgresql://",
            "mysql://",
            "sqlite://",
            "oracle://",
            "postgresql+asyncpg://",
            "mysql+aiomysql://",
            "sqlite+aiosqlite://",
        ]

        if not any(v.startswith(scheme) for scheme in supported_schemes):
            msg = f"Database URL must start with one of: {supported_schemes}"
            raise ValueError(
                msg,
            )
        return v

    @field_validator("redis_url", check_fields=False)
    @classmethod
    def validate_redis_url(cls: type[Any], v: Any) -> str:
        """Validate Redis URL - used in 3+ projects."""
        if not isinstance(v, str):
            msg = "Redis URL must be a string"
            raise TypeError(msg)
        if not v.startswith(("redis://", "rediss://", "unix://")):
            msg = "Redis URL must start with redis://, rediss://, or unix://"
            raise ValueError(
                msg,
            )
        return v

    @field_validator("ldap_uri", "ldap_url", check_fields=False)
    @classmethod
    def validate_ldap_uri(cls: type[Any], v: Any) -> str:
        """Validate LDAP URI - used in LDAP projects."""
        if not isinstance(v, str):
            msg = "LDAP URI must be a string"
            raise TypeError(msg)
        if not v.startswith(("ldap://", "ldaps://")):
            msg = "LDAP URI must start with ldap:// or ldaps://"
            raise ValueError(msg)
        return v.rstrip("/")

    @field_validator("percentage", check_fields=False)
    @classmethod
    def validate_percentage(cls: type[Any], v: Any) -> float:
        """Validate percentage value - used in monitoring configs."""
        if not isinstance(v, int | float):
            msg = "Percentage must be a number"
            raise TypeError(msg)
        if not 0 <= v <= 100:
            msg = "Percentage must be between 0 and 100"
            raise ValueError(msg)
        return float(v)

    @field_validator("sample_rate", "rate", check_fields=False)
    @classmethod
    def validate_sample_rate(cls: type[Any], v: Any) -> float:
        """Validate sample rate (0.0-1.0) - used in monitoring configs."""
        if not isinstance(v, int | float):
            msg = "Sample rate must be a number"
            raise TypeError(msg)
        if not 0.0 <= v <= 1.0:
            msg = "Sample rate must be between 0.0 and 1.0"
            raise ValueError(msg)
        return float(v)

    @field_validator("path", "file_path", check_fields=False)
    @classmethod
    def validate_path(cls: type[Any], v: Any) -> str:
        """Validate file/directory path - used in 4+ projects."""
        if not isinstance(v, str):
            msg = "Path must be a string"
            raise TypeError(msg)
        if not v.strip():
            msg = "Path cannot be empty"
            raise ValueError(msg)

        # Basic path validation - no null bytes or control characters
        if "\x00" in v or any(ord(c) < 32 for c in v if c not in "\t\n\r"):
            msg = "Path contains invalid characters"
            raise ValueError(msg)
        return v.strip()

    @field_validator("jwt_algorithm", check_fields=False)
    @classmethod
    def validate_jwt_algorithm(cls: type[Any], v: Any) -> str:
        """Validate JWT algorithm - used in auth configs."""
        allowed_algorithms = {
            "HS256",
            "HS384",
            "HS512",  # HMAC
            "RS256",
            "RS384",
            "RS512",  # RSA
            "ES256",
            "ES384",
            "ES512",  # ECDSA
        }
        if v not in allowed_algorithms:
            msg = f"JWT algorithm must be one of: {allowed_algorithms}"
            raise ValueError(msg)
        return v

    @field_validator("encoding", "charset", check_fields=False)
    @classmethod
    def validate_encoding(cls: type[Any], v: Any) -> str:
        """Validate character encoding - used in Oracle/DB configs."""
        allowed_encodings = {
            "UTF-8",
            "UTF-16",
            "UTF-32",
            "ASCII",
            "ISO-8859-1",
            "ISO-8859-15",
            "CP1252",
            "AL32UTF8",
            "WE8ISO8859P1",  # Oracle specific
        }
        if v.upper() not in allowed_encodings:
            msg = f"Encoding must be one of: {allowed_encodings}"
            raise ValueError(msg)
        return v.upper()


def create_choice_validator(
    field_name: str,
    allowed_choices: set[str],
    *,
    case_sensitive: bool = False,
) -> Callable[[type[Any], Any], str]:
    """Create a validator for choice fields."""

    def validator(cls: type[Any], v: Any) -> str:  # noqa: ARG001
        if not isinstance(v, str):
            msg = f"{field_name} must be a string"
            raise TypeError(msg)

        check_value = v if case_sensitive else v.lower()
        allowed_check = (
            allowed_choices
            if case_sensitive
            else {choice.lower() for choice in allowed_choices}
        )

        if check_value not in allowed_check:
            msg = f"{field_name} must be one of: {allowed_choices}"
            raise ValueError(msg)

        return v if case_sensitive else check_value

    return validator


def create_range_validator(
    field_name: str,
    min_value: float,
    max_value: float,
) -> Callable[[type[Any], Any], int | float]:
    """Create a validator for numeric range fields."""

    def validator(cls: type[Any], v: Any) -> int | float:  # noqa: ARG001
        if not isinstance(v, int | float):
            msg = f"{field_name} must be a number"
            raise TypeError(msg)

        if not min_value <= v <= max_value:
            msg = f"{field_name} must be between {min_value} and {max_value}"
            raise ValueError(
                msg,
            )

        return v

    return validator


# Pre-defined choice validators for common use cases
def validate_http_method(cls: type[Any], v: Any) -> str:
    """Validate HTTP method."""
    return create_choice_validator(
        "HTTP method",
        {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"},
        case_sensitive=True,
    )(cls, v)


def validate_compression_algorithm(cls: type[Any], v: Any) -> str:
    """Validate compression algorithm."""
    return create_choice_validator(
        "Compression algorithm",
        {"gzip", "deflate", "brotli", "lz4", "zstd"},
    )(cls, v)


def validate_hash_algorithm(cls: type[Any], v: Any) -> str:
    """Validate hash algorithm."""
    return create_choice_validator(
        "Hash algorithm",
        {"md5", "sha1", "sha256", "sha512", "blake2b"},
    )(cls, v)


__all__ = [
    "CommonValidators",
    "create_choice_validator",
    "create_range_validator",
    "validate_compression_algorithm",
    "validate_hash_algorithm",
    "validate_http_method",
]
