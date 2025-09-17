"""FLEXT Core Constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class FlextConstants:
    """Essential constants for FLEXT core functionality."""

    class Core:
        """Core system constants."""

        NAME: Final[str] = "FLEXT"  # Usage count: 1
        VERSION: Final[str] = "0.9.0"  # Usage count: 8

    class Network:
        """Network constants."""

        MIN_PORT: Final[int] = 1  # Usage count: 4
        MAX_PORT: Final[int] = 65535  # Usage count: 4
        TOTAL_TIMEOUT: Final[int] = 60  # Usage count: 0
        DEFAULT_TIMEOUT: Final[int] = 30  # Usage count: 4

    class Validation:
        """Validation constants."""

        MIN_NAME_LENGTH: Final[int] = 2  # Usage count: 1
        MAX_NAME_LENGTH: Final[int] = 100  # Usage count: 0
        MIN_SERVICE_NAME_LENGTH: Final[int] = 2  # Usage count: 0
        MAX_EMAIL_LENGTH: Final[int] = 254  # Usage count: 0
        MIN_PERCENTAGE: Final[float] = 0.0  # Usage count: 0
        MAX_PERCENTAGE: Final[float] = 100.0  # Usage count: 0
        MIN_SECRET_KEY_LENGTH: Final[int] = 32  # Usage count: 0

    class Errors:
        """Error codes."""

        VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"  # Usage count: 28
        TYPE_ERROR: Final[str] = "TYPE_ERROR"  # Usage count: 4
        SERIALIZATION_ERROR: Final[str] = "SERIALIZATION_ERROR"  # Usage count: 0
        CONFIG_ERROR: Final[str] = "CONFIG_ERROR"  # Usage count: 1
        OPERATION_ERROR: Final[str] = "OPERATION_ERROR"  # Usage count: 0
        BUSINESS_RULE_VIOLATION: Final[str] = (
            "BUSINESS_RULE_VIOLATION"  # Usage count: 0
        )
        NOT_FOUND_ERROR: Final[str] = "NOT_FOUND_ERROR"  # Usage count: 0
        NOT_FOUND: Final[str] = "NOT_FOUND"  # Usage count: 0
        GENERIC_ERROR: Final[str] = "GENERIC_ERROR"  # Usage count: 3
        COMMAND_PROCESSING_FAILED: Final[str] = (
            "COMMAND_PROCESSING_FAILED"  # Usage count: 4
        )
        COMMAND_BUS_ERROR: Final[str] = "COMMAND_BUS_ERROR"  # Usage count: 0
        TIMEOUT_ERROR: Final[str] = "TIMEOUT_ERROR"  # Usage count: 0
        PROCESSING_ERROR: Final[str] = "PROCESSING_ERROR"  # Usage count: 0
        PERMISSION_ERROR: Final[str] = "PERMISSION_ERROR"  # Usage count: 0
        EXCEPTION_ERROR: Final[str] = "EXCEPTION_ERROR"  # Usage count: 0
        CRITICAL_ERROR: Final[str] = "CRITICAL_ERROR"  # Usage count: 0
        CONNECTION_ERROR: Final[str] = "CONNECTION_ERROR"  # Usage count: 0
        CONFIGURATION_ERROR: Final[str] = "CONFIGURATION_ERROR"  # Usage count: 0
        COMMAND_HANDLER_NOT_FOUND: Final[str] = (
            "COMMAND_HANDLER_NOT_FOUND"  # Usage count: 0
        )
        AUTHENTICATION_ERROR: Final[str] = "AUTHENTICATION_ERROR"  # Usage count: 0
        ALREADY_EXISTS: Final[str] = "ALREADY_EXISTS"  # Usage count: 0
        RESOURCE_NOT_FOUND: Final[str] = "RESOURCE_NOT_FOUND"  # Usage count: 0
        MAP_ERROR: Final[str] = "MAP_ERROR"  # Usage count: 0
        EXTERNAL_SERVICE_ERROR: Final[str] = "EXTERNAL_SERVICE_ERROR"  # Usage count: 0
        CHAIN_ERROR: Final[str] = "CHAIN_ERROR"  # Usage count: 0
        BUSINESS_RULE_ERROR: Final[str] = "BUSINESS_RULE_ERROR"  # Usage count: 0
        BIND_ERROR: Final[str] = "BIND_ERROR"  # Usage count: 0
        UNWRAP_ERROR: Final[str] = "UNWRAP_ERROR"  # Usage count: 0
        UNKNOWN_ERROR: Final[str] = "UNKNOWN_ERROR"  # Usage count: 1

    class Messages:
        """Message constants."""

        TYPE_MISMATCH: Final[str] = "Type mismatch"  # Usage count: 2
        SERVICE_NAME_EMPTY: Final[str] = (
            "Service name cannot be empty"  # Usage count: 0
        )
        OPERATION_FAILED: Final[str] = "Operation failed"  # Usage count: 0
        INVALID_INPUT: Final[str] = "Invalid input"  # Usage count: 0
        VALUE_EMPTY: Final[str] = "Value cannot be empty"  # Usage count: 0
        VALIDATION_FAILED: Final[str] = "Validation failed"  # Usage count: 0
        UNKNOWN_ERROR: Final[str] = "Unknown error"  # Usage count: 0
        NULL_DATA: Final[str] = "Data cannot be null"  # Usage count: 0

    class Entities:
        """Entity constants."""

        ENTITY_ID_EMPTY: Final[str] = "Entity ID cannot be empty"  # Usage count: 0

    class Defaults:
        """Default values."""

        TIMEOUT: Final[int] = 30  # Usage count: 18
        PAGE_SIZE: Final[int] = 100  # Usage count: 2

    class Limits:
        """Limits."""

        MAX_STRING_LENGTH: Final[int] = 1000  # Usage count: 0
        MAX_LIST_SIZE: Final[int] = 10000  # Usage count: 0
        MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB  # Usage count: 0

    class Utilities:
        """Utility constants."""

        SECONDS_PER_MINUTE: Final[int] = 60  # Usage count: 0
        SECONDS_PER_HOUR: Final[int] = 3600  # Usage count: 0
        BYTES_PER_KB: Final[int] = 1024  # Usage count: 0

    class Patterns:
        """Regex patterns."""  # Usage count: 0

    class Config:
        """Config constants."""

        ENVIRONMENTS: Final[list[str]] = [  # Usage count: 3
            "development",
            "staging",
            "production",
            "test",
            "local",
        ]
        DEFAULT_ENVIRONMENT: Final[str] = "development"  # Usage count: 0
        DOTENV_FILES: Final[list[str]] = [
            ".env",
            ".env.local",
            ".env.production",
        ]  # Usage count: 0

        # Configuration validation constants (moved from FlextConfig.Constants)
        SEMANTIC_VERSION_MIN_PARTS: Final[int] = 3
        MIN_PRODUCTION_WORKERS: Final[int] = 2
        HIGH_TIMEOUT_THRESHOLD: Final[int] = 120
        MIN_WORKERS_FOR_HIGH_TIMEOUT: Final[int] = 4
        MAX_WORKERS_THRESHOLD: Final[int] = 50

        # Configuration profile constants
        PROFILE_WEB_SERVICE: Final[str] = "web_service"
        PROFILE_DATA_PROCESSOR: Final[str] = "data_processor"
        PROFILE_API_CLIENT: Final[str] = "api_client"
        PROFILE_BATCH_JOB: Final[str] = "batch_job"
        PROFILE_MICROSERVICE: Final[str] = "microservice"

        class ConfigSource(StrEnum):
            """Config source enumeration."""

            FILE = "file"  # Usage count: 0
            ENVIRONMENT = "env"  # Usage count: 0
            CLI = "cli"  # Usage count: 0
            DEFAULT = "default"  # Usage count: 0
            DOTENV = "dotenv"  # Usage count: 0

        class LogLevel(StrEnum):
            """Log level enumeration."""

            DEBUG = "DEBUG"  # Usage count: 4
            INFO = "INFO"  # Usage count: 4
            WARNING = "WARNING"  # Usage count: 4
            ERROR = "ERROR"  # Usage count: 3
            CRITICAL = "CRITICAL"  # Usage count: 3

    class Enums:
        """Enumerations."""

        class FieldType(StrEnum):
            """Field type enumeration."""

            STRING = "string"  # Usage count: 0
            INTEGER = "integer"  # Usage count: 0
            FLOAT = "float"  # Usage count: 0
            BOOLEAN = "boolean"  # Usage count: 0
            DATE = "date"  # Usage count: 0
            DATETIME = "datetime"  # Usage count: 0
            UUID = "uuid"  # Usage count: 0
            EMAIL = "email"  # Usage count: 0

    class Platform:
        """Platform constants."""

        FLEXT_API_PORT: Final[int] = 8000  # Usage count: 4
        DEFAULT_HOST: Final[str] = "localhost"  # Usage count: 0

    class Observability:
        """Observability constants."""

        DEFAULT_LOG_LEVEL: Final[str] = "INFO"  # Usage count: 0

    class Performance:
        """Performance constants."""

        DEFAULT_BATCH_SIZE: Final[int] = 1000  # Usage count: 2

    class Reliability:
        """Reliability constants."""

        MAX_RETRY_ATTEMPTS: Final[int] = 3  # Usage count: 1

    class Environment:
        """Environment constants."""

        class ConfigEnvironment(StrEnum):
            """Configuration environment types."""

            DEVELOPMENT = "development"  # Usage count: 1
            STAGING = "staging"  # Usage count: 0
            PRODUCTION = "production"  # Usage count: 1
            TESTING = "testing"  # Usage count: 0

        class ValidationLevel(StrEnum):
            """Validation level types."""

            STRICT = "strict"  # Usage count: 0
            NORMAL = "normal"  # Usage count: 0
            RELAXED = "relaxed"  # Usage count: 0


__all__ = ["FlextConstants"]
