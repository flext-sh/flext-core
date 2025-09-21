"""Shared constants backing the FLEXT-Core 1.0.0 modernization plan.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class FlextConstants:
    """Essential constants mirroring the modernization plan defaults.

    Each nested namespace lines up with limits, error codes, and defaults
    captured in ``README.md`` and ``docs/architecture.md`` so dependants share
    consistent thresholds during the 1.0.0 rollout.
    """

    class Core:
        """Core identifiers hardened for the 1.0.0 release cycle."""

        NAME: Final[str] = "FLEXT"  # Usage count: 1
        VERSION: Final[str] = "0.9.0"  # Usage count: 8

    class Network:
        """Network defaults shared across dispatcher-aligned services."""

        MIN_PORT: Final[int] = 1  # Usage count: 4
        MAX_PORT: Final[int] = 65535  # Usage count: 4
        TOTAL_TIMEOUT: Final[int] = 60  # Usage count: 0
        DEFAULT_TIMEOUT: Final[int] = 30  # Usage count: 4

    class Validation:
        """Validation guardrails referenced in modernization docs."""

        MIN_NAME_LENGTH: Final[int] = 2  # Usage count: 1
        MAX_NAME_LENGTH: Final[int] = 100  # Usage count: 0
        MIN_SERVICE_NAME_LENGTH: Final[int] = 2  # Usage count: 0
        MAX_EMAIL_LENGTH: Final[int] = 254  # Usage count: 0
        MIN_PERCENTAGE: Final[float] = 0.0  # Usage count: 0
        MAX_PERCENTAGE: Final[float] = 100.0  # Usage count: 0
        MIN_SECRET_KEY_LENGTH: Final[int] = 32  # Usage count: 0

    class Errors:
        """Canonical error codes surfaced in telemetry narratives."""

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
        """User-facing validation and failure messages."""

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
        """Entity validation prompts reused across services."""

        ENTITY_ID_EMPTY: Final[str] = "Entity ID cannot be empty"  # Usage count: 0

    class Defaults:
        """Baseline defaults called out in onboarding docs."""

        TIMEOUT: Final[int] = 30  # Usage count: 18
        PAGE_SIZE: Final[int] = 100  # Usage count: 2

    class Limits:
        """Upper bounds safeguarding payload and resource usage."""

        MAX_STRING_LENGTH: Final[int] = 1000  # Usage count: 0
        MAX_LIST_SIZE: Final[int] = 10000  # Usage count: 0
        MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB  # Usage count: 0

    class Utilities:
        """Utility constants reused by helper modules."""

        SECONDS_PER_MINUTE: Final[int] = 60  # Usage count: 0
        SECONDS_PER_HOUR: Final[int] = 3600  # Usage count: 0
        BYTES_PER_KB: Final[int] = 1024  # Usage count: 0

    class Patterns:
        """Regex placeholders for downstream ecosystem adapters."""  # Usage count: 0

    class Config:
        """Configuration defaults anchoring the unified lifecycle."""

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
            """Enumerate configuration origins supported by FlextConfig."""

            FILE = "file"  # Usage count: 0
            ENVIRONMENT = "env"  # Usage count: 0
            CLI = "cli"  # Usage count: 0
            DEFAULT = "default"  # Usage count: 0
            DOTENV = "dotenv"  # Usage count: 0

        class LogLevel(StrEnum):
            """Standard log levels mirroring FlextLogger semantics."""

            DEBUG = "DEBUG"  # Usage count: 4
            INFO = "INFO"  # Usage count: 4
            WARNING = "WARNING"  # Usage count: 4
            ERROR = "ERROR"  # Usage count: 3
            CRITICAL = "CRITICAL"  # Usage count: 3

    class Enums:
        """Shared enumerations referenced across the API surface."""

        class FieldType(StrEnum):
            """Normalized field types for configuration and model metadata."""

            STRING = "string"  # Usage count: 0
            INTEGER = "integer"  # Usage count: 0
            FLOAT = "float"  # Usage count: 0
            BOOLEAN = "boolean"  # Usage count: 0
            DATE = "date"  # Usage count: 0
            DATETIME = "datetime"  # Usage count: 0
            UUID = "uuid"  # Usage count: 0
            EMAIL = "email"  # Usage count: 0

    class Platform:
        """Platform defaults referenced by CLI and adapter packages."""

        FLEXT_API_PORT: Final[int] = 8000  # Usage count: 4
        DEFAULT_HOST: Final[str] = "localhost"  # Usage count: 0

    class Observability:
        """Observability defaults consumed by FlextLogger."""

        DEFAULT_LOG_LEVEL: Final[str] = "INFO"  # Usage count: 0

    class Performance:
        """Performance tuning knobs surfaced in roadmap metrics."""

        DEFAULT_BATCH_SIZE: Final[int] = 1000  # Usage count: 2

    class Reliability:
        """Reliability thresholds backing retry guidance."""

        MAX_RETRY_ATTEMPTS: Final[int] = 3  # Usage count: 1
        DEFAULT_MAX_RETRIES: Final[int] = 3  # Usage count: 1 (referenced in models.py)
        DEFAULT_BACKOFF_STRATEGY: Final[str] = (
            "exponential"  # Usage count: 1 (referenced in models.py)
        )
        DEFAULT_FAILURE_THRESHOLD: Final[int] = 5  # Circuit breaker failure threshold
        DEFAULT_RECOVERY_TIMEOUT: Final[int] = (
            60  # Circuit breaker recovery timeout in seconds
        )
        DEFAULT_TIMEOUT_SECONDS: Final[float] = (
            30.0  # Default timeout for domain services
        )

    class Environment:
        """Environment enumerations used by configuration profiles."""

        class ConfigEnvironment(StrEnum):
            """Enumerate core deployment environments in docs."""

            DEVELOPMENT = "development"  # Usage count: 1
            STAGING = "staging"  # Usage count: 0
            PRODUCTION = "production"  # Usage count: 1
            TESTING = "testing"  # Usage count: 0

        class ValidationLevel(StrEnum):
            """Validation strictness tiers adopted by tooling."""

            STRICT = "strict"  # Usage count: 0
            NORMAL = "normal"  # Usage count: 0
            RELAXED = "relaxed"  # Usage count: 0

    class Cqrs:
        """CQRS (Command Query Responsibility Segregation) constants."""

        # Handler types
        DEFAULT_HANDLER_TYPE: Final[str] = "command"
        COMMAND_HANDLER_TYPE: Final[str] = "command"
        QUERY_HANDLER_TYPE: Final[str] = "query"

        # Timeout constants
        DEFAULT_TIMEOUT: Final[int] = 30000  # milliseconds
        MIN_TIMEOUT: Final[int] = 1000  # milliseconds
        MAX_TIMEOUT: Final[int] = 300000  # milliseconds (5 minutes)

        # Retry constants
        DEFAULT_RETRIES: Final[int] = 0
        MIN_RETRIES: Final[int] = 0
        MAX_RETRIES: Final[int] = 5

        # Pagination constants
        DEFAULT_PAGE_SIZE: Final[int] = 10
        MAX_PAGE_SIZE: Final[int] = 1000

        # Error constants for CQRS operations
        CQRS_OPERATION_FAILED: Final[str] = "CQRS_OPERATION_FAILED"
        COMMAND_VALIDATION_FAILED: Final[str] = "COMMAND_VALIDATION_FAILED"
        QUERY_VALIDATION_FAILED: Final[str] = "QUERY_VALIDATION_FAILED"
        HANDLER_CONFIG_INVALID: Final[str] = "HANDLER_CONFIG_INVALID"
        COMMAND_PROCESSING_FAILED: Final[str] = "COMMAND_PROCESSING_FAILED"

    class Container:
        """Container configuration constants for FlextContainer."""

        # Worker configuration
        MAX_WORKERS: Final[int] = 4
        MIN_WORKERS: Final[int] = 1
        DEFAULT_WORKERS: Final[int] = 2

        # Timeout configuration
        TIMEOUT_SECONDS: Final[float] = 30.0
        MIN_TIMEOUT_SECONDS: Final[float] = 0.1
        MAX_TIMEOUT_SECONDS: Final[float] = 600.0

        # Registration limits
        MAX_SERVICES: Final[int] = 1000
        MAX_FACTORIES: Final[int] = 500

    class Logging:
        """Logging configuration constants."""

        # Log levels
        DEFAULT_LEVEL: Final[str] = "INFO"
        VALID_LEVELS: Final[set[str]] = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        # Log formatting
        DEFAULT_FORMAT: Final[str] = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

        # FlextLogger optimization constants for Pydantic models
        INCLUDE_SOURCE: Final[bool] = True
        STRUCTURED_OUTPUT: Final[bool] = True
        VERBOSITY: Final[str] = "detailed"

        # Performance tracking constants
        TRACK_PERFORMANCE: Final[bool] = True
        TRACK_MEMORY: Final[bool] = False
        TRACK_TIMING: Final[bool] = True
        PERFORMANCE_THRESHOLD_WARNING: Final[float] = 1000.0  # milliseconds
        PERFORMANCE_THRESHOLD_CRITICAL: Final[float] = 5000.0  # milliseconds

    class Dispatcher:
        """Constants for FlextDispatcher operations.

        USAGE: Centralized constants for dispatcher modes, defaults, and validation.
        OPTIMIZATION: Eliminates magic strings and provides type-safe constants.
        """

        # Handler Modes
        HANDLER_MODE_COMMAND = "command"
        HANDLER_MODE_QUERY = "query"
        VALID_HANDLER_MODES = (HANDLER_MODE_COMMAND, HANDLER_MODE_QUERY)
        DEFAULT_HANDLER_MODE = HANDLER_MODE_COMMAND

        # Default Settings
        DEFAULT_AUTO_CONTEXT = True
        DEFAULT_ENABLE_LOGGING = True
        DEFAULT_ENABLE_METRICS = True
        DEFAULT_TIMEOUT_SECONDS = 30

        # Validation Limits
        MIN_TIMEOUT_SECONDS = 1
        MAX_TIMEOUT_SECONDS = 600
        MIN_REGISTRATION_ID_LENGTH = 1
        MIN_REQUEST_ID_LENGTH = 1

        # Error Messages
        ERROR_INVALID_HANDLER_MODE = "handler_mode must be 'command' or 'query'"
        ERROR_HANDLER_REQUIRED = "handler cannot be None"
        ERROR_MESSAGE_REQUIRED = "message cannot be None"
        ERROR_POSITIVE_TIMEOUT = "timeout must be positive"
        ERROR_INVALID_REGISTRATION_ID = "registration_id must be non-empty string"
        ERROR_INVALID_REQUEST_ID = "request_id must be non-empty string"

        # Registration Status
        REGISTRATION_STATUS_ACTIVE = "active"
        REGISTRATION_STATUS_INACTIVE = "inactive"
        REGISTRATION_STATUS_ERROR = "error"
        VALID_REGISTRATION_STATUSES = (
            REGISTRATION_STATUS_ACTIVE,
            REGISTRATION_STATUS_INACTIVE,
            REGISTRATION_STATUS_ERROR,
        )

    class Mixins:
        """Constants for FlextMixins operations.

        USAGE: Centralized constants for mixin field names, states, and defaults.
        OPTIMIZATION: Eliminates magic strings and provides type-safe constants.
        """

        # Field Names
        FIELD_ID = "id"
        FIELD_STATE = "state"
        FIELD_CREATED_AT = "created_at"
        FIELD_UPDATED_AT = "updated_at"
        FIELD_VALIDATED = "validated"

        # Default States
        STATE_ACTIVE = "active"
        STATE_INACTIVE = "inactive"
        STATE_PENDING = "pending"
        STATE_COMPLETED = "completed"
        STATE_FAILED = "failed"

        # Validation Types (simple string constants)
        VALIDATION_BASIC = "basic"
        VALIDATION_STRICT = "strict"
        VALIDATION_CUSTOM = "custom"

        # Log Levels (simple string constants)
        LOG_LEVEL_DEBUG = "DEBUG"
        LOG_LEVEL_INFO = "INFO"
        LOG_LEVEL_WARNING = "WARNING"
        LOG_LEVEL_ERROR = "ERROR"
        LOG_LEVEL_CRITICAL = "CRITICAL"

        # Serialization Defaults
        DEFAULT_JSON_INDENT = 2
        DEFAULT_ENCODING = "utf-8"
        DEFAULT_SORT_KEYS = False
        DEFAULT_ENSURE_ASCII = False

        # Timestamp Defaults
        DEFAULT_USE_UTC = True
        DEFAULT_AUTO_UPDATE = True

        # Validation Limits
        MAX_OPERATION_NAME_LENGTH = 100
        MAX_STATE_VALUE_LENGTH = 50
        MAX_FIELD_NAME_LENGTH = 50
        MIN_FIELD_NAME_LENGTH = 1

        # Error Messages
        ERROR_EMPTY_OPERATION = "Operation name cannot be empty"
        ERROR_EMPTY_STATE = "State value cannot be empty"
        ERROR_EMPTY_FIELD_NAME = "Field name cannot be empty"
        ERROR_INVALID_ENCODING = "Invalid character encoding"
        ERROR_MISSING_TIMESTAMP_FIELDS = "Required timestamp fields missing"
        ERROR_INVALID_LOG_LEVEL = "Invalid log level"


__all__ = ["FlextConstants"]
