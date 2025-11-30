"""Immutable, shared constants for the FLEXT ecosystem.

Centralize configuration defaults, validation limits, error codes, and runtime
enums in a pure Layer 0 module so dispatcher handlers, services, and utilities
share the same source of truth without circular imports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, TypedDict, TypeIs, cast

from pydantic import ConfigDict

from flext_core.typings import FlextTypes


class DockerContainerConfig(TypedDict):
    """Type definition for Docker container configuration."""

    compose_file: str
    service: str
    port: int


class FlextConstants:
    """Namespace-organized constants for configuration and validation.

    Architecture: Pure constants (no Layer 1+ imports)
    Foundation for the FLEXT ecosystem, exposing immutable values grouped by
    purpose—core identifiers, validation limits, error codes, defaults, and
    enums—so dispatcher pipelines, services, and utilities share consistent
    defaults. Structural typing keeps the class compatible with
    ``FlextProtocols.Constants`` without inheritance.
    """

    # Core identifiers
    NAME: Final[str] = "FLEXT"
    ZERO: Final[int] = 0
    INITIAL_TIME: Final[float] = 0.0

    def __getitem__(self, key: str) -> FlextTypes.ConstantValue:
        """Dynamic access: FlextConstants['Errors.VALIDATION_ERROR'].

        Args:
            key: Dot-separated path (e.g., 'Errors.VALIDATION_ERROR')

        Returns:
            The constant value (str, int, float, bool, ConfigDict, frozenset,
            tuple, Mapping, or StrEnum)

        Raises:
            AttributeError: If path not found

        """
        parts = key.split(".")
        result: object = self
        try:
            for part in parts:
                attr = getattr(result, part)
                result = attr
            return cast("FlextTypes.ConstantValue", result)
        except AttributeError as e:
            msg = f"Constant path '{key}' not found"
            raise AttributeError(msg) from e

    class Network:
        """Network-related defaults."""

        MIN_PORT: Final[int] = 1
        MAX_PORT: Final[int] = 65535
        DEFAULT_TIMEOUT: Final[int] = 30
        DEFAULT_CONNECTION_POOL_SIZE: Final[int] = 10
        MAX_CONNECTION_POOL_SIZE: Final[int] = 100
        MAX_HOSTNAME_LENGTH: Final[int] = 253

    class Validation:
        """Input validation limits."""

        MIN_NAME_LENGTH: Final[int] = 2
        MAX_NAME_LENGTH: Final[int] = 100
        MAX_EMAIL_LENGTH: Final[int] = 254
        EMAIL_PARTS_COUNT: Final[int] = 2
        LEVEL_PREFIX_PARTS_COUNT: Final[int] = 4
        MIN_PHONE_DIGITS: Final[int] = 10
        MAX_PHONE_DIGITS: Final[int] = 20
        MIN_USERNAME_LENGTH: Final[int] = 3
        MAX_AGE: Final[int] = 150
        MIN_AGE: Final[int] = 0
        PREVIEW_LENGTH: Final[int] = 50
        VALIDATION_TIMEOUT_MS: Final[int] = 100
        MAX_UNCOMMITTED_EVENTS: Final[int] = 100
        DISCOUNT_THRESHOLD: Final[int] = 100
        DISCOUNT_RATE: Final[float] = 0.05
        SLOW_OPERATION_THRESHOLD: Final[float] = 0.1
        RESOURCE_LIMIT_MIN: Final[int] = 50
        FILTER_THRESHOLD: Final[int] = 5
        RETRY_COUNT_MAX: Final[int] = 3
        MAX_WORKERS_LIMIT: Final[int] = 100

    class Errors:
        """Error codes for categorization."""

        # Core
        VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"
        TYPE_ERROR: Final[str] = "TYPE_ERROR"
        ATTRIBUTE_ERROR: Final[str] = "ATTRIBUTE_ERROR"
        CONFIG_ERROR: Final[str] = "CONFIG_ERROR"
        GENERIC_ERROR: Final[str] = "GENERIC_ERROR"
        COMMAND_PROCESSING_FAILED: Final[str] = "COMMAND_PROCESSING_FAILED"
        UNKNOWN_ERROR: Final[str] = "UNKNOWN_ERROR"

        # Applicative
        FIRST_ARG_FAILED_MSG: Final[str] = "First argument failed"
        SECOND_ARG_FAILED_MSG: Final[str] = "Second argument failed"

        # Serialization
        SERIALIZATION_ERROR: Final[str] = "SERIALIZATION_ERROR"
        MAP_ERROR: Final[str] = "MAP_ERROR"
        BIND_ERROR: Final[str] = "BIND_ERROR"
        CHAIN_ERROR: Final[str] = "CHAIN_ERROR"
        UNWRAP_ERROR: Final[str] = "UNWRAP_ERROR"

        # Business
        OPERATION_ERROR: Final[str] = "OPERATION_ERROR"
        SERVICE_ERROR: Final[str] = "SERVICE_ERROR"
        BUSINESS_RULE_VIOLATION: Final[str] = "BUSINESS_RULE_VIOLATION"
        BUSINESS_RULE_ERROR: Final[str] = "BUSINESS_RULE_ERROR"

        # Resource
        NOT_FOUND_ERROR: Final[str] = "NOT_FOUND_ERROR"
        NOT_FOUND: Final[str] = "NOT_FOUND"
        RESOURCE_NOT_FOUND: Final[str] = "RESOURCE_NOT_FOUND"
        ALREADY_EXISTS: Final[str] = "ALREADY_EXISTS"

        # CQRS
        COMMAND_BUS_ERROR: Final[str] = "COMMAND_BUS_ERROR"
        COMMAND_HANDLER_NOT_FOUND: Final[str] = "COMMAND_HANDLER_NOT_FOUND"
        DOMAIN_EVENT_ERROR: Final[str] = "DOMAIN_EVENT_ERROR"

        # Infrastructure
        TIMEOUT_ERROR: Final[str] = "TIMEOUT_ERROR"
        PROCESSING_ERROR: Final[str] = "PROCESSING_ERROR"
        CONNECTION_ERROR: Final[str] = "CONNECTION_ERROR"
        CONFIGURATION_ERROR: Final[str] = "CONFIGURATION_ERROR"
        EXTERNAL_SERVICE_ERROR: Final[str] = "EXTERNAL_SERVICE_ERROR"

        # Security
        PERMISSION_ERROR: Final[str] = "PERMISSION_ERROR"
        AUTHENTICATION_ERROR: Final[str] = "AUTHENTICATION_ERROR"
        AUTHORIZATION_ERROR: Final[str] = "AUTHORIZATION_ERROR"

        # System
        EXCEPTION_ERROR: Final[str] = "EXCEPTION_ERROR"
        CRITICAL_ERROR: Final[str] = "CRITICAL_ERROR"
        NONEXISTENT_ERROR: Final[str] = "NONEXISTENT_ERROR"

    class Test:
        """Test constants."""

        DEFAULT_PASSWORD: Final[str] = "password123"
        DEFAULT_USERNAME: Final[str] = "testuser"
        NONEXISTENT_USERNAME: Final[str] = "nonexistent"

        class Docker:
            """Docker test infrastructure constants."""

            DEFAULT_LOG_TAIL: Final[int] = 100
            DEFAULT_CONTAINER_CHOICES: Final[tuple[str, ...]] = (
                "flext-shared-ldap",
                "flext-postgres",
                "flext-redis",
                "flext-oracle",
            )
            SHARED_CONTAINERS: Final[Mapping[str, DockerContainerConfig]] = {
                "flext-openldap-test": {
                    "compose_file": "flext-ldap/docker/docker-compose.yml",
                    "service": "openldap",
                    "port": 3390,
                },
                "flext-postgres-test": {
                    "compose_file": "flext-db-postgres/docker/docker-compose.yml",
                    "service": "postgres",
                    "port": 5433,
                },
                "flext-redis-test": {
                    "compose_file": "flext-redis/docker/docker-compose.yml",
                    "service": "redis",
                    "port": 6380,
                },
                "flext-oracle-db-test": {
                    "compose_file": "flext-db-oracle/docker/docker-compose.yml",
                    "service": "oracle-db",
                    "port": 1522,
                },
            }

    class Exceptions:
        """Exception handling configuration."""

        class FailureLevel(StrEnum):
            """Exception failure levels."""

            STRICT = "strict"
            WARN = "warn"
            PERMISSIVE = "permissive"

        FAILURE_LEVEL_DEFAULT: Final[FailureLevel] = FailureLevel.PERMISSIVE

    class Messages:
        """User-facing message templates."""

        TYPE_MISMATCH: Final[str] = "Type mismatch"
        VALIDATION_FAILED: Final[str] = "Validation failed"
        REDACTED_SECRET: Final[str] = "***REDACTED***"

    class Defaults:
        """Default values."""

        TIMEOUT: Final[int] = 30
        PAGE_SIZE: Final[int] = 100
        TIMEOUT_SECONDS: Final[int] = 30
        CACHE_TTL: Final[int] = 300
        DEFAULT_CACHE_TTL: Final[int] = CACHE_TTL
        MAX_CACHE_SIZE: Final[int] = 100
        DEFAULT_MAX_CACHE_SIZE: Final[int] = MAX_CACHE_SIZE
        MAX_MESSAGE_LENGTH: Final[int] = 100
        DEFAULT_MIDDLEWARE_ORDER: Final[int] = 0
        OPERATION_TIMEOUT_SECONDS: Final[int] = 30

    class Utilities:
        """Utility constants."""

        DEFAULT_ENCODING: Final[str] = "utf-8"
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        MAX_TIMEOUT_SECONDS: Final[int] = 3600
        LONG_UUID_LENGTH: Final[int] = 12
        SHORT_UUID_LENGTH: Final[int] = 8
        VERSION_MODULO: Final[int] = 100
        CONTROL_CHARS_PATTERN: Final[str] = r"[\x00-\x1F\x7F]"
        CACHE_ATTRIBUTE_NAMES: Final[tuple[str, ...]] = (
            "_cache",
            "_ttl",
            "_cached_at",
            "_cached_value",
        )

    class Settings:
        """Configuration defaults."""

        MAX_WORKERS_THRESHOLD: Final[int] = 50
        DEFAULT_ENABLE_CACHING: Final[bool] = True
        DEFAULT_ENABLE_METRICS: Final[bool] = False
        DEFAULT_ENABLE_TRACING: Final[bool] = False
        DEFAULT_TIMEOUT: Final[int] = 30
        DEFAULT_DEBUG_MODE: Final[bool] = False
        DEFAULT_TRACE_MODE: Final[bool] = False

        class LogLevel(StrEnum):
            """Standard log levels."""

            DEBUG = "DEBUG"
            INFO = "INFO"
            WARNING = "WARNING"
            ERROR = "ERROR"
            CRITICAL = "CRITICAL"

        class Environment(StrEnum):
            """Environment types."""

            DEVELOPMENT = "development"
            STAGING = "staging"
            PRODUCTION = "production"
            TESTING = "testing"
            LOCAL = "local"

    class ModelConfig:
        """Pydantic model configuration defaults."""

        BASE: Final[ConfigDict] = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            hide_input_in_errors=True,
        )

    class Platform:
        """Platform-specific constants."""

        ENV_PREFIX: Final[str] = "FLEXT_"
        ENV_FILE_DEFAULT: Final[str] = ".env"
        ENV_FILE_ENV_VAR: Final[str] = (
            "FLEXT_ENV_FILE"  # Env var to customize .env path
        )
        ENV_NESTED_DELIMITER: Final[str] = "__"
        DEFAULT_APP_NAME: Final[str] = "flext"
        FLEXT_API_PORT: Final[int] = 8000
        DEFAULT_HOST: Final[str] = "localhost"
        DEFAULT_HTTP_PORT: Final[int] = 80
        MIME_TYPE_JSON: Final[str] = "application/json"
        PATTERN_EMAIL: Final[str] = (
            r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
        )
        PATTERN_URL: Final[str] = (
            r"^https?://(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?"
        )
        PATTERN_PHONE_NUMBER: Final[str] = r"^\+?[\d\s\-\(\)]{10,20}$"
        PATTERN_UUID: Final[str] = (
            r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
        )
        PATTERN_PATH: Final[str] = r'^[^<>"|?*\x00-\x1F]+$'
        EXT_PYTHON: Final[str] = ".py"
        EXT_YAML: Final[str] = ".yaml"
        EXT_JSON: Final[str] = ".json"
        EXT_TOML: Final[str] = ".toml"
        EXT_XML: Final[str] = ".xml"
        EXT_TXT: Final[str] = ".txt"
        EXT_MD: Final[str] = ".md"
        DIR_CONFIG: Final[str] = "config"
        DIR_PLUGINS: Final[str] = "plugins"
        DIR_LOGS: Final[str] = "logs"
        DIR_DATA: Final[str] = "data"
        DIR_TEMP: Final[str] = "temp"

    class Performance:
        """Performance thresholds."""

        DEFAULT_DB_POOL_SIZE: Final[int] = 10
        MIN_DB_POOL_SIZE: Final[int] = 1
        MAX_DB_POOL_SIZE: Final[int] = 100
        MAX_RETRY_ATTEMPTS_LIMIT: Final[int] = 10
        DEFAULT_TIMEOUT_LIMIT: Final[int] = 300
        MIN_CURRENT_STEP: Final[int] = 0
        DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
        DEFAULT_MAX_DELAY_SECONDS: Final[float] = 60.0
        DEFAULT_INITIAL_DELAY_SECONDS: Final[float] = 1.0
        MAX_BATCH_SIZE: Final[int] = 10000
        DEFAULT_TIME_RANGE_SECONDS: Final[int] = 3600
        DEFAULT_TTL_SECONDS: Final[int] = 3600
        DEFAULT_VERSION: Final[int] = 1
        MIN_VERSION: Final[int] = 1
        DEFAULT_PAGE_SIZE: Final[int] = 10
        HIGH_MEMORY_THRESHOLD_BYTES: Final[int] = 1073741824
        MAX_TIMEOUT_SECONDS: Final[int] = 600
        MAX_BATCH_OPERATIONS: Final[int] = 1000
        MAX_OPERATION_NAME_LENGTH: Final[int] = 100
        EXPECTED_TUPLE_LENGTH: Final[int] = 2
        DEFAULT_EMPTY_STRING: Final[str] = ""

        class BatchProcessing:
            """Batch processing constants."""

            DEFAULT_SIZE: Final[int] = 1000
            MAX_ITEMS: Final[int] = 10000
            MAX_VALIDATION_SIZE: Final[int] = 1000

        CLI_PERFORMANCE_CRITICAL_MS: Final[float] = 10000.0

    class Reliability:
        """Reliability thresholds."""

        MAX_RETRY_ATTEMPTS: Final[int] = 3
        DEFAULT_MAX_RETRIES: Final[int] = 3
        DEFAULT_RETRY_DELAY_SECONDS: Final[int] = 1
        RETRY_BACKOFF_BASE: Final[float] = 2.0
        RETRY_BACKOFF_MAX: Final[float] = 60.0
        RETRY_COUNT_MIN: Final[int] = 1
        DEFAULT_BACKOFF_STRATEGY: Final[str] = "exponential"
        DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
        DEFAULT_RECOVERY_TIMEOUT: Final[int] = 60
        DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
        DEFAULT_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
        DEFAULT_RATE_LIMIT_MAX_REQUESTS: Final[int] = 100
        DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
        DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[int] = 60
        DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD: Final[int] = 3

        class CircuitBreakerState(StrEnum):
            """Circuit breaker states."""

            CLOSED = "closed"
            OPEN = "open"
            HALF_OPEN = "half_open"

            @classmethod
            def validate(cls, value: str) -> bool:
                """Validate state value.

                Args:
                    value: State value to validate.

                Returns:
                    True if value is a valid state, False otherwise.

                """
                return value in cls.__members__.values()

    class Security:
        """Security constants."""

        JWT_DEFAULT_ALGORITHM: Final[str] = "HS256"
        CREDENTIAL_BCRYPT_ROUNDS: Final[int] = 12

    class Logging:
        """Logging configuration."""

        DEBUG: Final[str] = "DEBUG"
        INFO: Final[str] = "INFO"
        WARNING: Final[str] = "WARNING"
        ERROR: Final[str] = "ERROR"
        CRITICAL: Final[str] = "CRITICAL"
        DEFAULT_LEVEL: Final[str] = "INFO"
        DEFAULT_LEVEL_DEVELOPMENT: Final[str] = "DEBUG"
        DEFAULT_LEVEL_PRODUCTION: Final[str] = "WARNING"
        DEFAULT_LEVEL_TESTING: Final[str] = "INFO"
        VALID_LEVELS: Final[tuple[str, ...]] = (
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        )
        # Immutable set for O(1) membership testing (Python 3.13+ frozenset)
        VALID_LEVELS_SET: Final[frozenset[str]] = frozenset(VALID_LEVELS)
        JSON_OUTPUT_DEFAULT: Final[bool] = False
        DEFAULT_FORMAT: Final[str] = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        STRUCTURED_OUTPUT: Final[bool] = True
        INCLUDE_SOURCE: Final[bool] = True
        VERBOSITY: Final[str] = "compact"
        MAX_FILE_SIZE: Final[int] = 10485760
        BACKUP_COUNT: Final[int] = 5
        CONSOLE_ENABLED: Final[bool] = True
        CONSOLE_COLOR_ENABLED: Final[bool] = True
        TRACK_PERFORMANCE: Final[bool] = False
        TRACK_TIMING: Final[bool] = False
        INCLUDE_CONTEXT: Final[bool] = True
        INCLUDE_CORRELATION_ID: Final[bool] = True
        MAX_CONTEXT_KEYS: Final[int] = 50
        MASK_SENSITIVE_DATA: Final[bool] = True

        # String constants for context operations (runtime use)
        # Using Literal types directly in Final for Python 3.13+ type safety
        CONTEXT_OPERATION_BIND: Final = "bind"
        CONTEXT_OPERATION_UNBIND: Final = "unbind"
        CONTEXT_OPERATION_CLEAR: Final = "clear"
        CONTEXT_OPERATION_GET: Final = "get"

        # Type aliases for logging operations (Python 3.13+ PEP 695 best practices)
        # Using PEP 695 type keyword for better type checking and IDE support
        type ContextOperationGetLiteral = Literal["get"]
        type ContextOperationModifyLiteral = Literal["bind", "unbind", "clear"]
        type ReturnResultTrueLiteral = Literal[True]
        type ReturnResultFalseLiteral = Literal[False]

    class Literals:
        """Literal type aliases for type-safe annotations.

        Uses Python 3.13+ PEP 695 best practices.

        These type aliases provide strict type checking for common string values
        used throughout the flext-core codebase. They correspond to StrEnum values
        and constant values defined in other namespace classes.

        Uses PEP 695 `type` keyword syntax for modern Python 3.13+ type aliases.
        """

        # Log level literal (references Settings.LogLevel StrEnum members)
        # NÃO duplica strings - referencia o enum member!
        type LogLevelLiteral = Literal[
            FlextConstants.Settings.LogLevel.DEBUG,
            FlextConstants.Settings.LogLevel.INFO,
            FlextConstants.Settings.LogLevel.WARNING,
            FlextConstants.Settings.LogLevel.ERROR,
            FlextConstants.Settings.LogLevel.CRITICAL,
        ]

        # Environment literal (references Settings.Environment StrEnum members)
        # NÃO duplica strings - referencia o enum member!
        type EnvironmentLiteral = Literal[
            FlextConstants.Settings.Environment.DEVELOPMENT,
            FlextConstants.Settings.Environment.STAGING,
            FlextConstants.Settings.Environment.PRODUCTION,
            FlextConstants.Settings.Environment.TESTING,
            FlextConstants.Settings.Environment.LOCAL,
        ]

        # Registration status literal (references FlextConstants.Cqrs.RegistrationStatus StrEnum members)
        type RegistrationStatusLiteral = Literal[
            FlextConstants.Cqrs.RegistrationStatus.ACTIVE,
            FlextConstants.Cqrs.RegistrationStatus.INACTIVE,
        ]

        # State literal - NOTE: This combines multiple StrEnum values
        # References FlextConstants.Domain.Status and FlextConstants.Cqrs.HealthStatus members
        type StateLiteral = Literal[
            FlextConstants.Domain.Status.ACTIVE,
            FlextConstants.Domain.Status.INACTIVE,
            FlextConstants.Domain.Status.PENDING,
            FlextConstants.Cqrs.ProcessingStatus.COMPLETED,
            FlextConstants.Cqrs.ProcessingStatus.FAILED,
            FlextConstants.Cqrs.ProcessingStatus.RUNNING,
            FlextConstants.Cqrs.Status.COMPENSATING,
            FlextConstants.Cqrs.NotificationStatus.SENT,
            FlextConstants.Cqrs.CircuitBreakerStatus.IDLE,
            FlextConstants.Cqrs.HealthStatus.HEALTHY,
            FlextConstants.Cqrs.HealthStatus.DEGRADED,
            FlextConstants.Cqrs.HealthStatus.UNHEALTHY,
        ]

        # =====================================================================
        # ADVANCED ENUM-DERIVED LITERAL TYPES (Python 3.13+ PEP 695 best practices)
        # =====================================================================
        # These Literal types match StrEnum classes defined above
        # Use these for type annotations in Pydantic models and function signatures
        # All Literal values must match their corresponding StrEnum values exactly
        # for type safety and consistency
        #
        # Python 3.13+ advanced patterns:
        # - collections.abc.Mapping for immutable validation mappings
        # - frozenset for O(1) membership testing
        # - discriminated unions for better type narrowing

        # Exceptions.FailureLevel - references FailureLevel StrEnum members
        type FailureLevelLiteral = Literal[
            FlextConstants.Exceptions.FailureLevel.STRICT,
            FlextConstants.Exceptions.FailureLevel.WARN,
            FlextConstants.Exceptions.FailureLevel.PERMISSIVE,
        ]

        # FlextConstants.Cqrs.ProcessingMode - references ProcessingMode StrEnum members
        type ProcessingModeLiteral = Literal[
            FlextConstants.Cqrs.ProcessingMode.BATCH,
            FlextConstants.Cqrs.ProcessingMode.STREAM,
            FlextConstants.Cqrs.ProcessingMode.PARALLEL,
            FlextConstants.Cqrs.ProcessingMode.SEQUENTIAL,
        ]

        # FlextConstants.Cqrs.ProcessingStatus - references ProcessingStatus StrEnum members
        type ProcessingStatusLiteral = Literal[
            FlextConstants.Cqrs.ProcessingStatus.PENDING,
            FlextConstants.Cqrs.ProcessingStatus.RUNNING,
            FlextConstants.Cqrs.ProcessingStatus.COMPLETED,
            FlextConstants.Cqrs.ProcessingStatus.FAILED,
            FlextConstants.Cqrs.ProcessingStatus.CANCELLED,
        ]

        # FlextConstants.Cqrs.ValidationLevel - references ValidationLevel StrEnum members
        type ValidationLevelLiteral = Literal[
            FlextConstants.Cqrs.ValidationLevel.STRICT,
            FlextConstants.Cqrs.ValidationLevel.LENIENT,
            FlextConstants.Cqrs.ValidationLevel.STANDARD,
        ]

        # FlextConstants.Cqrs.ProcessingPhase - references ProcessingPhase StrEnum members
        type ProcessingPhaseLiteral = Literal[
            FlextConstants.Cqrs.ProcessingPhase.PREPARE,
            FlextConstants.Cqrs.ProcessingPhase.EXECUTE,
            FlextConstants.Cqrs.ProcessingPhase.VALIDATE,
            FlextConstants.Cqrs.ProcessingPhase.COMPLETE,
        ]

        # FlextConstants.Cqrs.BindType - references BindType StrEnum members
        type BindTypeLiteral = Literal[
            FlextConstants.Cqrs.BindType.TEMPORARY,
            FlextConstants.Cqrs.BindType.PERMANENT,
        ]

        # FlextConstants.Cqrs.MergeStrategy - references MergeStrategy StrEnum members
        type MergeStrategyLiteral = Literal[
            FlextConstants.Cqrs.MergeStrategy.REPLACE,
            FlextConstants.Cqrs.MergeStrategy.UPDATE,
            FlextConstants.Cqrs.MergeStrategy.MERGE_DEEP,
        ]

        # FlextConstants.Cqrs.Status - references Status StrEnum members
        type StatusLiteral = Literal[
            FlextConstants.Cqrs.Status.PENDING,
            FlextConstants.Cqrs.Status.RUNNING,
            FlextConstants.Cqrs.Status.COMPLETED,
            FlextConstants.Cqrs.Status.FAILED,
            FlextConstants.Cqrs.Status.COMPENSATING,
        ]

        # FlextConstants.Cqrs.HealthStatus - references HealthStatus StrEnum members
        type HealthStatusLiteral = Literal[
            FlextConstants.Cqrs.HealthStatus.HEALTHY,
            FlextConstants.Cqrs.HealthStatus.DEGRADED,
            FlextConstants.Cqrs.HealthStatus.UNHEALTHY,
        ]

        # FlextConstants.Cqrs.TokenType - references TokenType StrEnum members
        type TokenTypeLiteral = Literal[
            FlextConstants.Cqrs.TokenType.BEARER,
            FlextConstants.Cqrs.TokenType.API_KEY,
            FlextConstants.Cqrs.TokenType.JWT,
        ]

        # FlextConstants.Cqrs.NotificationStatus - references NotificationStatus StrEnum members
        type NotificationStatusLiteral = Literal[
            FlextConstants.Cqrs.NotificationStatus.PENDING,
            FlextConstants.Cqrs.NotificationStatus.SENT,
            FlextConstants.Cqrs.NotificationStatus.FAILED,
        ]

        # FlextConstants.Cqrs.TokenStatus - references TokenStatus StrEnum members
        type TokenStatusLiteral = Literal[
            FlextConstants.Cqrs.TokenStatus.PENDING,
            FlextConstants.Cqrs.TokenStatus.RUNNING,
            FlextConstants.Cqrs.TokenStatus.COMPLETED,
            FlextConstants.Cqrs.TokenStatus.FAILED,
        ]

        # Reliability.CircuitBreakerState - references CircuitBreakerState StrEnum members
        type CircuitBreakerStateLiteral = Literal[
            FlextConstants.Reliability.CircuitBreakerState.CLOSED,
            FlextConstants.Reliability.CircuitBreakerState.OPEN,
            FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN,
        ]

        # FlextConstants.Cqrs.CircuitBreakerStatus - references CircuitBreakerStatus StrEnum members
        type CircuitBreakerStatusLiteral = Literal[
            FlextConstants.Cqrs.CircuitBreakerStatus.IDLE,
            FlextConstants.Cqrs.CircuitBreakerStatus.RUNNING,
            FlextConstants.Cqrs.CircuitBreakerStatus.COMPLETED,
            FlextConstants.Cqrs.CircuitBreakerStatus.FAILED,
        ]

        # FlextConstants.Cqrs.BatchStatus - references BatchStatus StrEnum members
        type BatchStatusLiteral = Literal[
            FlextConstants.Cqrs.BatchStatus.PENDING,
            FlextConstants.Cqrs.BatchStatus.PROCESSING,
            FlextConstants.Cqrs.BatchStatus.COMPLETED,
            FlextConstants.Cqrs.BatchStatus.FAILED,
        ]

        # FlextConstants.Cqrs.ExportStatus - references ExportStatus StrEnum members
        type ExportStatusLiteral = Literal[
            FlextConstants.Cqrs.ExportStatus.PENDING,
            FlextConstants.Cqrs.ExportStatus.PROCESSING,
            FlextConstants.Cqrs.ExportStatus.COMPLETED,
            FlextConstants.Cqrs.ExportStatus.FAILED,
        ]

        # FlextConstants.Cqrs.OperationStatus - references OperationStatus StrEnum members
        type OperationStatusLiteral = Literal[
            FlextConstants.Cqrs.OperationStatus.SUCCESS,
            FlextConstants.Cqrs.OperationStatus.FAILURE,
            FlextConstants.Cqrs.OperationStatus.PARTIAL,
        ]

        # FlextConstants.Cqrs.SerializationFormat - references SerializationFormat StrEnum members
        type SerializationFormatLiteral = Literal[
            FlextConstants.Cqrs.SerializationFormat.JSON,
            FlextConstants.Cqrs.SerializationFormat.YAML,
            FlextConstants.Cqrs.SerializationFormat.TOML,
            FlextConstants.Cqrs.SerializationFormat.MSGPACK,
        ]

        # FlextConstants.Cqrs.Compression - references Compression StrEnum members
        type CompressionLiteral = Literal[
            FlextConstants.Cqrs.Compression.NONE,
            FlextConstants.Cqrs.Compression.GZIP,
            FlextConstants.Cqrs.Compression.BZIP2,
            FlextConstants.Cqrs.Compression.LZ4,
        ]

        # FlextConstants.Cqrs.Aggregation - references Aggregation StrEnum members
        type AggregationLiteral = Literal[
            FlextConstants.Cqrs.Aggregation.SUM,
            FlextConstants.Cqrs.Aggregation.AVG,
            FlextConstants.Cqrs.Aggregation.MIN,
            FlextConstants.Cqrs.Aggregation.MAX,
            FlextConstants.Cqrs.Aggregation.COUNT,
        ]

        # FlextConstants.Cqrs.Action - references Action StrEnum members
        type ActionLiteral = Literal[
            FlextConstants.Cqrs.Action.GET,
            FlextConstants.Cqrs.Action.CREATE,
            FlextConstants.Cqrs.Action.UPDATE,
            FlextConstants.Cqrs.Action.DELETE,
            FlextConstants.Cqrs.Action.LIST,
        ]

        # FlextConstants.Cqrs.PersistenceLevel - references PersistenceLevel StrEnum members
        type PersistenceLevelLiteral = Literal[
            FlextConstants.Cqrs.PersistenceLevel.MEMORY,
            FlextConstants.Cqrs.PersistenceLevel.DISK,
            FlextConstants.Cqrs.PersistenceLevel.DISTRIBUTED,
        ]

        # FlextConstants.Cqrs.TargetFormat - references TargetFormat StrEnum members
        type TargetFormatLiteral = Literal[
            FlextConstants.Cqrs.TargetFormat.FULL,
            FlextConstants.Cqrs.TargetFormat.COMPACT,
            FlextConstants.Cqrs.TargetFormat.MINIMAL,
        ]

        # FlextConstants.Cqrs.WarningLevel - references WarningLevel StrEnum members
        type WarningLevelLiteral = Literal[
            FlextConstants.Cqrs.WarningLevel.NONE,
            FlextConstants.Cqrs.WarningLevel.WARN,
            FlextConstants.Cqrs.WarningLevel.ERROR,
        ]

        # FlextConstants.Cqrs.OutputFormat - references OutputFormat StrEnum members
        type OutputFormatLiteral = Literal[
            FlextConstants.Cqrs.OutputFormat.DICT,
            FlextConstants.Cqrs.OutputFormat.JSON,
        ]

        # FlextConstants.Cqrs.Mode - references Mode StrEnum members
        type ModeLiteral = Literal[
            FlextConstants.Cqrs.Mode.VALIDATION,
            FlextConstants.Cqrs.Mode.SERIALIZATION,
        ]

        # Error type literal (for error categorization)
        type ErrorTypeLiteral = Literal[
            "validation",
            "configuration",
            "operation",
            "connection",
            "timeout",
            "authorization",
            "authentication",
            "not_found",
            "attribute_access",
            "conflict",
            "rate_limit",
            "circuit_breaker",
            "type_error",
            "value_error",
            "runtime_error",
            "system_error",
        ]
        """Error type literals for error categorization and type-safe error handling."""

    # ═══════════════════════════════════════════════════════════════════
    # STRENUM + PYDANTIC 2: PADRÃO DEFINITIVO
    # ═══════════════════════════════════════════════════════════════════

    class Domain:
        """Domain-specific constants using StrEnum + Pydantic 2.

        PRINCÍPIO FUNDAMENTAL:
        ─────────────────────
        StrEnum + Pydantic 2 = Validação Automática!

        - NÃO precisa criar Literal separado para validação
        - NÃO precisa criar frozenset para validação
        - NÃO precisa criar AfterValidator
        - Pydantic valida automaticamente contra o StrEnum

        SUBSETS:
        ────────
        Use Literal[Status.MEMBER] para aceitar apenas ALGUNS valores.
        Isso referencia o enum member, não duplica strings!
        """

        # ─────────────────────────────────────────────────────────────────
        # STRENUM: Única declaração necessária
        # ─────────────────────────────────────────────────────────────────
        class Status(StrEnum):
            """Status values - used directly in Pydantic and methods.

            PYDANTIC MODELS:
                 model_config = ConfigDict(use_enum_values=True)
                 status: FlextConstants.Domain.Status

            Resultado:
                 - Aceita "active" or Status.ACTIVE
                 - Serializa como "active" (string)
                 - Valida automaticamente (rejeita valores inválidos)
            """

            ACTIVE = "active"
            INACTIVE = "inactive"
            PENDING = "pending"
            ARCHIVED = "archived"
            FAILED = "failed"

        # ─────────────────────────────────────────────────────────────────
        # SUBSETS: Literal referenciando membros do StrEnum
        # ─────────────────────────────────────────────────────────────────
        # Use para aceitar apenas ALGUNS valores do enum em métodos
        # Isso NÃO duplica strings - referencia o enum member!

        type ActiveStates = Literal[Status.ACTIVE, Status.INACTIVE, Status.PENDING]
        type TerminalStates = Literal[Status.ARCHIVED, Status.FAILED]

        # ─────────────────────────────────────────────────────────────────
        # EXEMPLOS DE USO EM MÉTODOS
        # ─────────────────────────────────────────────────────────────────

        # 1. Aceitar qualquer valor do StrEnum:
        #    def get_by_status(self, status: Status) -> FlextResult[list[Entry]]: ...

        # 2. Aceitar apenas subset do StrEnum:
        #    def process_active(self, status: ActiveStates) -> FlextResult[bool]: ...
        #    def finalize(self, status: TerminalStates) -> FlextResult[bool]: ...

        # ─────────────────────────────────────────────────────────────────
        # TYPEGUARD: Para narrowing em código Python (fora de Pydantic)
        # ─────────────────────────────────────────────────────────────────
        @classmethod
        def is_valid_status(cls, value: object) -> TypeIs[Status]:
            """Type narrowing for validating Status in Python code."""
            return isinstance(value, cls.Status) or (
                isinstance(value, str) and value in cls.Status._value2member_map_
            )

        @classmethod
        def is_active_state(cls, value: object) -> TypeIs[ActiveStates]:
            """Type narrowing for validating subset of active states."""
            if isinstance(value, cls.Status):
                return value in {
                    cls.Status.ACTIVE,
                    cls.Status.INACTIVE,
                    cls.Status.PENDING,
                }
            if isinstance(value, str):
                return value in {
                    cls.Status.ACTIVE.value,
                    cls.Status.INACTIVE.value,
                    cls.Status.PENDING.value,
                }
            return False

        @classmethod
        def is_terminal_state(cls, value: object) -> TypeIs[TerminalStates]:
            """Type narrowing for validating subset of terminal states."""
            if isinstance(value, cls.Status):
                return value in {cls.Status.ARCHIVED, cls.Status.FAILED}
            if isinstance(value, str):
                return value in {cls.Status.ARCHIVED.value, cls.Status.FAILED.value}
            return False

    # ═══════════════════════════════════════════════════════════════════
    # REFERÊNCIAS A FLEXT-CORE (quando necessário reutilizar)
    # ═══════════════════════════════════════════════════════════════════

    class Inherited:
        """Explicit references to inherited constants from FlextConstants.

        Use for documenting which constants from FlextConstants are used
        in this domain, without creating aliases.
        """

        # Apenas referências, não aliases
        # Use FlextConstants.Cqrs.Status directly in code

    class Cqrs:
        """CQRS pattern constants."""

        class HandlerType(StrEnum):
            """CQRS handler types enumeration."""

            COMMAND = "command"
            QUERY = "query"
            EVENT = "event"
            OPERATION = "operation"
            SAGA = "saga"

        # Type aliases for message type discrimination
        # (Python 3.13+ PEP 695 best practices)
        # Using PEP 695 type keyword for better type checking and IDE support
        # These Literals reference HandlerType StrEnum members - NÃO duplica strings!
        type CommandMessageTypeLiteral = Literal[HandlerType.COMMAND]
        type QueryMessageTypeLiteral = Literal[HandlerType.QUERY]
        type EventMessageTypeLiteral = Literal[HandlerType.EVENT]
        # HandlerTypeLiteral references HandlerType StrEnum members
        type HandlerTypeLiteral = Literal[
            HandlerType.COMMAND,
            HandlerType.QUERY,
            HandlerType.EVENT,
            HandlerType.OPERATION,
            HandlerType.SAGA,
        ]
        type ServiceMetricTypeLiteral = Literal[
            "counter",
            "gauge",
            "histogram",
            "summary",
        ]

        DEFAULT_HANDLER_TYPE: HandlerType = HandlerType.COMMAND
        COMMAND_HANDLER_TYPE: HandlerType = HandlerType.COMMAND
        QUERY_HANDLER_TYPE: HandlerType = HandlerType.QUERY
        EVENT_HANDLER_TYPE: HandlerType = HandlerType.EVENT
        OPERATION_HANDLER_TYPE: HandlerType = HandlerType.OPERATION
        SAGA_HANDLER_TYPE: HandlerType = HandlerType.SAGA

        # Valid handler modes as frozenset for O(1) validation
        VALID_HANDLER_MODES: Final[AbstractSet[str]] = frozenset({
            "command",
            "query",
            "event",
            "operation",
            "saga",
        })

        class ProcessingMode(StrEnum):
            """CQRS processing modes enumeration."""

            BATCH = "batch"
            STREAM = "stream"
            PARALLEL = "parallel"
            SEQUENTIAL = "sequential"

        class ProcessingStatus(StrEnum):
            """CQRS processing status enumeration."""

            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"
            CANCELLED = "cancelled"

        class ValidationLevel(StrEnum):
            """CQRS validation levels enumeration."""

            STRICT = "strict"
            LENIENT = "lenient"
            STANDARD = "standard"

        class ProcessingPhase(StrEnum):
            """CQRS processing phases enumeration."""

            PREPARE = "prepare"
            EXECUTE = "execute"
            VALIDATE = "validate"
            COMPLETE = "complete"

        class BindType(StrEnum):
            """CQRS binding types enumeration."""

            TEMPORARY = "temporary"
            PERMANENT = "permanent"

        class MergeStrategy(StrEnum):
            """CQRS merge strategies enumeration."""

            REPLACE = "replace"
            UPDATE = "update"
            MERGE_DEEP = "merge_deep"

        class Status(StrEnum):
            """CQRS status enumeration."""

            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"
            COMPENSATING = "compensating"

        class HealthStatus(StrEnum):
            """CQRS health status enumeration."""

            HEALTHY = "healthy"
            DEGRADED = "degraded"
            UNHEALTHY = "unhealthy"

        class TokenType(StrEnum):
            """CQRS token types enumeration."""

            BEARER = "bearer"
            API_KEY = "api_key"
            JWT = "jwt"

        class NotificationStatus(StrEnum):
            """CQRS notification status enumeration."""

            PENDING = "pending"
            SENT = "sent"
            FAILED = "failed"

        class TokenStatus(StrEnum):
            """CQRS token status enumeration."""

            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"

        class CircuitBreakerStatus(StrEnum):
            """CQRS circuit breaker status enumeration."""

            IDLE = "idle"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"

        class BatchStatus(StrEnum):
            """CQRS batch status enumeration."""

            PENDING = "pending"
            PROCESSING = "processing"
            COMPLETED = "completed"
            FAILED = "failed"

        class ExportStatus(StrEnum):
            """CQRS export status enumeration."""

            PENDING = "pending"
            PROCESSING = "processing"
            COMPLETED = "completed"
            FAILED = "failed"

        class OperationStatus(StrEnum):
            """CQRS operation status enumeration."""

            SUCCESS = "success"
            FAILURE = "failure"
            PARTIAL = "partial"

        class SerializationFormat(StrEnum):
            """CQRS serialization formats enumeration."""

            JSON = "json"
            YAML = "yaml"
            TOML = "toml"
            MSGPACK = "msgpack"

        class Compression(StrEnum):
            """CQRS compression formats enumeration."""

            NONE = "none"
            GZIP = "gzip"
            BZIP2 = "bzip2"
            LZ4 = "lz4"

        class Aggregation(StrEnum):
            """CQRS aggregation functions enumeration."""

            SUM = "sum"
            AVG = "avg"
            MIN = "min"
            MAX = "max"
            COUNT = "count"

        class Action(StrEnum):
            """CQRS action types enumeration."""

            GET = "get"
            CREATE = "create"
            UPDATE = "update"
            DELETE = "delete"
            LIST = "list"

        class PersistenceLevel(StrEnum):
            """CQRS persistence levels enumeration."""

            MEMORY = "memory"
            DISK = "disk"
            DISTRIBUTED = "distributed"

        class TargetFormat(StrEnum):
            """CQRS target formats enumeration."""

            FULL = "full"
            COMPACT = "compact"
            MINIMAL = "minimal"

        class WarningLevel(StrEnum):
            """CQRS warning levels enumeration."""

            NONE = "none"
            WARN = "warn"
            ERROR = "error"

        class OutputFormat(StrEnum):
            """CQRS output formats enumeration."""

            DICT = "dict"
            JSON = "json"

        class Mode(StrEnum):
            """CQRS operation modes enumeration."""

            VALIDATION = "validation"
            SERIALIZATION = "serialization"

        class RegistrationStatus(StrEnum):
            """CQRS registration status enumeration."""

            ACTIVE = "active"
            INACTIVE = "inactive"

        DEFAULT_COMMAND_TYPE: Final[str] = "generic_command"
        DEFAULT_TIMESTAMP: Final[str] = ""
        DEFAULT_TIMEOUT: Final[int] = 30000
        MIN_TIMEOUT: Final[int] = 1000
        MAX_TIMEOUT: Final[int] = 300000
        DEFAULT_COMMAND_TIMEOUT: Final[int] = 0
        DEFAULT_RETRIES: Final[int] = 0
        MIN_RETRIES: Final[int] = 0
        MAX_RETRIES: Final[int] = 5
        DEFAULT_MAX_COMMAND_RETRIES: Final[int] = 0
        DEFAULT_PAGE_SIZE: Final[int] = 10
        MAX_PAGE_SIZE: Final[int] = 1000
        DEFAULT_MAX_VALIDATION_ERRORS: Final[int] = 10
        DEFAULT_MINIMUM_THROUGHPUT: Final[int] = 10
        DEFAULT_PARALLEL_EXECUTION: Final[bool] = False
        DEFAULT_STOP_ON_ERROR: Final[bool] = True
        CQRS_OPERATION_FAILED: Final[str] = "CQRS_OPERATION_FAILED"
        COMMAND_VALIDATION_FAILED: Final[str] = "COMMAND_VALIDATION_FAILED"
        QUERY_VALIDATION_FAILED: Final[str] = "QUERY_VALIDATION_FAILED"
        HANDLER_CONFIG_INVALID: Final[str] = "HANDLER_CONFIG_INVALID"
        COMMAND_PROCESSING_FAILED: Final[str] = "COMMAND_PROCESSING_FAILED"

    class Context:
        """Context management constants."""

        SCOPE_GLOBAL: Final[str] = "global"
        SCOPE_REQUEST: Final[str] = "request"
        SCOPE_SESSION: Final[str] = "session"
        SCOPE_TRANSACTION: Final[str] = "transaction"
        SCOPE_APPLICATION: Final[str] = "application"
        SCOPE_OPERATION: Final[str] = "operation"
        CORRELATION_ID_PREFIX: Final[str] = "flext-"
        CORRELATION_ID_LENGTH: Final[int] = 12
        DEFAULT_CONTEXT_TIMEOUT: Final[int] = 30
        MAX_CONTEXT_DEPTH: Final[int] = 10
        MAX_CONTEXT_SIZE: Final[int] = 1000
        MILLISECONDS_PER_SECOND: Final[int] = 1000
        EXPORT_FORMAT_JSON: Final[str] = "json"
        EXPORT_FORMAT_DICT: Final[str] = "dict"
        METADATA_FIELDS: Final[AbstractSet[str]] = frozenset({
            "user_id",
            "correlation_id",
            "request_id",
            "session_id",
            "tenant_id",
        })

    class Container:
        """Dependency injection container constants."""

        DEFAULT_WORKERS: Final[int] = 4
        TIMEOUT_SECONDS: Final[int] = 30
        MIN_TIMEOUT_SECONDS: Final[int] = 1
        MAX_TIMEOUT_SECONDS: Final[int] = 300
        MAX_CACHE_SIZE: Final[int] = 100

    class Dispatcher:
        """Message dispatcher constants."""

        HANDLER_MODE_COMMAND: Final[str] = "command"
        HANDLER_MODE_QUERY: Final[str] = "query"
        VALID_HANDLER_MODES: Final[tuple[str, ...]] = (
            HANDLER_MODE_COMMAND,
            HANDLER_MODE_QUERY,
        )
        DEFAULT_HANDLER_MODE: Final[str] = HANDLER_MODE_COMMAND
        DEFAULT_AUTO_CONTEXT: Final[bool] = True
        DEFAULT_ENABLE_LOGGING: Final[bool] = True
        DEFAULT_ENABLE_METRICS: Final[bool] = True
        DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
        MIN_TIMEOUT_SECONDS: Final[int] = 1
        MAX_TIMEOUT_SECONDS: Final[int] = 600
        MIN_REGISTRATION_ID_LENGTH: Final[int] = 1
        MIN_REQUEST_ID_LENGTH: Final[int] = 1
        ERROR_INVALID_HANDLER_MODE: Final[str] = (
            "handler_mode must be 'command' or 'query'"
        )
        ERROR_HANDLER_REQUIRED: Final[str] = "handler cannot be None"
        ERROR_MESSAGE_REQUIRED: Final[str] = "message cannot be None"
        ERROR_POSITIVE_TIMEOUT: Final[str] = "timeout must be positive"
        ERROR_INVALID_REGISTRATION_ID: Final[str] = (
            "registration_id must be non-empty string"
        )
        ERROR_INVALID_REQUEST_ID: Final[str] = "request_id must be non-empty string"
        REGISTRATION_STATUS_ACTIVE: Final[str] = "active"
        REGISTRATION_STATUS_INACTIVE: Final[str] = "inactive"
        REGISTRATION_STATUS_ERROR: Final[str] = "error"
        VALID_REGISTRATION_STATUSES: Final[tuple[str, ...]] = (
            REGISTRATION_STATUS_ACTIVE,
            REGISTRATION_STATUS_INACTIVE,
            REGISTRATION_STATUS_ERROR,
        )

    class Pagination:
        """Pagination configuration."""

        DEFAULT_PAGE_NUMBER: Final[int] = 1
        DEFAULT_PAGE_SIZE: Final[int] = 10
        MAX_PAGE_SIZE: Final[int] = 1000
        MIN_PAGE_SIZE: Final[int] = 1
        MIN_PAGE_NUMBER: Final[int] = 1
        MAX_PAGE_NUMBER: Final[int] = 10000

    class Mixins:
        """Constants for mixin operations."""

        FIELD_ID: Final[str] = "unique_id"
        FIELD_NAME: Final[str] = "name"
        FIELD_TYPE: Final[str] = "type"
        FIELD_STATUS: Final[str] = "status"
        FIELD_DATA: Final[str] = "data"
        FIELD_CONFIG: Final[str] = "config"
        FIELD_STATE: Final[str] = "state"
        FIELD_CREATED_AT: Final[str] = "created_at"
        FIELD_UPDATED_AT: Final[str] = "updated_at"
        FIELD_VALIDATED: Final[str] = "validated"
        FIELD_CLASS: Final[str] = "class"
        FIELD_MODULE: Final[str] = "module"
        FIELD_REGISTERED: Final[str] = "registered"
        FIELD_EVENT_NAME: Final[str] = "event_name"
        FIELD_AGGREGATE_ID: Final[str] = "aggregate_id"
        FIELD_OCCURRED_AT: Final[str] = "occurred_at"
        STATE_ACTIVE: Final[str] = "active"
        STATE_INACTIVE: Final[str] = "inactive"
        STATE_PENDING: Final[str] = "pending"
        STATE_COMPLETED: Final[str] = "completed"
        STATE_FAILED: Final[str] = "failed"
        STATE_RUNNING: Final[str] = "running"
        STATE_COMPENSATING: Final[str] = "compensating"
        STATE_SENT: Final[str] = "sent"
        STATE_IDLE: Final[str] = "idle"
        STATE_HEALTHY: Final[str] = "healthy"
        STATE_DEGRADED: Final[str] = "degraded"
        STATE_UNHEALTHY: Final[str] = "unhealthy"
        STATUS_PASS: Final[str] = "PASS"
        STATUS_FAIL: Final[str] = "FAIL"
        STATUS_NO_TARGET: Final[str] = "NO_TARGET"
        STATUS_SKIP: Final[str] = "SKIP"
        STATUS_UNKNOWN: Final[str] = "UNKNOWN"
        IDENTIFIER_UNKNOWN: Final[str] = "unknown"
        IDENTIFIER_DEFAULT: Final[str] = "default"
        IDENTIFIER_ANONYMOUS: Final[str] = "anonymous"
        IDENTIFIER_GUEST: Final[str] = "guest"
        IDENTIFIER_SYSTEM: Final[str] = "system"
        METHOD_HANDLE: Final[str] = "handle"
        METHOD_PROCESS: Final[str] = "process"
        METHOD_EXECUTE: Final[str] = "execute"
        METHOD_PROCESS_COMMAND: Final[str] = "process_command"
        AUTH_BEARER: Final[str] = "bearer"
        AUTH_API_KEY: Final[str] = "api_key"
        AUTH_JWT: Final[str] = "jwt"
        HANDLER_COMMAND: Final[str] = "command"
        HANDLER_QUERY: Final[str] = "query"
        METHOD_VALIDATE: Final[str] = "validate"
        VALIDATION_BASIC: Final[str] = "basic"
        VALIDATION_STRICT: Final[str] = "strict"
        VALIDATION_CUSTOM: Final[str] = "custom"
        DEFAULT_JSON_INDENT: Final[int] = 2
        DEFAULT_ENCODING: Final[str] = "utf-8"
        DEFAULT_SORT_KEYS: Final[bool] = False
        DEFAULT_ENSURE_ASCII: Final[bool] = False
        BOOL_TRUE_STRINGS: Final[AbstractSet[str]] = frozenset({
            "true",
            "1",
            "yes",
            "on",
            "enabled",
        })
        BOOL_FALSE_STRINGS: Final[AbstractSet[str]] = frozenset({
            "false",
            "0",
            "no",
            "off",
            "disabled",
        })
        STRING_TRUE: Final[str] = "true"
        STRING_FALSE: Final[str] = "false"
        DEFAULT_USE_UTC: Final[bool] = True
        DEFAULT_AUTO_UPDATE: Final[bool] = True
        MAX_OPERATION_NAME_LENGTH: Final[int] = 100
        MAX_STATE_VALUE_LENGTH: Final[int] = 50
        MAX_FIELD_NAME_LENGTH: Final[int] = 50
        MIN_FIELD_NAME_LENGTH: Final[int] = 1
        ERROR_EMPTY_OPERATION: Final[str] = "Operation name cannot be empty"
        ERROR_EMPTY_STATE: Final[str] = "State value cannot be empty"
        ERROR_EMPTY_FIELD_NAME: Final[str] = "Field name cannot be empty"
        ERROR_INVALID_ENCODING: Final[str] = "Invalid character encoding"
        ERROR_MISSING_TIMESTAMP_FIELDS: Final[str] = "Required timestamp fields missing"
        ERROR_INVALID_LOG_LEVEL: Final[str] = "Invalid log level"

    class FlextWeb:
        """HTTP protocol constants."""

        HTTP_STATUS_MIN: Final[int] = 100
        HTTP_STATUS_MAX: Final[int] = 599

    class Processing:
        """Processing pipeline constants."""

        DEFAULT_MAX_WORKERS: Final[int] = 4
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        MAX_BATCH_SIZE: Final[int] = 10000

    # =============================================================================
    # ADVANCED VALIDATION HELPERS - Python 3.13+ collections.abc patterns
    # =============================================================================
    # Advanced validation methods using modern Python patterns
    # Similar to flext-cli patterns but adapted for flext-core scope

    # Log level validation mapping
    LOG_LEVEL_VALIDATION_MAP: Final[Mapping[str, str]] = MappingProxyType({
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    })

    # Log level validation set using frozenset for O(1) membership testing
    LOG_LEVEL_VALIDATION_SET: Final[AbstractSet[str]] = frozenset({
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    })

    # Environment validation mapping
    ENVIRONMENT_VALIDATION_MAP: Final[Mapping[str, str]] = MappingProxyType({
        "development": "development",
        "staging": "staging",
        "production": "production",
        "testing": "testing",
        "local": "local",
    })

    # Environment validation set
    ENVIRONMENT_VALIDATION_SET: Final[AbstractSet[str]] = frozenset({
        "development",
        "staging",
        "production",
        "testing",
        "local",
    })

    # =============================================================================
    # GENERALIZED VALIDATION HELPERS - Python 3.13+ collections.abc patterns
    # =============================================================================
    # Generic validation methods that can be reused by any FlextConstants subclass
    # Uses ValidationMappings pattern for type-safe validation

    @classmethod
    def validate_enum_value(
        cls,
        value: str,
        validation_set: AbstractSet[str],
        validation_map: Mapping[str, str],
    ) -> str | None:
        """Validate enum value using advanced patterns.

        Generic validation method that works with any ValidationMappings pattern.
        Uses frozenset for O(1) membership testing and collections.abc.Mapping
        for immutable validation data. Python 3.13+ best practice for validation.

        Args:
            value: Value string to validate
            validation_set: Set of valid values (frozenset for O(1) lookup)
            validation_map: Mapping of values to normalized values

        Returns:
            Valid normalized value string or None if invalid

        """
        if value in validation_set:
            return validation_map.get(value)
        return None

    @classmethod
    def get_valid_enum_values(cls, validation_set: AbstractSet[str]) -> Sequence[str]:
        """Get immutable sequence of valid enum values.

        Generic method to get sorted sequence from validation set.
        Returns collections.abc.Sequence for read-only iteration.
        Python 3.13+ best practice for exposing validation options.

        Args:
            validation_set: Set of valid values

        Returns:
            Immutable sequence of valid value strings

        """
        return tuple(sorted(validation_set))

    @classmethod
    def create_discriminated_union(
        cls,
        *enum_classes: type[StrEnum],
    ) -> Mapping[str, type[StrEnum]]:
        """Create discriminated union mapping from StrEnum classes.

        Advanced helper for creating validation mappings from StrEnum classes.
        Python 3.13+ discriminated union construction pattern.

        Args:
            *enum_classes: StrEnum classes to create union from

        Returns:
            Immutable mapping of enum values to enum classes

        """
        union_map: dict[str, type[StrEnum]] = {}
        for enum_class in enum_classes:
            for member in enum_class.__members__.values():
                union_map[member.value] = enum_class
        return MappingProxyType(union_map)

    # Domain-specific convenience methods using generic helpers
    @classmethod
    def validate_log_level(cls, value: str) -> str | None:
        """Validate log level string using advanced patterns.

        Uses inherited generic validation from validate_enum_value.
        Delegates to generic method for DRY compliance.

        Args:
            value: Log level string to validate

        Returns:
            Valid log level string or None if invalid

        """
        return cls.validate_enum_value(
            value,
            cls.LOG_LEVEL_VALIDATION_SET,
            cls.LOG_LEVEL_VALIDATION_MAP,
        )

    @classmethod
    def validate_environment(cls, value: str) -> str | None:
        """Validate environment string using discriminated union pattern.

        Uses inherited generic validation from validate_enum_value.
        Composes with Settings.Environment StrEnum for comprehensive validation.

        Args:
            value: Environment string to validate

        Returns:
            Valid environment string or None if invalid

        """
        return cls.validate_enum_value(
            value,
            cls.ENVIRONMENT_VALIDATION_SET,
            cls.ENVIRONMENT_VALIDATION_MAP,
        )

    @classmethod
    def get_valid_log_levels(cls) -> Sequence[str]:
        """Get immutable sequence of valid log levels.

        Uses inherited generic method from get_valid_enum_values.
        Returns collections.abc.Sequence for read-only iteration.

        Returns:
            Immutable sequence of valid log level strings

        """
        return cls.get_valid_enum_values(cls.LOG_LEVEL_VALIDATION_SET)

    @classmethod
    def get_valid_environments(cls) -> Sequence[str]:
        """Get immutable sequence of valid environments.

        Uses inherited generic method from get_valid_enum_values.
        Returns collections.abc.Sequence for safe iteration.

        Returns:
            Immutable sequence of valid environment strings

        """
        return cls.get_valid_enum_values(cls.ENVIRONMENT_VALIDATION_SET)

    @classmethod
    def create_enum_literal_mapping(
        cls,
        enum_class: type[StrEnum],
    ) -> Mapping[str, str]:
        """Create discriminated union mapping from StrEnum class.

        Advanced helper for creating validation mappings from StrEnum classes.
        Python 3.13+ discriminated union construction pattern.

        Args:
            enum_class: StrEnum class to create mapping from

        Returns:
            Immutable mapping of enum values to themselves

        """
        return {
            member.value: member.value for member in enum_class.__members__.values()
        }

    # =============================================================================
    # ENUM HELPERS - Extract values from StrEnum for Literal types
    # =============================================================================

    @staticmethod
    def extract_enum_values(enum_class: type[StrEnum]) -> tuple[str, ...]:
        """Extract all values from a StrEnum class as a tuple.

        Python 3.13+ helper for automatically deriving Literal types from StrEnum.
        Use this to create tuple constants that match StrEnum values, ensuring
        Literal types stay in sync with their corresponding StrEnum classes.

        Args:
            enum_class: The StrEnum class to extract values from

        Returns:
            Tuple of all enum values in definition order

        Example:
            >>> values = FlextConstants.extract_enum_values(
            ...     FlextConstants.Settings.LogLevel
            ... )
            >>> # Use values to create Literal type:
            >>> # type LogLevelLiteral = Literal[values[0], values[1], ...]

        """
        # Iterate over enum members using __members__ for proper type checking
        return tuple(member.value for member in enum_class.__members__.values())

    # =============================================================================
    # SHARED DOMAIN CONSTANTS - Cross-cutting domain enums for ecosystem consistency
    # =============================================================================

    class SharedDomain:
        """Cross-cutting domain constants shared across FLEXT ecosystem.

        Provides enums for common domain concepts used across multiple FLEXT projects,
        ensuring consistency and type safety for shared data formats and server types.
        """

        class LdifFormatType(StrEnum):
            """LDIF format types supported across FLEXT ecosystem.

            Defines the standard LDIF format types that all FLEXT LDIF processing
            components must support for interoperability.
            """

            LDIF = "ldif"  # Standard RFC 2849 LDIF format
            DSML = "dsml"  # Directory Services Markup Language

        class ServerType(StrEnum):
            """LDAP server types supported across FLEXT ecosystem.

            Defines the standard LDAP server implementations that FLEXT components
            must handle for comprehensive LDAP directory support.
            """

            RFC = "rfc"  # RFC 4512 compliant (baseline)
            OID = "oid"  # Oracle Internet Directory
            OUD = "oud"  # Oracle Unified Directory
            OPENLDAP = "openldap"  # OpenLDAP implementation

        # Pre-computed validation sets for performance
        _LDIF_FORMAT_VALIDATION_SET: Final[AbstractSet[str]] = frozenset(
            member.value for member in LdifFormatType.__members__.values()
        )

        _SERVER_TYPE_VALIDATION_SET: Final[AbstractSet[str]] = frozenset(
            member.value for member in ServerType.__members__.values()
        )

        @classmethod
        def is_valid_ldif_format(cls, value: str) -> TypeIs[LdifFormatType]:
            """Type guard for LDIF format validation.

            Args:
                value: String to validate as LDIF format

            Returns:
                TypeIs indicating if value is valid LdifFormatType

            """
            return value in cls._LDIF_FORMAT_VALIDATION_SET

        @classmethod
        def is_valid_server_type(cls, value: str) -> TypeIs[ServerType]:
            """Type guard for server type validation.

            Args:
                value: String to validate as server type

            Returns:
                TypeIs indicating if value is valid ServerType

            """
            return value in cls._SERVER_TYPE_VALIDATION_SET

        @classmethod
        def get_valid_ldif_formats(cls) -> Sequence[str]:
            """Get immutable sequence of valid LDIF formats.

            Returns:
                Sequence of valid LDIF format strings

            """
            return tuple(
                member.value for member in cls.LdifFormatType.__members__.values()
            )

        @classmethod
        def get_valid_server_types(cls) -> Sequence[str]:
            """Get immutable sequence of valid server types.

            Returns:
                Sequence of valid server type strings

            """
            return tuple(member.value for member in cls.ServerType.__members__.values())


__all__ = ["FlextConstants"]
