"""FLEXT Core Constants.

Centralized, immutable constants for the FLEXT ecosystem providing configuration
defaults, validation limits, error codes, and runtime enumerations in a pure
Layer 0 module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal


class AutoStrEnum(StrEnum):
    """Automatic string enumeration with lowercase values.

    Generates enum values automatically from member names by converting them
    to lowercase strings, reducing manual value specification.
    """

    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[str],
    ) -> str:
        """Generate enum value from member name.

        Args:
            name: The member name to convert.
            start: Start value (unused - required by parent interface).
            count: Count value (unused - required by parent interface).
            last_values: Last values list (unused - required by parent interface).

        Returns:
            The lowercased member name.

        """
        del start, count, last_values  # Mark as intentionally unused
        return name.lower()


class BiMapping[K, V]:
    """Bidirectional immutable mapping.

    Enables efficient bidirectional lookups using a single dictionary structure.
    Provides immutable forward and inverse mapping views. Values must be unique
    for proper inverse lookup functionality.
    """

    __slots__ = ("_forward", "_inverse")

    def __init__(self, data: dict[K, V]) -> None:
        """Initialize bidirectional mapping.

        Args:
            data: Dictionary to create bidirectional mapping from.

        """
        forward_dict: dict[K, V] = dict(data)
        self._forward: MappingProxyType[K, V] = MappingProxyType(forward_dict)
        self._inverse: MappingProxyType[V, K] = MappingProxyType({
            v: k for k, v in data.items()
        })

    @property
    def forward(self) -> MappingProxyType[K, V]:
        """Get forward mapping.

        Returns:
            Immutable mapping from keys to values.

        """
        return self._forward

    @property
    def inverse(self) -> MappingProxyType[V, K]:
        """Get inverse mapping.

        Returns:
            Immutable mapping from values to keys.

        """
        return self._inverse

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation of the BiMapping.

        """
        return f"BiMapping({dict(self._forward)})"


_DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
_MAX_TIMEOUT_SECONDS: Final[int] = 3600
_MIN_TIMEOUT_SECONDS: Final[int] = 1

_DEFAULT_MAX_CACHE_SIZE: Final[int] = 100

_DEFAULT_BATCH_SIZE: Final[int] = 1000

_DEFAULT_PAGE_SIZE: Final[int] = 10
_MAX_PAGE_SIZE: Final[int] = 1000
_MIN_PAGE_SIZE: Final[int] = 1

_DEFAULT_MAX_RETRY_ATTEMPTS: Final[int] = 3

_DEFAULT_WORKERS: Final[int] = 4

_DEFAULT_POOL_SIZE: Final[int] = 10
_MAX_POOL_SIZE: Final[int] = 100
_MIN_POOL_SIZE: Final[int] = 1

_MAX_NAME_LENGTH: Final[int] = 100
_MAX_OPERATION_NAME_LENGTH: Final[int] = 100

_MAX_PORT_NUMBER: Final[int] = 65535
_MIN_PORT_NUMBER: Final[int] = 1
_MAX_TIMEOUT_VALIDATION_SECONDS: Final[float] = 300.0
_MAX_RETRY_COUNT_VALIDATION: Final[int] = 10
_MAX_HOSTNAME_LENGTH_VALIDATION: Final[int] = 253
_MAX_WORKERS_VALIDATION: Final[int] = 100

_ZERO: Final[int] = 0
_EXPECTED_TUPLE_LENGTH: Final[int] = 2
_DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
_PREVIEW_LENGTH: Final[int] = 50
_DEFAULT_RECOVERY_TIMEOUT_SECONDS: Final[int] = 60
_IDENTIFIER_LENGTH: Final[int] = 12
_MAX_BATCH_SIZE_LIMIT: Final[int] = 10000
_DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
_DEFAULT_MAX_DELAY_SECONDS: Final[float] = 60.0
_MAX_TIMEOUT_SECONDS_PERFORMANCE: Final[int] = 600
_DEFAULT_HOUR_IN_SECONDS: Final[int] = 3600


class FlextConstants:
    """Centralized constants for the FLEXT ecosystem (Layer 0).

    Provides immutable, namespace-organized constants for system configuration,
    validation limits, error handling, and operational defaults.
    """

    class _Base:
        """Shared base StrEnums for common values across all domains."""

        class CommonStatus(StrEnum):
            """Common status values for system operations."""

            ACTIVE = "active"
            INACTIVE = "inactive"
            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"
            CANCELLED = "cancelled"
            COMPENSATING = "compensating"
            ARCHIVED = "archived"

        class CommonAction(StrEnum):
            """Common action types for data operations."""

            GET = "get"
            CREATE = "create"
            UPDATE = "update"
            DELETE = "delete"
            LIST = "list"

        class CommonFormat(StrEnum):
            """Common data serialization formats."""

            JSON = "json"
            YAML = "yaml"
            TOML = "toml"
            XML = "xml"

        class CommonLevel(StrEnum):
            """Common severity levels for operations."""

            NONE = "none"
            WARN = "warn"
            ERROR = "error"

    # Core identifiers
    NAME: Final[str] = "FLEXT"
    ZERO: Final[int] = _ZERO
    INITIAL_TIME: Final[float] = 0.0

    DEFAULT_TIMEOUT_SECONDS: Final[int] = _DEFAULT_TIMEOUT_SECONDS
    MAX_TIMEOUT_SECONDS: Final[int] = _MAX_TIMEOUT_SECONDS
    MIN_TIMEOUT_SECONDS: Final[int] = _MIN_TIMEOUT_SECONDS

    DEFAULT_MAX_CACHE_SIZE: Final[int] = _DEFAULT_MAX_CACHE_SIZE

    DEFAULT_BATCH_SIZE: Final[int] = _DEFAULT_BATCH_SIZE

    DEFAULT_PAGE_SIZE: Final[int] = _DEFAULT_PAGE_SIZE
    MAX_PAGE_SIZE: Final[int] = _MAX_PAGE_SIZE
    MIN_PAGE_SIZE: Final[int] = _MIN_PAGE_SIZE

    DEFAULT_MAX_RETRY_ATTEMPTS: Final[int] = _DEFAULT_MAX_RETRY_ATTEMPTS

    DEFAULT_WORKERS: Final[int] = _DEFAULT_WORKERS

    DEFAULT_POOL_SIZE: Final[int] = _DEFAULT_POOL_SIZE
    MAX_POOL_SIZE: Final[int] = _MAX_POOL_SIZE
    MIN_POOL_SIZE: Final[int] = _MIN_POOL_SIZE

    MAX_NAME_LENGTH: Final[int] = _MAX_NAME_LENGTH
    MAX_OPERATION_NAME_LENGTH: Final[int] = _MAX_OPERATION_NAME_LENGTH

    MAX_PORT_NUMBER: Final[int] = _MAX_PORT_NUMBER
    MIN_PORT_NUMBER: Final[int] = _MIN_PORT_NUMBER
    MAX_TIMEOUT_VALIDATION_SECONDS: Final[float] = _MAX_TIMEOUT_VALIDATION_SECONDS
    MAX_RETRY_COUNT_VALIDATION: Final[int] = _MAX_RETRY_COUNT_VALIDATION
    MAX_HOSTNAME_LENGTH_VALIDATION: Final[int] = _MAX_HOSTNAME_LENGTH_VALIDATION
    MAX_WORKERS_VALIDATION: Final[int] = _MAX_WORKERS_VALIDATION

    EXPECTED_TUPLE_LENGTH: Final[int] = _EXPECTED_TUPLE_LENGTH
    EVENT_TUPLE_SIZE: Final[int] = 2
    """Domain event tuple size (event_type, event_data)."""
    MIN_QUALNAME_PARTS_FOR_WRAPPER: Final[int] = 2
    """Minimum qualname parts for wrapper detection."""
    PERCENTAGE_MULTIPLIER: Final[int] = 100
    """Multiplier for percentage calculations (100 = 100%)."""
    MILLISECONDS_MULTIPLIER: Final[int] = 1000
    """Multiplier to convert seconds to milliseconds."""
    MICROSECONDS_MULTIPLIER: Final[int] = 1000000
    """Multiplier to convert seconds to microseconds."""
    DEFAULT_FAILURE_THRESHOLD: Final[int] = _DEFAULT_FAILURE_THRESHOLD
    PREVIEW_LENGTH: Final[int] = _PREVIEW_LENGTH
    DEFAULT_RECOVERY_TIMEOUT_SECONDS: Final[int] = _DEFAULT_RECOVERY_TIMEOUT_SECONDS
    IDENTIFIER_LENGTH: Final[int] = _IDENTIFIER_LENGTH
    MAX_BATCH_SIZE_LIMIT: Final[int] = _MAX_BATCH_SIZE_LIMIT
    DEFAULT_BACKOFF_MULTIPLIER: Final[float] = _DEFAULT_BACKOFF_MULTIPLIER
    DEFAULT_MAX_DELAY_SECONDS: Final[float] = _DEFAULT_MAX_DELAY_SECONDS
    MAX_TIMEOUT_SECONDS_PERFORMANCE: Final[int] = _MAX_TIMEOUT_SECONDS_PERFORMANCE
    DEFAULT_HOUR_IN_SECONDS: Final[int] = _DEFAULT_HOUR_IN_SECONDS

    class Network:
        """Network configuration constants and limits."""

        LOOPBACK_IP: Final[str] = "127.0.0.1"
        LOCALHOST: Final[str] = "localhost"

        MIN_PORT: Final[int] = _MIN_PORT_NUMBER
        MAX_PORT: Final[int] = _MAX_PORT_NUMBER
        DEFAULT_TIMEOUT: Final[int] = _DEFAULT_TIMEOUT_SECONDS
        DEFAULT_CONNECTION_POOL_SIZE: Final[int] = _DEFAULT_POOL_SIZE
        MAX_CONNECTION_POOL_SIZE: Final[int] = _MAX_POOL_SIZE
        MAX_HOSTNAME_LENGTH: Final[int] = _MAX_HOSTNAME_LENGTH_VALIDATION
        HTTP_STATUS_MIN: Final[int] = 100
        HTTP_STATUS_MAX: Final[int] = 599

    class Validation:
        """Input validation constraints and limits."""

        MIN_NAME_LENGTH: Final[int] = 2
        MAX_NAME_LENGTH: Final[int] = _MAX_NAME_LENGTH
        MAX_EMAIL_LENGTH: Final[int] = 254
        EMAIL_PARTS_COUNT: Final[int] = 2
        LEVEL_PREFIX_PARTS_COUNT: Final[int] = 4
        MIN_PHONE_DIGITS: Final[int] = 10
        MAX_PHONE_DIGITS: Final[int] = 20
        MIN_USERNAME_LENGTH: Final[int] = 3
        MAX_AGE: Final[int] = 150
        MIN_AGE: Final[int] = 0
        PREVIEW_LENGTH: Final[int] = _PREVIEW_LENGTH
        VALIDATION_TIMEOUT_MS: Final[int] = 100
        MAX_UNCOMMITTED_EVENTS: Final[int] = 100
        DISCOUNT_THRESHOLD: Final[int] = 100
        DISCOUNT_RATE: Final[float] = 0.05
        SLOW_OPERATION_THRESHOLD: Final[float] = 0.1
        RESOURCE_LIMIT_MIN: Final[int] = 50
        FILTER_THRESHOLD: Final[int] = 5
        RETRY_COUNT_MAX: Final[int] = 3
        MAX_WORKERS_LIMIT: Final[int] = _MAX_WORKERS_VALIDATION
        MAX_RETRY_STATUS_CODES: Final[int] = 100
        """Maximum number of HTTP status codes allowed in retry configuration."""
        MAX_CUSTOM_VALIDATORS: Final[int] = 50
        """Maximum number of custom validator callables allowed."""

    class Errors:
        """Standardized error codes for system error handling."""

        VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"
        TYPE_ERROR: Final[str] = "TYPE_ERROR"
        ATTRIBUTE_ERROR: Final[str] = "ATTRIBUTE_ERROR"
        CONFIG_ERROR: Final[str] = "CONFIG_ERROR"
        GENERIC_ERROR: Final[str] = "GENERIC_ERROR"
        COMMAND_PROCESSING_FAILED: Final[str] = "COMMAND_PROCESSING_FAILED"
        UNKNOWN_ERROR: Final[str] = "UNKNOWN_ERROR"
        FIRST_ARG_FAILED_MSG: Final[str] = "First argument failed"
        SECOND_ARG_FAILED_MSG: Final[str] = "Second argument failed"
        SERIALIZATION_ERROR: Final[str] = "SERIALIZATION_ERROR"
        MAP_ERROR: Final[str] = "MAP_ERROR"
        BIND_ERROR: Final[str] = "BIND_ERROR"
        CHAIN_ERROR: Final[str] = "CHAIN_ERROR"
        UNWRAP_ERROR: Final[str] = "UNWRAP_ERROR"
        OPERATION_ERROR: Final[str] = "OPERATION_ERROR"
        SERVICE_ERROR: Final[str] = "SERVICE_ERROR"
        BUSINESS_RULE_VIOLATION: Final[str] = "BUSINESS_RULE_VIOLATION"
        BUSINESS_RULE_ERROR: Final[str] = "BUSINESS_RULE_ERROR"
        NOT_FOUND_ERROR: Final[str] = "NOT_FOUND_ERROR"
        NOT_FOUND: Final[str] = "NOT_FOUND"
        RESOURCE_NOT_FOUND: Final[str] = "RESOURCE_NOT_FOUND"
        ALREADY_EXISTS: Final[str] = "ALREADY_EXISTS"
        COMMAND_BUS_ERROR: Final[str] = "COMMAND_BUS_ERROR"
        COMMAND_HANDLER_NOT_FOUND: Final[str] = "COMMAND_HANDLER_NOT_FOUND"
        DOMAIN_EVENT_ERROR: Final[str] = "DOMAIN_EVENT_ERROR"
        TIMEOUT_ERROR: Final[str] = "TIMEOUT_ERROR"
        PROCESSING_ERROR: Final[str] = "PROCESSING_ERROR"
        CONNECTION_ERROR: Final[str] = "CONNECTION_ERROR"
        CONFIGURATION_ERROR: Final[str] = "CONFIGURATION_ERROR"
        EXTERNAL_SERVICE_ERROR: Final[str] = "EXTERNAL_SERVICE_ERROR"
        PERMISSION_ERROR: Final[str] = "PERMISSION_ERROR"
        AUTHENTICATION_ERROR: Final[str] = "AUTHENTICATION_ERROR"
        AUTHORIZATION_ERROR: Final[str] = "AUTHORIZATION_ERROR"
        EXCEPTION_ERROR: Final[str] = "EXCEPTION_ERROR"
        CRITICAL_ERROR: Final[str] = "CRITICAL_ERROR"
        NONEXISTENT_ERROR: Final[str] = "NONEXISTENT_ERROR"

    class Exceptions:
        """Exception handling configuration."""

        class FailureLevel(StrEnum):
            """Exception failure levels."""

            STRICT = "strict"
            WARN = "warn"
            PERMISSIVE = "permissive"

        FAILURE_LEVEL_DEFAULT: Final[FailureLevel] = FailureLevel.PERMISSIVE

        class ErrorType(StrEnum):
            """Error type enumeration for error categorization."""

            VALIDATION = "validation"
            CONFIGURATION = "configuration"
            OPERATION = "operation"
            CONNECTION = "connection"
            TIMEOUT = "timeout"
            AUTHORIZATION = "authorization"
            AUTHENTICATION = "authentication"
            NOT_FOUND = "not_found"
            ATTRIBUTE_ACCESS = "attribute_access"
            CONFLICT = "conflict"
            RATE_LIMIT = "rate_limit"
            CIRCUIT_BREAKER = "circuit_breaker"
            TYPE_ERROR = "type_error"
            VALUE_ERROR = "value_error"
            RUNTIME_ERROR = "runtime_error"
            SYSTEM_ERROR = "system_error"

    class Messages:
        """User-facing message templates."""

        TYPE_MISMATCH: Final[str] = "Type mismatch"
        VALIDATION_FAILED: Final[str] = "Validation failed"
        REDACTED_SECRET: Final[str] = "***REDACTED***"

    class Defaults:
        """Default values."""

        TIMEOUT: Final[int] = _DEFAULT_TIMEOUT_SECONDS
        PAGE_SIZE: Final[int] = 100
        TIMEOUT_SECONDS: Final[int] = _DEFAULT_TIMEOUT_SECONDS
        CACHE_TTL: Final[int] = 300
        DEFAULT_CACHE_TTL: Final[int] = CACHE_TTL
        DEFAULT_MAX_CACHE_SIZE: Final[int] = _DEFAULT_MAX_CACHE_SIZE
        MAX_MESSAGE_LENGTH: Final[int] = 100
        DEFAULT_MIDDLEWARE_ORDER: Final[int] = 0
        OPERATION_TIMEOUT_SECONDS: Final[int] = _DEFAULT_TIMEOUT_SECONDS
        DATABASE_URL: Final[str] = "sqlite:///:memory:"
        DEFAULT_DATABASE_URL: Final[str] = DATABASE_URL

    class Utilities:
        """Utility constants."""

        DEFAULT_ENCODING: Final[str] = "utf-8"
        """Default encoding for string operations."""
        SERIALIZATION_ISO8601: Final = "iso8601"
        """ISO8601 format for datetime serialization."""
        SERIALIZATION_FLOAT: Final = "float"
        """Float format for datetime serialization."""
        SERIALIZATION_BASE64: Final = "base64"
        """Base64 format for bytes serialization."""
        SERIALIZATION_UTF8: Final = "utf8"
        """UTF-8 format for bytes serialization."""
        SERIALIZATION_HEX: Final = "hex"
        """Hex format for bytes serialization."""
        MAX_TIMEOUT_SECONDS: Final[int] = _MAX_TIMEOUT_SECONDS
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
        DEFAULT_ENABLE_TRACING: Final[bool] = False
        DEFAULT_TIMEOUT: Final[int] = _DEFAULT_TIMEOUT_SECONDS
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

        EXTRA_FORBID: Final = "forbid"
        """Extra fields behavior: forbid unknown fields."""
        EXTRA_IGNORE: Final = "ignore"
        """Extra fields behavior: ignore unknown fields."""
        EXTRA_ALLOW: Final = "allow"
        """Extra fields behavior: allow unknown fields."""

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

        class Reliability:
            """Reliability constants for system behavior."""

            MAX_RETRY_ATTEMPTS: Final[int] = 3
            DEFAULT_TIMEOUT: Final[float] = 30.0
            CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
            HEADER_REQUEST_ID: Final[str] = "X-Request-ID"

        PATTERN_UUID: Final[str] = (
            r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
        )
        PATTERN_PATH: Final[str] = r'^[^<>"|?*\x00-\x1F]+$'
        # Identifier patterns (valid names/keys)
        PATTERN_IDENTIFIER: Final[str] = r"^[a-zA-Z][a-zA-Z0-9_]*$"
        """Pattern for valid identifiers (handler names, resource types, etc.)."""
        PATTERN_IDENTIFIER_WITH_UNDERSCORE: Final[str] = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        """Pattern for identifiers that can start with underscore (context keys)."""
        PATTERN_SIMPLE_IDENTIFIER: Final[str] = r"^[a-zA-Z0-9]+$"
        """Pattern for simple alphanumeric identifiers."""
        # Path patterns
        PATTERN_MODULE_PATH: Final[str] = r"^[^:]+:[^:]+$"
        """Pattern for module:class paths (e.g., 'flext_core.dispatcher:FlextDispatcher')."""
        # Timestamp patterns
        PATTERN_ISO8601_TIMESTAMP: Final[str] = (
            r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[Z+\-][0-9:]*)?$"
        )
        """Pattern for ISO 8601 timestamps (optional, allows empty string)."""
        # LDAP/DN patterns
        PATTERN_DN_STRING: Final[str] = r"^(cn|ou|dc)=.*"
        """Pattern for LDAP DN strings (distinguished names)."""
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

        DEFAULT_DB_POOL_SIZE: Final[int] = _DEFAULT_POOL_SIZE
        MIN_DB_POOL_SIZE: Final[int] = _MIN_POOL_SIZE
        MAX_DB_POOL_SIZE: Final[int] = _MAX_POOL_SIZE
        MAX_RETRY_ATTEMPTS_LIMIT: Final[int] = 10
        DEFAULT_TIMEOUT_LIMIT: Final[int] = 300
        MIN_CURRENT_STEP: Final[int] = 0
        DEFAULT_INITIAL_DELAY_SECONDS: Final[float] = 1.0
        MAX_BATCH_SIZE: Final[int] = 10000
        DEFAULT_TIME_RANGE_SECONDS: Final[int] = 3600
        DEFAULT_TTL_SECONDS: Final[int] = 3600
        DEFAULT_VERSION: Final[int] = 1
        MIN_VERSION: Final[int] = 1
        DEFAULT_PAGE_SIZE: Final[int] = _DEFAULT_PAGE_SIZE
        HIGH_MEMORY_THRESHOLD_BYTES: Final[int] = 1073741824
        MAX_TIMEOUT_SECONDS: Final[int] = 600
        MAX_BATCH_OPERATIONS: Final[int] = 1000
        MAX_OPERATION_NAME_LENGTH: Final[int] = _MAX_OPERATION_NAME_LENGTH
        EXPECTED_TUPLE_LENGTH: Final[int] = 2
        DEFAULT_EMPTY_STRING: Final[str] = ""

        class BatchProcessing:
            """Batch processing constants."""

            DEFAULT_SIZE: Final[int] = _DEFAULT_BATCH_SIZE
            MAX_ITEMS: Final[int] = 10000
            MAX_VALIDATION_SIZE: Final[int] = _DEFAULT_BATCH_SIZE

        CLI_PERFORMANCE_CRITICAL_MS: Final[float] = 10000.0

    class Reliability:
        """Reliability thresholds."""

        MAX_RETRY_ATTEMPTS: Final[int] = _DEFAULT_MAX_RETRY_ATTEMPTS
        DEFAULT_MAX_RETRIES: Final[int] = _DEFAULT_MAX_RETRY_ATTEMPTS
        DEFAULT_RETRY_DELAY_SECONDS: Final[int] = 1
        RETRY_BACKOFF_BASE: Final[float] = 2.0
        RETRY_BACKOFF_MAX: Final[float] = 60.0
        DEFAULT_MAX_DELAY_SECONDS: Final[float] = 300.0
        """Default maximum delay in seconds for retry operations."""
        RETRY_COUNT_MIN: Final[int] = 1
        DEFAULT_BACKOFF_STRATEGY: Final[str] = "exponential"
        BACKOFF_STRATEGY_EXPONENTIAL: Final[str] = "exponential"
        """Exponential backoff strategy."""
        BACKOFF_STRATEGY_LINEAR: Final[str] = "linear"
        """Linear backoff strategy."""
        DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
        DEFAULT_RECOVERY_TIMEOUT: Final[int] = 60
        DEFAULT_TIMEOUT_SECONDS: Final[float] = float(_DEFAULT_TIMEOUT_SECONDS)
        DEFAULT_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
        DEFAULT_RATE_LIMIT_MAX_REQUESTS: Final[int] = 100
        DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
        DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[int] = 60
        DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD: Final[int] = 3

        class CircuitBreakerState(StrEnum):
            """Circuit breaker states.

            DRY Pattern:
                StrEnum is the single source of truth. Use CircuitBreakerState.CLOSED.value
                or CircuitBreakerState.CLOSED directly - no base strings needed.
            """

            CLOSED = "closed"
            OPEN = "open"
            HALF_OPEN = "half_open"

    class Security:
        """Security constants."""

        JWT_DEFAULT_ALGORITHM: Final[str] = "HS256"
        CREDENTIAL_BCRYPT_ROUNDS: Final[int] = 12

    class Logging:
        """Logging configuration."""

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
        ASYNC_ENABLED: Final[bool] = True
        ASYNC_QUEUE_SIZE: Final[int] = 10000
        ASYNC_WORKERS: Final[int] = 1
        ASYNC_BLOCK_ON_FULL: Final[bool] = False

        # Log level hierarchy for level-based context filtering
        LEVEL_HIERARCHY: Final[MappingProxyType[str, int]] = MappingProxyType({
            "debug": 10,
            "info": 20,
            "warning": 30,
            "error": 40,
            "critical": 50,
        })
        """Numeric log levels for comparison (lower = more verbose)."""

        class ContextOperation(StrEnum):
            """Context operation types enumeration."""

            BIND = "bind"
            UNBIND = "unbind"
            CLEAR = "clear"
            GET = "get"

    class Literals:
        """Literal type aliases for type-safe annotations."""

        type LogLevelLiteral = Literal[
            FlextConstants.Settings.LogLevel.DEBUG,
            FlextConstants.Settings.LogLevel.INFO,
            FlextConstants.Settings.LogLevel.WARNING,
            FlextConstants.Settings.LogLevel.ERROR,
            FlextConstants.Settings.LogLevel.CRITICAL,
        ]

        type EnvironmentLiteral = Literal[
            FlextConstants.Settings.Environment.DEVELOPMENT,
            FlextConstants.Settings.Environment.STAGING,
            FlextConstants.Settings.Environment.PRODUCTION,
            FlextConstants.Settings.Environment.TESTING,
            internal.invalid,
        ]

        type RegistrationStatusLiteral = Literal[
            "active",
            "inactive",
        ]

        type StateLiteral = Literal[
            FlextConstants.Domain.Status.ACTIVE,
            FlextConstants.Domain.Status.INACTIVE,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPENSATING,
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.SENT,
            FlextConstants.Cqrs.SpecialStatus.IDLE,
            FlextConstants.Cqrs.SpecialStatus.PROCESSING,
            FlextConstants.Cqrs.HealthStatus.HEALTHY,
            FlextConstants.Cqrs.HealthStatus.DEGRADED,
            FlextConstants.Cqrs.HealthStatus.UNHEALTHY,
        ]

        type ActionLiteral = Literal[
            FlextConstants.Cqrs.Action.GET,
            FlextConstants.Cqrs.Action.CREATE,
            FlextConstants.Cqrs.Action.UPDATE,
            FlextConstants.Cqrs.Action.DELETE,
            FlextConstants.Cqrs.Action.LIST,
        ]

        type FailureLevelLiteral = Literal[
            FlextConstants.Exceptions.FailureLevel.STRICT,
            FlextConstants.Exceptions.FailureLevel.WARN,
            FlextConstants.Exceptions.FailureLevel.PERMISSIVE,
        ]

        type ProcessingModeLiteral = Literal[
            FlextConstants.Cqrs.ProcessingMode.BATCH,
            FlextConstants.Cqrs.ProcessingMode.STREAM,
            FlextConstants.Cqrs.ProcessingMode.PARALLEL,
            FlextConstants.Cqrs.ProcessingMode.SEQUENTIAL,
        ]

        type ProcessingStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
            FlextConstants.Cqrs.CommonStatus.CANCELLED,
        ]

        type ValidationLevelLiteral = Literal[
            FlextConstants.Cqrs.ValidationLevel.STRICT,
            FlextConstants.Cqrs.ValidationLevel.LENIENT,
            FlextConstants.Cqrs.ValidationLevel.STANDARD,
        ]

        type ProcessingPhaseLiteral = Literal[
            FlextConstants.Cqrs.ProcessingPhase.PREPARE,
            FlextConstants.Cqrs.ProcessingPhase.EXECUTE,
            FlextConstants.Cqrs.ProcessingPhase.VALIDATE,
            FlextConstants.Cqrs.ProcessingPhase.COMPLETE,
        ]

        type BindTypeLiteral = Literal[
            FlextConstants.Cqrs.BindType.TEMPORARY,
            FlextConstants.Cqrs.BindType.PERMANENT,
        ]

        type MergeStrategyLiteral = Literal[
            FlextConstants.Cqrs.MergeStrategy.REPLACE,
            FlextConstants.Cqrs.MergeStrategy.UPDATE,
            FlextConstants.Cqrs.MergeStrategy.MERGE_DEEP,
        ]

        type StatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
            FlextConstants.Cqrs.CommonStatus.COMPENSATING,
        ]

        type HealthStatusLiteral = Literal[
            FlextConstants.Cqrs.HealthStatus.HEALTHY,
            FlextConstants.Cqrs.HealthStatus.DEGRADED,
            FlextConstants.Cqrs.HealthStatus.UNHEALTHY,
        ]

        type TokenTypeLiteral = Literal[
            FlextConstants.Cqrs.TokenType.BEARER,
            FlextConstants.Cqrs.TokenType.API_KEY,
            FlextConstants.Cqrs.TokenType.JWT,
        ]

        type NotificationStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.SENT,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

        type TokenStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

        type CircuitBreakerStateLiteral = Literal[
            FlextConstants.Reliability.CircuitBreakerState.CLOSED,
            FlextConstants.Reliability.CircuitBreakerState.OPEN,
            FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN,
        ]

        type CircuitBreakerStatusLiteral = Literal[
            FlextConstants.Cqrs.SpecialStatus.IDLE,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

        type BatchStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.PROCESSING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

        type ExportStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.PROCESSING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

        type OperationStatusLiteral = Literal[
            FlextConstants.Cqrs.OperationStatus.SUCCESS,
            FlextConstants.Cqrs.OperationStatus.FAILURE,
            FlextConstants.Cqrs.OperationStatus.PARTIAL,
        ]

        type SerializationFormatLiteral = Literal[
            FlextConstants.Cqrs.SerializationFormat.JSON,
            FlextConstants.Cqrs.SerializationFormat.YAML,
            FlextConstants.Cqrs.SerializationFormat.TOML,
            FlextConstants.Cqrs.SerializationFormat.MSGPACK,
        ]

        type CompressionLiteral = Literal[
            FlextConstants.Cqrs.Compression.NONE,
            FlextConstants.Cqrs.Compression.GZIP,
            FlextConstants.Cqrs.Compression.BZIP2,
            FlextConstants.Cqrs.Compression.LZ4,
        ]

        type AggregationLiteral = Literal[
            FlextConstants.Cqrs.Aggregation.SUM,
            FlextConstants.Cqrs.Aggregation.AVG,
            FlextConstants.Cqrs.Aggregation.MIN,
            FlextConstants.Cqrs.Aggregation.MAX,
            FlextConstants.Cqrs.Aggregation.COUNT,
        ]

        type PersistenceLevelLiteral = Literal[
            FlextConstants.Cqrs.PersistenceLevel.MEMORY,
            FlextConstants.Cqrs.PersistenceLevel.DISK,
            FlextConstants.Cqrs.PersistenceLevel.DISTRIBUTED,
        ]

        type TargetFormatLiteral = Literal[
            FlextConstants.Cqrs.TargetFormat.FULL,
            FlextConstants.Cqrs.TargetFormat.COMPACT,
            FlextConstants.Cqrs.TargetFormat.MINIMAL,
        ]

        type WarningLevelLiteral = Literal[
            FlextConstants.Cqrs.WarningLevel.NONE,
            FlextConstants.Cqrs.WarningLevel.WARN,
            FlextConstants.Cqrs.WarningLevel.ERROR,
        ]

        type OutputFormatLiteral = Literal[
            FlextConstants.Cqrs.OutputFormat.DICT,
            FlextConstants.Cqrs.OutputFormat.JSON,
        ]

        type ModeLiteral = Literal[
            FlextConstants.Cqrs.Mode.VALIDATION,
            FlextConstants.Cqrs.Mode.SERIALIZATION,
        ]

        # Does not duplicate strings - references the enum member!
        type ErrorTypeLiteral = Literal[
            FlextConstants.Exceptions.ErrorType.VALIDATION,
            FlextConstants.Exceptions.ErrorType.CONFIGURATION,
            FlextConstants.Exceptions.ErrorType.OPERATION,
            FlextConstants.Exceptions.ErrorType.CONNECTION,
            FlextConstants.Exceptions.ErrorType.TIMEOUT,
            FlextConstants.Exceptions.ErrorType.AUTHORIZATION,
            FlextConstants.Exceptions.ErrorType.AUTHENTICATION,
            FlextConstants.Exceptions.ErrorType.NOT_FOUND,
            FlextConstants.Exceptions.ErrorType.ATTRIBUTE_ACCESS,
            FlextConstants.Exceptions.ErrorType.CONFLICT,
            FlextConstants.Exceptions.ErrorType.RATE_LIMIT,
            FlextConstants.Exceptions.ErrorType.CIRCUIT_BREAKER,
            FlextConstants.Exceptions.ErrorType.TYPE_ERROR,
            FlextConstants.Exceptions.ErrorType.VALUE_ERROR,
            FlextConstants.Exceptions.ErrorType.RUNTIME_ERROR,
            FlextConstants.Exceptions.ErrorType.SYSTEM_ERROR,
        ]
        """Error type literals for error categorization and type-safe error handling."""

        type ContextOperationGetLiteral = Literal["get"]
        type ContextOperationModifyLiteral = Literal[
            "bind",
            "unbind",
            "clear",
        ]
        type ReturnResultTrueLiteral = Literal[True]
        type ReturnResultFalseLiteral = Literal[False]

        # Order status literals for type-safe operations
        type OrderStatusLiteral = Literal[
            FlextConstants.Domain.OrderStatus.PENDING,
            FlextConstants.Domain.OrderStatus.CONFIRMED,
            FlextConstants.Domain.OrderStatus.SHIPPED,
            FlextConstants.Domain.OrderStatus.DELIVERED,
            FlextConstants.Domain.OrderStatus.CANCELLED,
        ]

        # Active order statuses
        type ActiveOrderStatusLiteral = Literal[
            FlextConstants.Domain.OrderStatus.PENDING,
            FlextConstants.Domain.OrderStatus.CONFIRMED,
            FlextConstants.Domain.OrderStatus.SHIPPED,
        ]

        # Terminal order statuses
        type TerminalOrderStatusLiteral = Literal[
            FlextConstants.Domain.OrderStatus.DELIVERED,
            FlextConstants.Domain.OrderStatus.CANCELLED,
        ]

        # Currency literal for type-safe monetary operations
        type CurrencyLiteral = Literal[
            FlextConstants.Domain.Currency.USD,
            FlextConstants.Domain.Currency.EUR,
            FlextConstants.Domain.Currency.GBP,
            FlextConstants.Domain.Currency.BRL,
        ]

    # ═══════════════════════════════════════════════════════════════════
    # STRENUM + PYDANTIC 2: DEFINITIVE PATTERN
    # ═══════════════════════════════════════════════════════════════════

    class Domain:
        """Domain-specific constants using StrEnum + Pydantic 2."""

        class Status(StrEnum):
            """Status values for domain entities."""

            ACTIVE = "active"
            INACTIVE = "inactive"
            ARCHIVED = "archived"

        class Currency(StrEnum):
            """Currency enumeration for monetary operations."""

            USD = "USD"
            EUR = "EUR"
            GBP = "GBP"
            BRL = "BRL"

        class OrderStatus(StrEnum):
            """Order status enumeration for order lifecycle."""

            PENDING = "pending"
            CONFIRMED = "confirmed"
            SHIPPED = "shipped"
            DELIVERED = "delivered"
            CANCELLED = "cancelled"

        type ActiveStates = Literal[
            FlextConstants.Domain.Status.ACTIVE,
            FlextConstants.Domain.Status.INACTIVE,
        ]
        type TerminalStates = Literal[FlextConstants.Domain.Status.ARCHIVED,]

    # ═══════════════════════════════════════════════════════════════════
    # REFERÊNCIAS A FLEXT-CORE (quando necessário reutilizar)
    # ═══════════════════════════════════════════════════════════════════

    class Inherited:
        """Explicit references to inherited constants from FlextConstants.

        Use for documenting which constants from FlextConstants are used
        in this domain, without creating aliases.
        """

        # Only references, not aliases
        # Use FlextConstants.Cqrs.Status directly in code

    class Cqrs:
        """CQRS pattern constants."""

        class Status(StrEnum):
            """CQRS status enumeration."""

            STOPPED = "stopped"

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
        type CommandMessageTypeLiteral = Literal["command"]
        type QueryMessageTypeLiteral = Literal["query"]
        type EventMessageTypeLiteral = Literal["event"]
        type HandlerTypeLiteral = Literal[
            "command",
            "query",
            "event",
            "operation",
            "saga",
        ]

        class CommonStatus(StrEnum):
            """CQRS common status enumeration."""

            ACTIVE = "active"
            INACTIVE = "inactive"
            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"
            CANCELLED = "cancelled"
            COMPENSATING = "compensating"
            ARCHIVED = "archived"

        class MetricType(StrEnum):
            """Service metric types enumeration."""

            COUNTER = "counter"
            GAUGE = "gauge"
            HISTOGRAM = "histogram"
            SUMMARY = "summary"

        type ServiceMetricTypeLiteral = Literal[
            FlextConstants.Cqrs.MetricType.COUNTER,
            FlextConstants.Cqrs.MetricType.GAUGE,
            FlextConstants.Cqrs.MetricType.HISTOGRAM,
            FlextConstants.Cqrs.MetricType.SUMMARY,
        ]

        class ServiceMetricCategory(StrEnum):
            """Service metric categories enumeration.

            DRY Pattern:
                StrEnum is the single source of truth. Use ServiceMetricCategory.PERFORMANCE.value
                or ServiceMetricCategory.PERFORMANCE directly - no base strings needed.
            """

            PERFORMANCE = "performance"
            ERRORS = "errors"
            THROUGHPUT = "throughput"

        DEFAULT_METRIC_CATEGORIES: Final[tuple[str, ...]] = (
            ServiceMetricCategory.PERFORMANCE,
            ServiceMetricCategory.ERRORS,
            ServiceMetricCategory.THROUGHPUT,
        )
        """Default metric categories for service metrics requests."""

        DEFAULT_HANDLER_TYPE: HandlerType = HandlerType.COMMAND

        class ProcessingMode(StrEnum):
            """CQRS processing modes enumeration."""

            BATCH = "batch"
            STREAM = "stream"
            PARALLEL = "parallel"
            SEQUENTIAL = "sequential"

        type ProcessingStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
            FlextConstants.Cqrs.CommonStatus.CANCELLED,
        ]
        type SagaStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
            FlextConstants.Cqrs.CommonStatus.COMPENSATING,
        ]

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

        class HealthStatus(StrEnum):
            """CQRS health status enumeration."""

            HEALTHY = "healthy"
            DEGRADED = "degraded"
            UNHEALTHY = "unhealthy"

        class SpecialStatus(StrEnum):
            """Special status values not in CommonStatus."""

            SENT = "sent"
            IDLE = "idle"
            PROCESSING = "processing"

        class TokenType(StrEnum):
            """CQRS token types enumeration."""

            BEARER = "bearer"
            API_KEY = "api_key"
            JWT = "jwt"

        # More specialized status literals from CommonStatus and SpecialStatus
        type NotificationStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.SENT,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]
        type TokenStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]
        type CircuitBreakerStatusLiteral = Literal[
            FlextConstants.Cqrs.SpecialStatus.IDLE,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]
        type BatchStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.PROCESSING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]
        type ExportStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.PROCESSING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

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
            """CQRS action types enumeration.

            DRY Pattern:
                StrEnum is the single source of truth. Use Action.GET.value
                or Action.GET directly - no base strings needed.
            """

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
            """CQRS warning levels enumeration.

            DRY Pattern:
                StrEnum is the single source of truth. Use WarningLevel.NONE.value
                or WarningLevel.NONE directly - no base strings needed.
            """

            NONE = "none"
            WARN = "warn"
            ERROR = "error"

        class OutputFormat(StrEnum):
            """CQRS output formats enumeration.

            DRY Pattern:
                StrEnum is the single source of truth. Use OutputFormat.DICT.value
                or OutputFormat.DICT directly - no base strings needed.
            """

            DICT = "dict"
            JSON = "json"

        class Mode(StrEnum):
            """CQRS operation modes enumeration.

            DRY Pattern:
                StrEnum is the single source of truth. Use Mode.VALIDATION.value
                or Mode.VALIDATION directly - no base strings needed.
            """

            VALIDATION = "validation"
            SERIALIZATION = "serialization"

        class RegistrationStatus(StrEnum):
            """CQRS registration status enumeration.

            DRY Pattern:
                Values match _Base.CommonStatus. These StrEnum values are the
                single source of truth.
            """

            ACTIVE = "active"  # Matches _Base.CommonStatus.ACTIVE
            INACTIVE = "inactive"  # Matches _Base.CommonStatus.INACTIVE

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
        DEFAULT_PAGE_SIZE: Final[int] = _DEFAULT_PAGE_SIZE
        MAX_PAGE_SIZE: Final[int] = _MAX_PAGE_SIZE
        DEFAULT_MAX_VALIDATION_ERRORS: Final[int] = 10
        DEFAULT_MINIMUM_THROUGHPUT: Final[int] = 10
        DEFAULT_PARALLEL_EXECUTION: Final[bool] = False
        DEFAULT_STOP_ON_ERROR: Final[bool] = True
        CQRS_OPERATION_FAILED: Final[str] = "CQRS_OPERATION_FAILED"
        COMMAND_VALIDATION_FAILED: Final[str] = "COMMAND_VALIDATION_FAILED"
        QUERY_VALIDATION_FAILED: Final[str] = "QUERY_VALIDATION_FAILED"
        HANDLER_CONFIG_INVALID: Final[str] = "HANDLER_CONFIG_INVALID"

    class Context:
        """Context management constants."""

        SCOPE_GLOBAL: Final[str] = "global"
        SCOPE_REQUEST: Final[str] = "request"
        SCOPE_USER: Final[str] = "user"
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

        class MetadataField(StrEnum):
            """Metadata field names used in context operations."""

            USER_ID = "user_id"
            CORRELATION_ID = "correlation_id"
            REQUEST_ID = "request_id"
            SESSION_ID = "session_id"
            TENANT_ID = "tenant_id"

        # Context operation names for statistics
        OPERATION_SET: Final[str] = "set"
        OPERATION_GET: Final[str] = "get"
        OPERATION_REMOVE: Final[str] = "remove"
        OPERATION_CLEAR: Final[str] = "clear"
        # Context keys
        KEY_OPERATION_ID: Final[str] = "operation_id"
        KEY_USER_ID: Final[str] = "user_id"
        KEY_CORRELATION_ID: Final[str] = "correlation_id"
        KEY_PARENT_CORRELATION_ID: Final[str] = "parent_correlation_id"
        KEY_SERVICE_NAME: Final[str] = "service_name"
        KEY_OPERATION_NAME: Final[str] = "operation_name"
        KEY_REQUEST_ID: Final[str] = "request_id"
        KEY_SERVICE_VERSION: Final[str] = "service_version"
        KEY_OPERATION_START_TIME: Final[str] = "operation_start_time"
        KEY_OPERATION_METADATA: Final[str] = "operation_metadata"
        KEY_REQUEST_TIMESTAMP: Final[str] = "request_timestamp"
        # HTTP headers for context propagation
        HEADER_CORRELATION_ID: Final[str] = "X-Correlation-Id"
        HEADER_PARENT_CORRELATION_ID: Final[str] = "X-Parent-Correlation-Id"
        HEADER_SERVICE_NAME: Final[str] = "X-Service-Name"
        HEADER_USER_ID: Final[str] = "X-User-Id"
        # Metadata keys for operation timing
        METADATA_KEY_START_TIME: Final[str] = "start_time"
        METADATA_KEY_END_TIME: Final[str] = "end_time"
        METADATA_KEY_DURATION_SECONDS: Final[str] = "duration_seconds"

    class Container:
        """Dependency injection container constants."""

        DEFAULT_WORKERS: Final[int] = _DEFAULT_WORKERS
        TIMEOUT_SECONDS: Final[int] = _DEFAULT_TIMEOUT_SECONDS
        MIN_TIMEOUT_SECONDS: Final[int] = _MIN_TIMEOUT_SECONDS
        MAX_TIMEOUT_SECONDS: Final[int] = 300
        MAX_CACHE_SIZE: Final[int] = _DEFAULT_MAX_CACHE_SIZE
        DEFAULT_MAX_SERVICES: Final[int] = 1000
        """Default maximum number of services allowed in container."""
        DEFAULT_MAX_FACTORIES: Final[int] = 500
        """Default maximum number of factories allowed in container."""
        MAX_FACTORIES: Final[int] = 5000
        """Maximum number of factories allowed in container."""

    class Dispatcher:
        """Message dispatcher constants."""

        THREAD_NAME_PREFIX: Final[str] = "flext-dispatcher"
        """Thread name prefix for dispatcher thread pool executor."""
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
        DEFAULT_TIMEOUT_SECONDS: Final[int] = _DEFAULT_TIMEOUT_SECONDS
        MIN_TIMEOUT_SECONDS: Final[int] = _MIN_TIMEOUT_SECONDS
        MAX_TIMEOUT_SECONDS: Final[int] = 600
        MIN_REGISTRATION_ID_LENGTH: Final[int] = 1
        DEFAULT_DISPATCHER_PATH: Final[str] = "flext_core.dispatcher:FlextDispatcher"
        """Default dispatcher implementation path."""
        DEFAULT_SERVICE_NAME: Final[str] = "default_service"
        """Default service name for service models."""
        DEFAULT_RESOURCE_TYPE: Final[str] = "default_resource"
        """Default resource type for service models."""
        MIN_REQUEST_ID_LENGTH: Final[int] = 1
        SINGLE_HANDLER_ARG_COUNT: Final[int] = 1
        TWO_HANDLER_ARG_COUNT: Final[int] = 2
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
        """Registration status: active (matches Cqrs.RegistrationStatus.ACTIVE.value)."""
        REGISTRATION_STATUS_INACTIVE: Final[str] = "inactive"
        """Registration status: inactive (matches Cqrs.RegistrationStatus.INACTIVE.value)."""
        REGISTRATION_STATUS_ERROR: Final[str] = "error"
        """Registration status: error (not part of RegistrationStatus StrEnum)."""
        VALID_REGISTRATION_STATUSES: Final[tuple[str, ...]] = (
            REGISTRATION_STATUS_ACTIVE,
            REGISTRATION_STATUS_INACTIVE,
            REGISTRATION_STATUS_ERROR,
        )

    class Pagination:
        """Pagination configuration."""

        DEFAULT_PAGE_NUMBER: Final[int] = 1
        DEFAULT_PAGE_SIZE: Final[int] = _DEFAULT_PAGE_SIZE
        MAX_PAGE_SIZE: Final[int] = _MAX_PAGE_SIZE
        MIN_PAGE_SIZE: Final[int] = _MIN_PAGE_SIZE
        MIN_PAGE_NUMBER: Final[int] = 1
        MAX_PAGE_NUMBER: Final[int] = 10000
        # Example/default values for pagination utilities
        DEFAULT_PAGE_SIZE_EXAMPLE: Final[int] = 20
        """Default page size for examples and utilities (different from DEFAULT_PAGE_SIZE)."""
        MAX_PAGE_SIZE_EXAMPLE: Final[int] = 1000
        """Maximum page size for examples and utilities."""

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
        """State: active."""
        STATE_INACTIVE: Final[str] = "inactive"
        """State: inactive."""
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
        OPERATION_OVERRIDE: Final[str] = "override"
        """Override operation mode."""
        OPERATION_COLLECTION: Final[str] = "collection"
        """Collection operation mode."""
        AUTH_BEARER: Final[str] = "bearer"
        AUTH_API_KEY: Final[str] = "api_key"
        AUTH_JWT: Final[str] = "jwt"
        HANDLER_COMMAND: Final[str] = "command"
        HANDLER_QUERY: Final[str] = "query"
        METHOD_VALIDATE: Final[str] = "validate"
        DEFAULT_JSON_INDENT: Final[int] = 2
        DEFAULT_SORT_KEYS: Final[bool] = False
        DEFAULT_ENSURE_ASCII: Final[bool] = False

        class BoolTrueValue(StrEnum):
            """String representations of boolean true values."""

            TRUE = "true"
            ONE = "1"
            YES = "yes"
            ON = "on"
            ENABLED = "enabled"

        class BoolFalseValue(StrEnum):
            """String representations of boolean false values."""

            FALSE = "false"
            ZERO = "0"
            NO = "no"
            OFF = "off"
            DISABLED = "disabled"

        STRING_TRUE: Final[str] = "true"
        STRING_FALSE: Final[str] = "false"
        DEFAULT_USE_UTC: Final[bool] = True
        DEFAULT_AUTO_UPDATE: Final[bool] = True
        MAX_OPERATION_NAME_LENGTH: Final[int] = _MAX_OPERATION_NAME_LENGTH
        MAX_STATE_VALUE_LENGTH: Final[int] = 50
        MAX_FIELD_NAME_LENGTH: Final[int] = 50
        MIN_FIELD_NAME_LENGTH: Final[int] = 1
        ERROR_EMPTY_OPERATION: Final[str] = "Operation name cannot be empty"
        ERROR_EMPTY_STATE: Final[str] = "State value cannot be empty"
        ERROR_EMPTY_FIELD_NAME: Final[str] = "Field name cannot be empty"
        ERROR_INVALID_ENCODING: Final[str] = "Invalid character encoding"
        ERROR_MISSING_TIMESTAMP_FIELDS: Final[str] = "Required timestamp fields missing"
        ERROR_INVALID_LOG_LEVEL: Final[str] = "Invalid log level"

    class Processing:
        """Processing pipeline constants."""

        DEFAULT_MAX_WORKERS: Final[int] = _DEFAULT_WORKERS
        DEFAULT_BATCH_SIZE: Final[int] = _DEFAULT_BATCH_SIZE
        # Tuple pattern validation constants
        PATTERN_TUPLE_MIN_LENGTH: Final[int] = 2
        """Minimum length for tuple patterns in parsing operations."""
        PATTERN_TUPLE_MAX_LENGTH: Final[int] = 3
        """Maximum length for tuple patterns in parsing operations."""

    class Discovery:
        """Constants for auto-discovery of handlers/factories.

        Used by `@h.handler()` and `@d.factory()` decorators to mark methods
        for automatic discovery and registration by FlextService and FlextContainer.

        Attributes:
            HANDLER_ATTR: Attribute name for storing handler decorator configuration.
            FACTORY_ATTR: Attribute name for storing factory decorator configuration.
            DEFAULT_PRIORITY: Default handler priority (higher = processed first).
            DEFAULT_TIMEOUT: Default handler timeout (None = no timeout).

        """

        HANDLER_ATTR: Final[str] = "_flext_handler_config_"
        """Attribute name for storing handler decorator configuration on methods."""

        FACTORY_ATTR: Final[str] = "_flext_factory_config_"
        """Attribute name for storing factory decorator configuration on functions."""

        DEFAULT_PRIORITY: Final[int] = 0
        """Default priority for handlers (0 = normal priority)."""

        DEFAULT_TIMEOUT: Final[float | None] = None
        """Default timeout for handlers (None = no timeout)."""

    class Test:
        """Test constants for unit and integration testing.

        Provides standardized test data values used across the FLEXT ecosystem.
        These constants ensure consistent test behavior and avoid magic strings.
        """

        DEFAULT_PASSWORD: Final[str] = "test_password"
        """Default password for test user authentication."""

        NONEXISTENT_USERNAME: Final[str] = "nonexistent"
        """Username that should not exist in test scenarios."""

    # Root-level aliases for commonly used constants
    TIMEOUT: Final[int] = Network.DEFAULT_TIMEOUT
    """Default timeout in seconds."""
    VALIDATION_ERROR: Final[str] = Errors.VALIDATION_ERROR
    """Validation error code."""
    NOT_FOUND: Final[str] = Errors.NOT_FOUND
    """Not found error code."""
    ENCODING: Final[str] = Utilities.DEFAULT_ENCODING
    """Default encoding."""
    PAGE_SIZE: Final[int] = Pagination.DEFAULT_PAGE_SIZE
    """Default page size."""
    MAX_RETRIES: Final[int] = Reliability.MAX_RETRY_ATTEMPTS
    """Maximum retry attempts."""


c = FlextConstants

__all__ = ["FlextConstants", "c"]
