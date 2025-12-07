"""Immutable, shared constants for the FLEXT ecosystem.

Centralize configuration defaults, validation limits, error codes, and runtime
enums in a pure Layer 0 module so dispatcher handlers, services, and utilities
share the same source of truth without circular imports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from enum import StrEnum
from typing import Final, Literal

from pydantic import ConfigDict

# Centralized timeout constants (reused by all namespaces)
_DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
_MAX_TIMEOUT_SECONDS: Final[int] = 3600
_MIN_TIMEOUT_SECONDS: Final[int] = 1

# Centralized cache constants (reused by all namespaces)
_DEFAULT_MAX_CACHE_SIZE: Final[int] = 100

# Centralized batch size constants (reused by all namespaces)
_DEFAULT_BATCH_SIZE: Final[int] = 1000

# Centralized page size constants (reused by all namespaces)
_DEFAULT_PAGE_SIZE: Final[int] = 10
_MAX_PAGE_SIZE: Final[int] = 1000
_MIN_PAGE_SIZE: Final[int] = 1

# Centralized retry constants (reused by all namespaces)
_DEFAULT_MAX_RETRY_ATTEMPTS: Final[int] = 3

# Centralized workers constants (reused by all namespaces)
_DEFAULT_WORKERS: Final[int] = 4

# Centralized pool size constants (reused by all namespaces)
_DEFAULT_POOL_SIZE: Final[int] = 10
_MAX_POOL_SIZE: Final[int] = 100
_MIN_POOL_SIZE: Final[int] = 1

# Centralized name length constants (reused by all namespaces)
_MAX_NAME_LENGTH: Final[int] = 100
_MAX_OPERATION_NAME_LENGTH: Final[int] = 100

# Centralized validation limit constants (reused by typings.py Field annotations)
_MAX_PORT_NUMBER: Final[int] = 65535
_MIN_PORT_NUMBER: Final[int] = 1
_MAX_TIMEOUT_VALIDATION_SECONDS: Final[float] = 300.0
_MAX_RETRY_COUNT_VALIDATION: Final[int] = 10
_MAX_HOSTNAME_LENGTH_VALIDATION: Final[int] = 253
_MAX_WORKERS_VALIDATION: Final[int] = 100

# Centralized generic constants (reused across multiple domains)
_ZERO: Final[int] = 0
_EXPECTED_TUPLE_LENGTH: Final[int] = 2
_DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
_PREVIEW_LENGTH: Final[int] = 50
_DEFAULT_RECOVERY_TIMEOUT_SECONDS: Final[int] = 60
_IDENTIFIER_LENGTH: Final[int] = 12  # UUID, correlation ID, bcrypt rounds
_MAX_BATCH_SIZE_LIMIT: Final[int] = 10000
_DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
_DEFAULT_MAX_DELAY_SECONDS: Final[float] = 60.0
_MAX_TIMEOUT_SECONDS_PERFORMANCE: Final[int] = 600
_DEFAULT_HOUR_IN_SECONDS: Final[int] = 3600


class FlextConstants:
    """Namespace-organized constants for configuration and validation.

    Architecture: Layer 0 (Pure Constants - No Layer 1+ Imports)
    ============================================================
    Provides immutable constants organized in namespaces for configuration,
    validation, error handling, and system defaults. All constants use
    typing.Final for immutability and serve as the single source of truth
    throughout the FLEXT ecosystem.

    **Structural Typing and Protocol Compliance**:
    This class satisfies p.Constants through structural typing
    (duck typing) via the following protocol-compliant interface:
    - 40+ nested namespace classes providing organized constant access
    - typing.Final type annotations for immutability guarantees
    - Hierarchical access patterns (FlextConstants.Namespace.CONSTANT)
    - Compile-time constant verification via strict type checking

    **Core Features**:
    1. **Immutable Constants**: All constants wrapped with typing.Final
    2. **Namespace Organization**: 30+ nested classes for logical grouping
    3. **Type Safety**: Complete type annotations (no Any, no implicit types)
    4. **Hierarchical Access**: Multi-level namespace access patterns
    5. **Ecosystem Single Source of Truth**: Centralized configuration
    7. **Integration Ready**: Seamless use throughout 32+ FLEXT projects
    8. **Error Code Registry**: 50+ error codes with reserved purposes
    9. **Literal Type Definitions**: Type-safe literal types for annotations
    10. **StrEnum Integration**: Platform and configuration StrEnums

    **Namespace Organization**:
    - **Core**: Basic identifiers and version information
    - **Network**: Network-related defaults, timeouts, connection pools
    - **Validation**: Input validation limits and patterns
    - **Errors**: 50+ error codes with purpose documentation
    - **Messages**: User-facing message templates
    - **Entities**: Entity-specific constants
    - **Defaults**: Default values for common operations
    - **Limits**: Upper bounds for resource usage
    - **Utilities**: Utility constants (encoding, batch sizes)
    - **Patterns**: Regex patterns for validation
    - **Config**: Configuration defaults and StrEnum types
    - **Enums**: Shared enumeration types (FieldType, WorkspaceStatus)
    - **Platform**: HTTP, database, file type, MIME type constants
    - **Observability**: Logging and monitoring defaults
    - **Performance**: Performance thresholds, connection limits, batch sizes
    - **Reliability**: Retry, circuit breaker, rate limiting constants
    - **Security**: JWT, authentication, session management constants
    - **Logging**: Log levels, output formats, context inclusion
    - **Cqrs**: CQRS patterns, handler types, timeout/retry settings
    - **Context**: Context scope, correlation ID, export formats
    - **Container**: Dependency injection container constants
    - **Dispatcher**: Message dispatcher modes and settings
    - **Pagination**: Pagination configuration constants
    - **Mixins**: Field names, status constants, default states
    - **Web**: Web application timeouts
    - **Batch**: Batch processing size constants
    - **Processing**: Processing pipeline constants

    **Error Code Registry** (50+ codes with explicit purposes):
    - **Core errors**: VALIDATION_ERROR, TYPE_ERROR, ATTRIBUTE_ERROR (active)
    - **FlextResult operations**: MAP_ERROR, BIND_ERROR, CHAIN_ERROR, UNWRAP_ERROR
    - **Business logic**: OPERATION_ERROR, SERVICE_ERROR, BUSINESS_RULE_VIOLATION
    - **CRUD/Resource**: NOT_FOUND_ERROR, ALREADY_EXISTS, RESOURCE_NOT_FOUND
    - **CQRS**: COMMAND_BUS_ERROR, COMMAND_HANDLER_NOT_FOUND, DOMAIN_EVENT_ERROR
    - **Infrastructure**: TIMEOUT_ERROR, CONNECTION_ERROR, CONFIGURATION_ERROR
    - **Security**: PERMISSION_ERROR, AUTHENTICATION_ERROR
    - **System**: EXCEPTION_ERROR, CRITICAL_ERROR, NONEXISTENT_ERROR

    **Constants Organization Standards (FLEXT Standardization Plan)**:
    All FLEXT projects MUST follow these patterns for constants organization:

    1. **Constants Organization**:
       - ALL constants MUST be inside the main constants class
         (no module-level constants)
       - Use nested classes for logical grouping (e.g., class Errors:)
       - Layer 0 purity: Only constants, no functions or behavior

    2. **Inheritance Pattern**:
       - All domain constants MUST extend FlextConstants directly
       - Use composition for domain relationships: Reference other domain constants
       - Example: class FlextLdapConstants(FlextConstants):
           with LdifConstants = FlextLdifConstants

    3. **Declaration Style**:
       - Use Final[Type] for ALL immutable constants
       - Use ClassVar[Type] only for special cases (rare)
       - Always specify explicit types - no implicit typing
       - Use StrEnum for string enumerations

    4. **Composition Pattern**:
       - Reference core constants via composition when extending functionality
       - Example: CoreErrors = FlextConstants.Errors for easy access
       - Use inheritance for domain-specific extensions

    5. **Import Pattern**:
       - Import only FlextConstants from flext_core
       - Additional imports only for StrEnum, Literal, etc.
       - NO wildcard imports

    6. **Documentation Pattern**:
       - Comprehensive class docstrings with usage examples
       - Section headers for different constant groups
       - Type hints in comments where helpful

    **Usage Patterns**:

    1. Direct namespace access:
        >>> from flext_core import FlextConstants
        >>> error = FlextConstants.Errors.VALIDATION_ERROR
        >>> timeout = FlextConstants.Settings.DEFAULT_TIMEOUT

    2. Dynamic access via __getitem__:
        >>> const = FlextConstants["Errors.VALIDATION_ERROR"]
        >>> port = FlextConstants["Platform.FLEXT_API_PORT"]

    3. Literal type annotations:
        >>> from flext_core import FlextConstants
        >>> def process_error(code: FlextConstants.ErrorCategory) -> None: ...

    4. StrEnum access for configuration:
        >>> env = FlextConstants.Settings.Environment.PRODUCTION
        >>> log_level = FlextConstants.Settings.LogLevel.INFO

    5. Error code patterns:
        >>> validation_errors = [
        ...     FlextConstants.Errors.VALIDATION_ERROR,
        ...     FlextConstants.Errors.TYPE_ERROR,
        ... ]

    6. Network and timeout constants:
        >>> timeout_sec = FlextConstants.Network.DEFAULT_TIMEOUT
        >>> pool_size = FlextConstants.Network.DEFAULT_CONNECTION_POOL_SIZE

    7. Performance thresholds:
        >>> batch_size = FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE
        >>> max_retries = FlextConstants.Reliability.MAX_RETRY_ATTEMPTS

    8. Security and authentication:
        >>> jwt_algo = FlextConstants.Security.JWT_DEFAULT_ALGORITHM
        >>> bcrypt_rounds = FlextConstants.Security.CREDENTIAL_BCRYPT_ROUNDS

    9. Platform constants:
        >>> api_port = FlextConstants.Platform.FLEXT_API_PORT
        >>> json_mime = FlextConstants.Platform.MIME_TYPE_JSON
        >>> email_pattern = FlextConstants.Platform.PATTERN_EMAIL

    10. Logging configuration:
        >>> log_level = FlextConstants.Logging.DEFAULT_LEVEL
        >>> include_context = FlextConstants.Logging.INCLUDE_CONTEXT

    **Advanced Patterns**:

    1. Error code grouping for handling:
        >>> business_errors = [
        ...     FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
        ...     FlextConstants.Errors.BUSINESS_RULE_ERROR,
        ... ]

    2. Configuration profile constants:
        >>> if env == FlextConstants.Settings.Environment.PRODUCTION:
        ...     timeout = FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS

    3. Validation limit application:
        >>> if len(name) < FlextConstants.Validation.MIN_NAME_LENGTH:
        ...     return r[str].fail(FlextConstants.Errors.VALIDATION_ERROR)

    4. Dynamic constant path resolution:
        >>> error_path = "Errors.VALIDATION_ERROR"
        >>> error_code = FlextConstants[error_path]

    5. Type-safe literal unions:
        >>> handler_types: list[FlextConstants.Cqrs.HandlerType] = [
            ...     FlextConstants.Cqrs.HandlerType.COMMAND,
            ...     FlextConstants.Cqrs.HandlerType.QUERY,
            ... ]

    6. Platform-specific constant selection:
        >>> if FlextConstants.Settings.Environment.DEVELOPMENT:
        ...     log_level = FlextConstants.Logging.DEFAULT_LEVEL_DEVELOPMENT
        ... else:
        ...     log_level = FlextConstants.Logging.DEFAULT_LEVEL_PRODUCTION

    7. Performance threshold application:
        >>> if elapsed_ms > FlextConstants.Performance.CLI_PERFORMANCE_CRITICAL_MS:
        ...     logger.warning("Critical performance threshold exceeded")

    8. Circuit breaker initialization:
        >>> circuit_breaker = CircuitBreaker(
        ...     failure_threshold=FlextConstants.Reliability.DEFAULT_FAILURE_THRESHOLD,
        ...     recovery_timeout=FlextConstants.Reliability.DEFAULT_RECOVERY_TIMEOUT,
        ... )

    **Complete Usage Example**:
            >>> from flext_core.result import r
        >>>
        >>> def validate_input(name: str) -> r[str]:
        ...     if len(name) < FlextConstants.Validation.MIN_NAME_LENGTH:
        ...         error = FlextConstants.Errors.VALIDATION_ERROR
        ...         return r[str].fail(error)
        ...
        ...     if len(name) > FlextConstants.Validation.MAX_NAME_LENGTH:
        ...         error = FlextConstants.Errors.VALIDATION_ERROR
        ...         return r[str].fail(error)
        ...
        ...     return r[str].ok(name)
        >>>
        >>> # Use in configuration
        >>> timeout = FlextConstants.Network.DEFAULT_TIMEOUT
        >>> env = FlextConstants.Settings.Environment.PRODUCTION
        >>>
        >>> # Use in error handling
        >>> if result.error == FlextConstants.Errors.VALIDATION_ERROR:
        ...     handle_validation_error(result)

    **Thread Safety**:
    All constants are immutable (typing.Final). Access is thread-safe as
    there is no mutable state in this module.

    **Performance Characteristics**:
    - O(1) constant access via namespace attributes
    - O(1) constant access via __getitem__ (dict-like lookup)
    - No runtime compilation or validation overhead
    - Constants are resolved at import time

    **Integration with FLEXT Ecosystem**:
    - **FlextResult**: Uses error codes from Errors namespace
    - **FlextConfig**: References Config and Platform constants
    - **FlextLogger**: Uses Logging constants for configuration
    - **FlextDispatcher**: Uses Cqrs constants for handler types
    - **FlextContext**: Uses Context constants for scope management
    - **FlextContainer**: Uses Container constants for lifecycle
    - **Domain Services**: Reference validation and business error codes
    - **All 32+ projects**: Depend on this centralized constant registry
    """

    # Core identifiers
    NAME: Final[str] = "FLEXT"
    ZERO: Final[int] = _ZERO
    INITIAL_TIME: Final[float] = 0.0

    # Centralized timeout constants (reused by all namespaces)
    DEFAULT_TIMEOUT_SECONDS: Final[int] = _DEFAULT_TIMEOUT_SECONDS
    MAX_TIMEOUT_SECONDS: Final[int] = _MAX_TIMEOUT_SECONDS
    MIN_TIMEOUT_SECONDS: Final[int] = _MIN_TIMEOUT_SECONDS

    # Centralized cache constants (reused by all namespaces)
    DEFAULT_MAX_CACHE_SIZE: Final[int] = _DEFAULT_MAX_CACHE_SIZE

    # Centralized batch size constants (reused by all namespaces)
    DEFAULT_BATCH_SIZE: Final[int] = _DEFAULT_BATCH_SIZE

    # Centralized page size constants (reused by all namespaces)
    DEFAULT_PAGE_SIZE: Final[int] = _DEFAULT_PAGE_SIZE
    MAX_PAGE_SIZE: Final[int] = _MAX_PAGE_SIZE
    MIN_PAGE_SIZE: Final[int] = _MIN_PAGE_SIZE

    # Centralized retry constants (reused by all namespaces)
    DEFAULT_MAX_RETRY_ATTEMPTS: Final[int] = _DEFAULT_MAX_RETRY_ATTEMPTS

    # Centralized workers constants (reused by all namespaces)
    DEFAULT_WORKERS: Final[int] = _DEFAULT_WORKERS

    # Centralized pool size constants (reused by all namespaces)
    DEFAULT_POOL_SIZE: Final[int] = _DEFAULT_POOL_SIZE
    MAX_POOL_SIZE: Final[int] = _MAX_POOL_SIZE
    MIN_POOL_SIZE: Final[int] = _MIN_POOL_SIZE

    # Centralized name length constants (reused by all namespaces)
    MAX_NAME_LENGTH: Final[int] = _MAX_NAME_LENGTH
    MAX_OPERATION_NAME_LENGTH: Final[int] = _MAX_OPERATION_NAME_LENGTH

    # Centralized validation limit constants (reused by typings.py Field annotations)
    MAX_PORT_NUMBER: Final[int] = _MAX_PORT_NUMBER
    MIN_PORT_NUMBER: Final[int] = _MIN_PORT_NUMBER
    MAX_TIMEOUT_VALIDATION_SECONDS: Final[float] = _MAX_TIMEOUT_VALIDATION_SECONDS
    MAX_RETRY_COUNT_VALIDATION: Final[int] = _MAX_RETRY_COUNT_VALIDATION
    MAX_HOSTNAME_LENGTH_VALIDATION: Final[int] = _MAX_HOSTNAME_LENGTH_VALIDATION
    MAX_WORKERS_VALIDATION: Final[int] = _MAX_WORKERS_VALIDATION

    # Centralized generic constants (reused across multiple domains)
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
        """Network-related defaults."""

        MIN_PORT: Final[int] = _MIN_PORT_NUMBER
        MAX_PORT: Final[int] = _MAX_PORT_NUMBER
        DEFAULT_TIMEOUT: Final[int] = _DEFAULT_TIMEOUT_SECONDS
        DEFAULT_CONNECTION_POOL_SIZE: Final[int] = _DEFAULT_POOL_SIZE
        MAX_CONNECTION_POOL_SIZE: Final[int] = _MAX_POOL_SIZE
        MAX_HOSTNAME_LENGTH: Final[int] = _MAX_HOSTNAME_LENGTH_VALIDATION
        HTTP_STATUS_MIN: Final[int] = 100
        """Minimum valid HTTP status code."""
        HTTP_STATUS_MAX: Final[int] = 599
        """Maximum valid HTTP status code."""

    class Validation:
        """Input validation limits."""

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

        TIMEOUT: Final[int] = _DEFAULT_TIMEOUT_SECONDS
        PAGE_SIZE: Final[int] = 100
        TIMEOUT_SECONDS: Final[int] = _DEFAULT_TIMEOUT_SECONDS
        CACHE_TTL: Final[int] = 300
        DEFAULT_CACHE_TTL: Final[int] = CACHE_TTL
        # MAX_CACHE_SIZE removed - use c.Container.MAX_CACHE_SIZE instead
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
        # Pydantic ConfigDict ser_json_timedelta literal types
        SERIALIZATION_ISO8601: Final = "iso8601"
        """ISO8601 format for datetime serialization."""
        SERIALIZATION_FLOAT: Final = "float"
        """Float format for datetime serialization."""
        # Pydantic ConfigDict ser_json_bytes literal types
        SERIALIZATION_BASE64: Final = "base64"
        """Base64 format for bytes serialization."""
        SERIALIZATION_UTF8: Final = "utf8"
        """UTF-8 format for bytes serialization."""
        SERIALIZATION_HEX: Final = "hex"
        """Hex format for bytes serialization."""
        # DEFAULT_BATCH_SIZE removed - use c.DEFAULT_BATCH_SIZE instead
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
        # DEFAULT_ENABLE_METRICS removed - use c.Dispatcher.DEFAULT_ENABLE_METRICS instead
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

        BASE: Final[ConfigDict] = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",  # Use c.ModelConfig.EXTRA_FORBID in code
            ser_json_timedelta="iso8601",  # Use c.Utilities.SERIALIZATION_ISO8601 in code
            ser_json_bytes="base64",  # Use c.Utilities.SERIALIZATION_BASE64 in code
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
        # DEFAULT_BACKOFF_MULTIPLIER removed - use c.DEFAULT_BACKOFF_MULTIPLIER instead
        # DEFAULT_MAX_DELAY_SECONDS removed - use c.DEFAULT_MAX_DELAY_SECONDS instead
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
            """Circuit breaker states."""

            CLOSED = "closed"
            OPEN = "open"
            HALF_OPEN = "half_open"

        # Valid circuit breaker states derived from CircuitBreakerState StrEnum
        VALID_CIRCUIT_BREAKER_STATES: Final[frozenset[str]] = frozenset(
            member.value for member in CircuitBreakerState
        )

    class Security:
        """Security constants."""

        JWT_DEFAULT_ALGORITHM: Final[str] = "HS256"
        CREDENTIAL_BCRYPT_ROUNDS: Final[int] = 12

    class Logging:
        """Logging configuration."""

        # Log level constants - direct access for referencing
        DEBUG: Final[str] = "DEBUG"
        INFO: Final[str] = "INFO"
        WARNING: Final[str] = "WARNING"
        ERROR: Final[str] = "ERROR"
        CRITICAL: Final[str] = "CRITICAL"

        # Log levels tuple and set
        VALID_LEVELS: Final[tuple[str, ...]] = (
            DEBUG,
            INFO,
            WARNING,
            ERROR,
            CRITICAL,
        )
        VALID_LEVELS_SET: Final[frozenset[str]] = frozenset(VALID_LEVELS)

        # Default log level references
        DEFAULT_LEVEL: Final[str] = INFO
        DEFAULT_LEVEL_DEVELOPMENT: Final[str] = DEBUG
        DEFAULT_LEVEL_PRODUCTION: Final[str] = WARNING
        DEFAULT_LEVEL_TESTING: Final[str] = INFO

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

        class ContextOperation(StrEnum):
            """Context operation types enumeration (single source of truth)."""

            BIND = "bind"
            UNBIND = "unbind"
            CLEAR = "clear"
            GET = "get"

        # Type aliases derived from ContextOperation StrEnum (NÃO DUPLICA!)
        type ContextOperationGetLiteral = Literal[ContextOperation.GET]
        type ContextOperationModifyLiteral = Literal[
            ContextOperation.BIND,
            ContextOperation.UNBIND,
            ContextOperation.CLEAR,
        ]
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
            internal.invalid,
        ]

        # Registration status literal (references
        # FlextConstants.Cqrs.RegistrationStatus StrEnum members)
        type RegistrationStatusLiteral = Literal[
            FlextConstants.Cqrs.RegistrationStatus.ACTIVE,
            FlextConstants.Cqrs.RegistrationStatus.INACTIVE,
        ]

        # State literal - NOTE: This combines multiple StrEnum values
        # References FlextConstants.Domain.Status,
        # FlextConstants.Cqrs.CommonStatus, and
        # FlextConstants.Cqrs.HealthStatus members
        type StateLiteral = Literal[
            FlextConstants.Domain.Status.ACTIVE,
            FlextConstants.Domain.Status.INACTIVE,
            FlextConstants.Domain.Status.PENDING,
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

        # Action literal (references Cqrs.Action StrEnum members)
        # NÃO duplica strings - referencia o enum member!
        type ActionLiteral = Literal[
            FlextConstants.Cqrs.Action.GET,
            FlextConstants.Cqrs.Action.CREATE,
            FlextConstants.Cqrs.Action.UPDATE,
            FlextConstants.Cqrs.Action.DELETE,
            FlextConstants.Cqrs.Action.LIST,
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

        # FlextConstants.Cqrs.ProcessingStatus - references CommonStatus StrEnum members directly
        # ProcessingStatus uses CommonStatus values - NO duplicate enum!
        type ProcessingStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
            FlextConstants.Cqrs.CommonStatus.CANCELLED,
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

        # FlextConstants.Cqrs.CommonStatus - references CommonStatus StrEnum members
        type StatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
            FlextConstants.Cqrs.CommonStatus.COMPENSATING,
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

        # FlextConstants.Cqrs.NotificationStatus - references CommonStatus and SpecialStatus StrEnum members
        # NO duplicate enum - uses CommonStatus and SpecialStatus directly!
        type NotificationStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.SENT,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

        # FlextConstants.Cqrs.TokenStatus - references CommonStatus StrEnum members directly
        # NO duplicate enum - uses CommonStatus directly!
        type TokenStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

        # Reliability.CircuitBreakerState - references CircuitBreakerState StrEnum members
        type CircuitBreakerStateLiteral = Literal[
            FlextConstants.Reliability.CircuitBreakerState.CLOSED,
            FlextConstants.Reliability.CircuitBreakerState.OPEN,
            FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN,
        ]

        # FlextConstants.Cqrs.CircuitBreakerStatus - references CommonStatus and SpecialStatus StrEnum members
        # NO duplicate enum - uses CommonStatus and SpecialStatus directly!
        type CircuitBreakerStatusLiteral = Literal[
            FlextConstants.Cqrs.SpecialStatus.IDLE,
            FlextConstants.Cqrs.CommonStatus.RUNNING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

        # FlextConstants.Cqrs.BatchStatus - references CommonStatus and SpecialStatus StrEnum members
        # NO duplicate enum - uses CommonStatus and SpecialStatus directly!
        type BatchStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.PROCESSING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
        ]

        # FlextConstants.Cqrs.ExportStatus - references CommonStatus and SpecialStatus StrEnum members
        # NO duplicate enum - uses CommonStatus and SpecialStatus directly!
        type ExportStatusLiteral = Literal[
            FlextConstants.Cqrs.CommonStatus.PENDING,
            FlextConstants.Cqrs.SpecialStatus.PROCESSING,
            FlextConstants.Cqrs.CommonStatus.COMPLETED,
            FlextConstants.Cqrs.CommonStatus.FAILED,
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

        # Model config for domain classes
        DOMAIN_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(
            use_enum_values=True,
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            str_strip_whitespace=True,
            arbitrary_types_allowed=False,
            extra="forbid",
        )

        # ─────────────────────────────────────────────────────────────────
        # SUBSETS: Literal referenciando membros do StrEnum
        # ─────────────────────────────────────────────────────────────────
        # Use para aceitar apenas ALGUNS valores do enum em métodos
        # Isso NÃO duplica strings - referencia o enum member!

        type ActiveStates = Literal[
            FlextConstants.Domain.Status.ACTIVE,
            FlextConstants.Domain.Status.INACTIVE,
            FlextConstants.Domain.Status.PENDING,
        ]
        type TerminalStates = Literal[
            FlextConstants.Domain.Status.ARCHIVED,
            FlextConstants.Domain.Status.FAILED,
        ]

        # Order status literals for type-safe operations
        type OrderStatusLiteral = Literal[
            OrderStatus.PENDING,
            OrderStatus.CONFIRMED,
            OrderStatus.SHIPPED,
            OrderStatus.DELIVERED,
            OrderStatus.CANCELLED,
        ]

        # Active order statuses
        type ActiveOrderStatusLiteral = Literal[
            OrderStatus.PENDING,
            OrderStatus.CONFIRMED,
            OrderStatus.SHIPPED,
        ]

        # Terminal order statuses
        type TerminalOrderStatusLiteral = Literal[
            OrderStatus.DELIVERED,
            OrderStatus.CANCELLED,
        ]

        # Currency literal for type-safe monetary operations
        type CurrencyLiteral = Literal[
            Currency.USD,
            Currency.EUR,
            Currency.GBP,
            Currency.BRL,
        ]

        # ─────────────────────────────────────────────────────────────────
        # EXEMPLOS DE USO EM MÉTODOS
        # ─────────────────────────────────────────────────────────────────

        # 1. Aceitar qualquer valor do StrEnum:
        #    def get_by_status(self, status: Status) -> r[list[Entry]]: ...

        # 2. Aceitar apenas subset do StrEnum:
        #    def process_active(self, status: ActiveStates) -> r[bool]: ...
        #    def finalize(self, status: TerminalStates) -> r[bool]: ...

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

        class Status(StrEnum):
            """CQRS status enumeration."""

            RUNNING = "running"
            STOPPED = "stopped"
            FAILED = "failed"

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

        # CONSOLIDATED STATUS ENUM - Single Source of Truth (DRY)
        # All other status enums derive specialized Literals from this
        class CommonStatus(StrEnum):
            """Base status enumeration - single source of truth for all status types (FLEXT standard).

            All specialized status Literals (ProcessingStatusLiteral, NotificationStatusLiteral, etc.)
            use Literal type aliases derived from these values and SpecialStatus to prevent duplication.
            This follows DRY and SOLID principles without losing semantic meaning.
            """

            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"
            CANCELLED = "cancelled"
            COMPENSATING = "compensating"

        class MetricType(StrEnum):
            """Service metric types enumeration - single source of truth (FLEXT standard)."""

            COUNTER = "counter"
            GAUGE = "gauge"
            HISTOGRAM = "histogram"
            SUMMARY = "summary"

        type ServiceMetricTypeLiteral = Literal[
            MetricType.COUNTER,
            MetricType.GAUGE,
            MetricType.HISTOGRAM,
            MetricType.SUMMARY,
        ]

        class ServiceMetricCategory(StrEnum):
            """Service metric categories enumeration."""

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

        # Valid handler modes derived from HandlerType StrEnum (single source of truth)
        VALID_HANDLER_MODES: Final[frozenset[str]] = frozenset(
            member.value for member in HandlerType
        )

        class ProcessingMode(StrEnum):
            """CQRS processing modes enumeration."""

            BATCH = "batch"
            STREAM = "stream"
            PARALLEL = "parallel"
            SEQUENTIAL = "sequential"

        # SPECIALIZED STATUS LITERAL ALIASES (Python 3.13+ PEP 695)
        # These reference CommonStatus StrEnum members directly - NO string duplication!
        # Note: CommonStatus must be defined before these type aliases
        type ProcessingStatusLiteral = Literal[
            CommonStatus.PENDING,
            CommonStatus.RUNNING,
            CommonStatus.COMPLETED,
            CommonStatus.FAILED,
            CommonStatus.CANCELLED,
        ]
        type SagaStatusLiteral = Literal[
            CommonStatus.PENDING,
            CommonStatus.RUNNING,
            CommonStatus.COMPLETED,
            CommonStatus.FAILED,
            CommonStatus.COMPENSATING,
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
            """Special status values not in CommonStatus - single source of truth for unique status values."""

            SENT = "sent"  # Unique to notifications
            IDLE = "idle"  # Unique to circuit breaker
            PROCESSING = "processing"  # Similar to RUNNING but with specific semantics for batch/export

        class TokenType(StrEnum):
            """CQRS token types enumeration."""

            BEARER = "bearer"
            API_KEY = "api_key"
            JWT = "jwt"

        # More specialized status literals from CommonStatus and SpecialStatus
        # All Literals reference StrEnum members directly - NO string duplication!
        type NotificationStatusLiteral = Literal[
            CommonStatus.PENDING,
            SpecialStatus.SENT,
            CommonStatus.FAILED,
        ]
        type TokenStatusLiteral = Literal[
            CommonStatus.PENDING,
            CommonStatus.RUNNING,
            CommonStatus.COMPLETED,
            CommonStatus.FAILED,
        ]
        type CircuitBreakerStatusLiteral = Literal[
            SpecialStatus.IDLE,
            CommonStatus.RUNNING,
            CommonStatus.COMPLETED,
            CommonStatus.FAILED,
        ]
        type BatchStatusLiteral = Literal[
            CommonStatus.PENDING,
            SpecialStatus.PROCESSING,
            CommonStatus.COMPLETED,
            CommonStatus.FAILED,
        ]
        type ExportStatusLiteral = Literal[
            CommonStatus.PENDING,
            SpecialStatus.PROCESSING,
            CommonStatus.COMPLETED,
            CommonStatus.FAILED,
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
        # COMMAND_PROCESSING_FAILED removed - use c.Errors.COMMAND_PROCESSING_FAILED instead

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
        METADATA_FIELDS: Final[AbstractSet[str]] = frozenset({
            "user_id",
            "correlation_id",
            "request_id",
            "session_id",
            "tenant_id",
        })
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
        STATE_INACTIVE: Final[str] = "inactive"
        # STATE_* constants removed - use c.Cqrs.CommonStatus StrEnum instead
        # STATE_PENDING -> c.Cqrs.CommonStatus.PENDING
        # STATE_COMPLETED -> c.Cqrs.CommonStatus.COMPLETED
        # STATE_FAILED -> c.Cqrs.CommonStatus.FAILED
        # STATE_RUNNING -> c.Cqrs.CommonStatus.RUNNING
        # STATE_COMPENSATING -> c.Cqrs.CommonStatus.COMPENSATING
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
        # Validation strings removed - use c.Cqrs.ValidationLevel StrEnum instead
        # VALIDATION_BASIC, VALIDATION_STRICT, VALIDATION_CUSTOM removed to avoid duplication
        DEFAULT_JSON_INDENT: Final[int] = 2
        # DEFAULT_ENCODING removed - use c.Utilities.DEFAULT_ENCODING instead
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

    # =========================================================================
    # NAMESPACE ACCESS
    # =========================================================================
    # All constants are accessed via namespaces (e.g., FlextConstants.Errors.*)
    # No aliases - use namespaces directly following FLEXT architecture patterns
    # Example: FlextConstants.Errors.VALIDATION_ERROR (not FlextConstants.VALIDATION_ERROR)


c = FlextConstants

__all__ = ["FlextConstants", "c"]
