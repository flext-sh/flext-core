"""FlextConstants - Foundation constants for FLEXT ecosystem.

This module provides centralized, immutable constants used throughout flext-core
and all dependent FLEXT applications. Serves as single source of truth for all
configuration, validation, error codes, and system parameters across 32+ projects.

Architecture: Layer 0 (Pure Constants - No Layer 1+ Imports)
===========================================================
Provides immutable constants organized in hierarchical namespace classes.
All constants use typing.Final for immutability and serve as the complete
constant registry that other FLEXT components depend on.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import ConfigDict


class FlextConstants:
    """Foundation constants for the FLEXT ecosystem.

    Architecture: Layer 0 (Pure Constants - No Layer 1+ Imports)
    ============================================================
    Provides immutable constants organized in namespaces for configuration,
    validation, error handling, and system defaults. All constants use
    typing.Final for immutability and serve as the single source of truth
    throughout the FLEXT ecosystem.

    **Structural Typing and Protocol Compliance**:
    This class satisfies FlextProtocols.Constants through structural typing
    (duck typing) via the following protocol-compliant interface:
    - 40+ nested namespace classes providing organized constant access
    - typing.Final type annotations for immutability guarantees
    - Hierarchical access patterns (FlextConstants.Namespace.CONSTANT)
    - Compile-time constant verification via strict type checking
    - Runtime constant validation through __getitem__

    **Core Features**:
    1. **Immutable Constants**: All constants wrapped with typing.Final
    2. **Namespace Organization**: 30+ nested classes for logical grouping
    3. **Type Safety**: Complete type annotations (no Any, no implicit types)
    4. **Hierarchical Access**: Multi-level namespace access patterns
    5. **Alternative Access**: __getitem__ for dynamic constant retrieval
    6. **Ecosystem Single Source of Truth**: Centralized configuration
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
    - **FlextWeb**: HTTP status codes, methods, content types, headers
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

    **Integration with FlextProtocols**:
    This module provides the complete constant registry that other FLEXT
    components depend on. By centralizing all constants here:
    - Error codes are consistent across all 32+ projects
    - Configuration defaults are standardized
    - Type-safe literals support strict type checking
    - All components reference single source of truth
    - Reduces configuration drift and inconsistencies

    **Constants Organization Standards (FLEXT Standardization Plan)**:
    All FLEXT projects MUST follow these patterns for constants organization:

    1. **Constants Organization**:
       - ALL constants MUST be inside the main constants class (no module-level constants)
       - Use nested classes for logical grouping (e.g., class Errors:)
       - Layer 0 purity: Only constants, no functions or behavior

    2. **Inheritance Pattern**:
       - All domain constants MUST extend FlextConstants directly
       - Use composition for domain relationships: Reference other domain constants
       - Example: class FlextLdapConstants(FlextConstants): with LdifConstants = FlextLdifConstants

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
        ...     return FlextResult[str].fail(FlextConstants.Errors.VALIDATION_ERROR)

    4. Dynamic constant path resolution:
        >>> error_path = "Errors.VALIDATION_ERROR"
        >>> error_code = FlextConstants[error_path]

    5. Type-safe literal unions:
        >>> handler_types: list[FlextConstants.HandlerType] = [
        ...     FlextConstants.Cqrs.HandlerTypeLiteral.COMMAND,
        ...     FlextConstants.Cqrs.HandlerTypeLiteral.QUERY,
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
        >>> from flext_core import FlextResult
        >>>
        >>> def validate_input(name: str) -> FlextResult[str]:
        ...     if len(name) < FlextConstants.Validation.MIN_NAME_LENGTH:
        ...         error = FlextConstants.Errors.VALIDATION_ERROR
        ...         return FlextResult[str].fail(error)
        ...
        ...     if len(name) > FlextConstants.Validation.MAX_NAME_LENGTH:
        ...         error = FlextConstants.Errors.VALIDATION_ERROR
        ...         return FlextResult[str].fail(error)
        ...
        ...     return FlextResult[str].ok(name)
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
    ZERO: Final[int] = 0
    INITIAL_TIME: Final[float] = 0.0

    def __getitem__(self, key: str) -> object:
        """Dynamic access: FlextConstants['Errors.VALIDATION_ERROR'].

        Args:
            key: Dot-separated path (e.g., 'Errors.VALIDATION_ERROR')

        Returns:
            The constant value

        Raises:
            AttributeError: If path not found

        """
        parts = key.split(".")
        value: object = self
        try:
            for part in parts:
                value = getattr(value, part)
            return value
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
            SHARED_CONTAINERS: Final[dict[str, dict[str, str | int]]] = {
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

        FAILURE_LEVEL_DEFAULT: Final[str] = FailureLevel.PERMISSIVE

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
                """Validate state value."""
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

    class Cqrs:
        """CQRS pattern constants."""

        class HandlerType(StrEnum):
            """CQRS handler types enumeration."""

            COMMAND = "command"
            QUERY = "query"
            EVENT = "event"
            OPERATION = "operation"
            SAGA = "saga"

        # Type literals for message type discrimination
        CommandMessageTypeLiteral = Literal["command"]
        QueryMessageTypeLiteral = Literal["query"]
        EventMessageTypeLiteral = Literal["event"]
        HandlerModeLiteral = Literal["command", "query", "event", "operation", "saga"]
        HandlerTypeLiteral = Literal["command", "query", "event", "operation", "saga"]
        ServiceMetricTypeLiteral = Literal["counter", "gauge", "histogram", "summary"]

        DEFAULT_HANDLER_TYPE: HandlerType = HandlerType.COMMAND
        COMMAND_HANDLER_TYPE: HandlerType = HandlerType.COMMAND
        QUERY_HANDLER_TYPE: HandlerType = HandlerType.QUERY
        EVENT_HANDLER_TYPE: HandlerType = HandlerType.EVENT
        OPERATION_HANDLER_TYPE: HandlerType = HandlerType.OPERATION
        SAGA_HANDLER_TYPE: HandlerType = HandlerType.SAGA

        # Valid handler modes as frozenset for validation without enum iteration
        VALID_HANDLER_MODES: frozenset[str] = frozenset({
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
        CORRELATION_ID_PREFIX: Final[str] = "flext-"
        CORRELATION_ID_LENGTH: Final[int] = 12
        DEFAULT_CONTEXT_TIMEOUT: Final[int] = 30
        MAX_CONTEXT_DEPTH: Final[int] = 10
        MAX_CONTEXT_SIZE: Final[int] = 1000
        MILLISECONDS_PER_SECOND: Final[int] = 1000
        EXPORT_FORMAT_JSON: Final[str] = "json"
        EXPORT_FORMAT_DICT: Final[str] = "dict"
        METADATA_FIELDS: Final[frozenset[str]] = frozenset({
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

        HANDLER_MODE_COMMAND = "command"
        HANDLER_MODE_QUERY = "query"
        VALID_HANDLER_MODES = (HANDLER_MODE_COMMAND, HANDLER_MODE_QUERY)
        DEFAULT_HANDLER_MODE = HANDLER_MODE_COMMAND
        DEFAULT_AUTO_CONTEXT = True
        DEFAULT_ENABLE_LOGGING = True
        DEFAULT_ENABLE_METRICS = True
        DEFAULT_TIMEOUT_SECONDS = 30
        MIN_TIMEOUT_SECONDS = 1
        MAX_TIMEOUT_SECONDS = 600
        MIN_REGISTRATION_ID_LENGTH = 1
        MIN_REQUEST_ID_LENGTH = 1
        ERROR_INVALID_HANDLER_MODE = "handler_mode must be 'command' or 'query'"
        ERROR_HANDLER_REQUIRED = "handler cannot be None"
        ERROR_MESSAGE_REQUIRED = "message cannot be None"
        ERROR_POSITIVE_TIMEOUT = "timeout must be positive"
        ERROR_INVALID_REGISTRATION_ID = "registration_id must be non-empty string"
        ERROR_INVALID_REQUEST_ID = "request_id must be non-empty string"
        REGISTRATION_STATUS_ACTIVE = "active"
        REGISTRATION_STATUS_INACTIVE = "inactive"
        REGISTRATION_STATUS_ERROR = "error"
        VALID_REGISTRATION_STATUSES = (
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

        FIELD_ID = "unique_id"
        FIELD_NAME = "name"
        FIELD_TYPE = "type"
        FIELD_STATUS = "status"
        FIELD_DATA = "data"
        FIELD_CONFIG = "config"
        FIELD_STATE = "state"
        FIELD_CREATED_AT = "created_at"
        FIELD_UPDATED_AT = "updated_at"
        FIELD_VALIDATED = "validated"
        FIELD_CLASS = "class"
        FIELD_MODULE = "module"
        FIELD_REGISTERED = "registered"
        FIELD_EVENT_NAME = "event_name"
        FIELD_AGGREGATE_ID = "aggregate_id"
        FIELD_OCCURRED_AT = "occurred_at"
        STATE_ACTIVE = "active"
        STATE_INACTIVE = "inactive"
        STATE_PENDING = "pending"
        STATE_COMPLETED = "completed"
        STATE_FAILED = "failed"
        STATE_RUNNING = "running"
        STATE_COMPENSATING = "compensating"
        STATE_SENT = "sent"
        STATE_IDLE = "idle"
        STATE_HEALTHY = "healthy"
        STATE_DEGRADED = "degraded"
        STATE_UNHEALTHY = "unhealthy"
        STATUS_PASS = "PASS"
        STATUS_FAIL = "FAIL"
        STATUS_NO_TARGET = "NO_TARGET"
        STATUS_SKIP = "SKIP"
        STATUS_UNKNOWN = "UNKNOWN"
        IDENTIFIER_UNKNOWN = "unknown"
        IDENTIFIER_DEFAULT = "default"
        IDENTIFIER_ANONYMOUS = "anonymous"
        IDENTIFIER_GUEST = "guest"
        IDENTIFIER_SYSTEM = "system"
        METHOD_HANDLE = "handle"
        METHOD_PROCESS = "process"
        METHOD_EXECUTE = "execute"
        METHOD_PROCESS_COMMAND = "process_command"
        AUTH_BEARER = "bearer"
        AUTH_API_KEY = "api_key"
        AUTH_JWT = "jwt"
        HANDLER_COMMAND = "command"
        HANDLER_QUERY = "query"
        METHOD_VALIDATE = "validate"
        VALIDATION_BASIC = "basic"
        VALIDATION_STRICT = "strict"
        VALIDATION_CUSTOM = "custom"
        DEFAULT_JSON_INDENT = 2
        DEFAULT_ENCODING = "utf-8"
        DEFAULT_SORT_KEYS = False
        DEFAULT_ENSURE_ASCII = False
        BOOL_TRUE_STRINGS: Final[frozenset[str]] = frozenset({
            "true",
            "1",
            "yes",
            "on",
            "enabled",
        })
        BOOL_FALSE_STRINGS: Final[frozenset[str]] = frozenset({
            "false",
            "0",
            "no",
            "off",
            "disabled",
        })
        STRING_TRUE = "true"
        STRING_FALSE = "false"
        DEFAULT_USE_UTC = True
        DEFAULT_AUTO_UPDATE = True
        MAX_OPERATION_NAME_LENGTH = 100
        MAX_STATE_VALUE_LENGTH = 50
        MAX_FIELD_NAME_LENGTH = 50
        MIN_FIELD_NAME_LENGTH = 1
        ERROR_EMPTY_OPERATION = "Operation name cannot be empty"
        ERROR_EMPTY_STATE = "State value cannot be empty"
        ERROR_EMPTY_FIELD_NAME = "Field name cannot be empty"
        ERROR_INVALID_ENCODING = "Invalid character encoding"
        ERROR_MISSING_TIMESTAMP_FIELDS = "Required timestamp fields missing"
        ERROR_INVALID_LOG_LEVEL = "Invalid log level"

    class FlextWeb:
        """HTTP protocol constants."""

        HTTP_STATUS_MIN: Final[int] = 100
        HTTP_STATUS_MAX: Final[int] = 599

    class Processing:
        """Processing pipeline constants."""

        DEFAULT_MAX_WORKERS: Final[int] = 4
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        MAX_BATCH_SIZE: Final[int] = 10000


__all__ = ["FlextConstants"]
