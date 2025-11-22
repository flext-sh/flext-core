"""Foundation constants for the FLEXT ecosystem.

This module provides centralized constants used throughout the flext-core framework
and dependent applications. All constants are immutable and serve as the single
source of truth for configuration defaults, validation limits, and system parameters.

All constants use typing.Final for immutability and are safe to import from any module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from enum import StrEnum
from typing import (
    Final,
    Literal,
    TypeAlias,
)

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

    **Literal Type Definitions** (Type-Safe Annotations):
    - Status, CircuitBreakerState, HandlerType, HandlerMode
    - Compression, ErrorCategory, ErrorSeverity
    - ServiceType, ServiceLifecycleState, ServiceProtocol
    - ContextScope, ContextExportFormat, LoggingLevel
    - ProcessingOutputFormat, ProcessingSerializationFormat, ProcessingCompressionFormat
    - WorkflowProcessingStatus, WorkflowProcessingMode, WorkflowValidationLevel
    - CqrsMode, WorkspaceStatus, ProjectType, ProjectStatus

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

    **Integration with FLEXT Ecosystem**:
    - **FlextResult**: Uses error codes from Errors namespace
    - **FlextConfig**: References Config and Platform constants
    - **FlextLogger**: Uses Logging constants for configuration
    - **FlextBus**: Uses Cqrs constants for handler types
    - **FlextContext**: Uses Context constants for scope management
    - **FlextContainer**: Uses Container constants for lifecycle
    - **Domain Services**: Reference validation and business error codes
    - **All 32+ projects**: Depend on this centralized constant registry

    **Thread Safety**:
    All constants are immutable (typing.Final). Access is thread-safe as
    there is no mutable state in this module.

    **Performance Characteristics**:
    - O(1) constant access via namespace attributes
    - O(1) constant access via __getitem__ (dict-like lookup)
    - No runtime compilation or validation overhead
    - Constants are resolved at import time

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
        >>> from flext_core import FlextConstants, FlextResult
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
    """

    """Core identifiers."""
    NAME: Final[str] = "FLEXT"

    # Semantic zero and initial values
    ZERO: Final[int] = 0
    INITIAL_TIME: Final[float] = 0.0

    def __getitem__(self, key: str) -> object:
        """Direct value access: FlextConstants['Errors.VALIDATION_ERROR'].

        Enables simplified constant access by string key path.

        Args:
            key: Dot-separated path to constant (e.g., 'Errors.VALIDATION_ERROR')

        Returns:
            The constant value at the specified path

        Raises:
            AttributeError: If the key path doesn't exist

        Example:
            >>> FlextConstants.Errors.VALIDATION_ERROR
            'VALIDATION_ERROR'
            >>> FlextConstants.Settings.DEFAULT_TIMEOUT
            30

        """
        # Parse nested key like "Errors.VALIDATION_ERROR"
        parts = key.split(".")
        value: object = self

        try:
            for part in parts:
                value = getattr(value, part)
            return value
        except AttributeError as e:
            msg = f"Constant path '{key}' not found in FlextConstants"
            raise AttributeError(msg) from e

    class Network:
        """Network-related defaults and limits."""

        MIN_PORT: Final[int] = 1
        MAX_PORT: Final[int] = 65535
        DEFAULT_TIMEOUT: Final[int] = 30
        DEFAULT_CONNECTION_POOL_SIZE: Final[int] = 10
        MAX_CONNECTION_POOL_SIZE: Final[int] = 100
        MAX_HOSTNAME_LENGTH: Final[int] = 253  # RFC 1035: max 253 characters

    class Validation:
        """Input validation limits and patterns."""

        MIN_NAME_LENGTH: Final[int] = 2  # Usage count: 1
        MAX_NAME_LENGTH: Final[int] = 100  # Maximum name length for validation
        MAX_EMAIL_LENGTH: Final[int] = 254  # Maximum email length (RFC 5321)
        EMAIL_PARTS_COUNT: Final[int] = 2  # Expected parts when splitting email by @
        LEVEL_PREFIX_PARTS_COUNT: Final[int] = (
            4  # Expected parts when splitting _level_<level>_<key>
        )

        # Phone number validation
        MIN_PHONE_DIGITS: Final[int] = 10  # Minimum phone number length
        MAX_PHONE_DIGITS: Final[int] = (
            20  # Maximum phone number length (international standard)
        )
        MIN_USERNAME_LENGTH: Final[int] = 3  # Minimum username length for validation
        MAX_AGE: Final[int] = 150  # Maximum valid age for persons
        MIN_AGE: Final[int] = 0  # Minimum valid age for persons
        PREVIEW_LENGTH: Final[int] = 50  # Maximum length for string previews/truncation
        VALIDATION_TIMEOUT_MS: Final[int] = (
            100  # Maximum validation time in milliseconds
        )
        MAX_UNCOMMITTED_EVENTS: Final[int] = (
            100  # Maximum uncommitted domain events per aggregate
        )
        DISCOUNT_THRESHOLD: Final[int] = 100  # Minimum total for discount eligibility
        DISCOUNT_RATE: Final[float] = 0.05  # 5% discount rate
        SLOW_OPERATION_THRESHOLD: Final[float] = (
            0.1  # Threshold for slow operation warning (seconds)
        )
        RESOURCE_LIMIT_MIN: Final[int] = 50  # Minimum resource limit
        FILTER_THRESHOLD: Final[int] = 5  # General filter threshold for comparisons
        RETRY_COUNT_MAX: Final[int] = 3  # Maximum retry attempts
        MAX_WORKERS_LIMIT: Final[int] = (
            100  # Maximum reasonable worker count for validation
        )

    class Errors:
        """Error codes for exception categorization."""

        # Core error codes (actively used)
        VALIDATION_ERROR: Final[str] = "VALIDATION_ERROR"  # Usage count: 28
        TYPE_ERROR: Final[str] = "TYPE_ERROR"  # Usage count: 4
        ATTRIBUTE_ERROR: Final[str] = "ATTRIBUTE_ERROR"  # Usage count: 1
        CONFIG_ERROR: Final[str] = "CONFIG_ERROR"  # Usage count: 1
        GENERIC_ERROR: Final[str] = "GENERIC_ERROR"  # Usage count: 3
        COMMAND_PROCESSING_FAILED: Final[str] = "COMMAND_PROCESSING_FAILED"
        UNKNOWN_ERROR: Final[str] = "UNKNOWN_ERROR"  # Usage count: 1

        # Applicative pattern error messages
        FIRST_ARG_FAILED_MSG: Final[str] = "First argument failed"
        SECOND_ARG_FAILED_MSG: Final[str] = "Second argument failed"

        # Serialization errors (reserved for FlextResult operations)
        SERIALIZATION_ERROR: Final[str] = (
            "SERIALIZATION_ERROR"  # Reserved for 1.0.0 API
        )
        MAP_ERROR: Final[str] = "MAP_ERROR"  # Reserved for FlextResult.map operations
        BIND_ERROR: Final[str] = "BIND_ERROR"  # Reserved for FlextResult.flat_map
        CHAIN_ERROR: Final[str] = "CHAIN_ERROR"  # Reserved for FlextResult chaining
        UNWRAP_ERROR: Final[str] = "UNWRAP_ERROR"  # Reserved for FlextResult.unwrap

        # Business logic errors (reserved for domain services)
        OPERATION_ERROR: Final[str] = (
            "OPERATION_ERROR"  # Reserved for service operations
        )
        SERVICE_ERROR: Final[str] = "SERVICE_ERROR"  # Reserved for service-level errors
        BUSINESS_RULE_VIOLATION: Final[str] = (
            "BUSINESS_RULE_VIOLATION"  # Reserved for DDD
        )
        BUSINESS_RULE_ERROR: Final[str] = (
            "BUSINESS_RULE_ERROR"  # Reserved for validation
        )

        # Resource errors (reserved for CRUD operations)
        NOT_FOUND_ERROR: Final[str] = "NOT_FOUND_ERROR"  # Reserved for queries
        NOT_FOUND: Final[str] = "NOT_FOUND"  # Reserved for entity lookup
        RESOURCE_NOT_FOUND: Final[str] = "RESOURCE_NOT_FOUND"  # Reserved for REST APIs
        ALREADY_EXISTS: Final[str] = "ALREADY_EXISTS"  # Reserved for create operations

        # CQRS errors (reserved for command/query bus)
        COMMAND_BUS_ERROR: Final[str] = "COMMAND_BUS_ERROR"  # Reserved for FlextBus
        COMMAND_HANDLER_NOT_FOUND: Final[str] = (
            "COMMAND_HANDLER_NOT_FOUND"  # Reserved for dispatch
        )
        DOMAIN_EVENT_ERROR: Final[str] = (
            "DOMAIN_EVENT_ERROR"  # Reserved for domain events
        )

        # Infrastructure errors (reserved for technical failures)
        TIMEOUT_ERROR: Final[str] = "TIMEOUT_ERROR"  # Reserved for operations
        PROCESSING_ERROR: Final[str] = (
            "PROCESSING_ERROR"  # Reserved for batch processing
        )
        CONNECTION_ERROR: Final[str] = "CONNECTION_ERROR"  # Reserved for network/DB
        CONFIGURATION_ERROR: Final[str] = (
            "CONFIGURATION_ERROR"  # Reserved for FlextConfig
        )
        EXTERNAL_SERVICE_ERROR: Final[str] = (
            "EXTERNAL_SERVICE_ERROR"  # Reserved for integrations
        )

        # Security errors (reserved for authentication/authorization)
        PERMISSION_ERROR: Final[str] = "PERMISSION_ERROR"  # Reserved for access control
        AUTHENTICATION_ERROR: Final[str] = "AUTHENTICATION_ERROR"  # Reserved for auth
        AUTHORIZATION_ERROR: Final[str] = (
            "AUTHORIZATION_ERROR"  # Reserved for authorization failures
        )

        # System errors (reserved for critical failures)
        EXCEPTION_ERROR: Final[str] = (
            "EXCEPTION_ERROR"  # Reserved for exception wrapping
        )
        CRITICAL_ERROR: Final[str] = (
            "CRITICAL_ERROR"  # Reserved for system failures  # Usage count: 1
        )
        NONEXISTENT_ERROR: Final[str] = "NONEXISTENT_ERROR"

    class Test:
        """Test constants for development and testing."""

        # Test authentication credentials
        DEFAULT_PASSWORD: Final[str] = "password123"
        DEFAULT_USERNAME: Final[str] = "testuser"
        NONEXISTENT_USERNAME: Final[str] = "nonexistent"

    class Exceptions:
        """Exception handling configuration and failure levels."""

        class FailureLevel(StrEnum):
            """Exception handling strictness levels for hierarchical exception system.

            - STRICT: Specific exceptions required, broad catching forbidden
            - WARN: Allow broad catching but log warnings when used
            - PERMISSIVE: Allow broad catching silently
            """

            STRICT = "strict"
            WARN = "warn"
            PERMISSIVE = "permissive"

        # Default failure level (Layer 0 constant)
        FAILURE_LEVEL_DEFAULT: Final[str] = FailureLevel.PERMISSIVE

    class Messages:
        """User-facing message templates."""

        TYPE_MISMATCH: Final[str] = "Type mismatch"
        VALIDATION_FAILED: Final[str] = "Validation failed"
        REDACTED_SECRET: Final[str] = "***REDACTED***"

    class Defaults:
        """Default values for common operations."""

        TIMEOUT: Final[int] = 30  # Usage count: 18
        PAGE_SIZE: Final[int] = 100  # Usage count: 2
        TIMEOUT_SECONDS: Final[int] = 30  # Default timeout for operations

        # Cache defaults
        CACHE_TTL: Final[int] = 300  # Cache time-to-live in seconds (5 minutes)
        DEFAULT_CACHE_TTL: Final[int] = CACHE_TTL
        MAX_CACHE_SIZE: Final[int] = 100  # Maximum cache entries
        DEFAULT_MAX_CACHE_SIZE: Final[int] = MAX_CACHE_SIZE

        # Message and operation defaults
        MAX_MESSAGE_LENGTH: Final[int] = 100  # Maximum message length for truncation
        DEFAULT_MIDDLEWARE_ORDER: Final[int] = 0  # Default middleware execution order
        OPERATION_TIMEOUT_SECONDS: Final[int] = 30  # Default operation timeout

    class Utilities:
        """Utility constants reused by helper modules."""

        DEFAULT_ENCODING: Final[str] = "utf-8"
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        MAX_TIMEOUT_SECONDS: Final[int] = 3600

        # UUID and identifier generation constants
        LONG_UUID_LENGTH: Final[int] = 12  # Length of UUID suffix for ID generation
        SHORT_UUID_LENGTH: Final[int] = (
            8  # Length of short UUID suffix for correlation IDs
        )
        VERSION_MODULO: Final[int] = 100  # Modulo value for version computation
        CONTROL_CHARS_PATTERN: Final[str] = (
            r"[\x00-\x1F\x7F]"  # Regex pattern for control characters
        )

        # Cache attribute names for mixin operations
        CACHE_ATTRIBUTE_NAMES: Final[tuple[str, ...]] = (
            "_cache",
            "_ttl",
            "_cached_at",
            "_cached_value",
        )

    class Settings:
        """Configuration defaults and limits (clear naming to avoid Pydantic v1 Config confusion)."""

        MAX_WORKERS_THRESHOLD: Final[int] = 50

        # Feature flags
        DEFAULT_ENABLE_CACHING: Final[bool] = True
        DEFAULT_ENABLE_METRICS: Final[bool] = False
        DEFAULT_ENABLE_TRACING: Final[bool] = False

        class LogLevel(StrEnum):
            """Standard log levels for centralized logging configuration."""

            DEBUG = "DEBUG"
            INFO = "INFO"
            WARNING = "WARNING"
            ERROR = "ERROR"
            CRITICAL = "CRITICAL"

        # Timeout defaults
        DEFAULT_TIMEOUT: Final[int] = 30

        class Environment(StrEnum):
            """Environment types for configuration."""

            DEVELOPMENT = "development"
            STAGING = "staging"
            PRODUCTION = "production"
            TESTING = "testing"
            LOCAL = "local"

    class ModelConfig:
        """Pydantic model configuration defaults for FlextModels.

        Centralized source of truth for all Pydantic ConfigDict settings
        used across FlextModels nested classes. Provides a complete BASE dict
        that can be unpacked with ** operator, allowing classes to override
        only what they need.

        Usage in FlextModels:
            model_config = ConfigDict(
                **FlextConstants.ModelConfig.BASE,
                frozen=True,  # Override only what's different
                json_schema_extra={...},
            )
        """

        # Complete base configuration - unpack with **
        # Note: frozen is excluded so classes can set it as needed
        # Using ConfigDict type directly for type-safe unpacking
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
        """Platform-specific constants for HTTP, database, and file types."""

        # Environment variable configuration
        ENV_PREFIX: Final[str] = "FLEXT_"
        ENV_FILE_DEFAULT: Final[str] = ".env"
        ENV_NESTED_DELIMITER: Final[str] = "__"

        # Application constants
        FLEXT_API_PORT: Final[int] = 8000
        DEFAULT_HOST: Final[str] = "localhost"
        DEFAULT_HTTP_PORT: Final[int] = 80

        # Common MIME types
        MIME_TYPE_JSON: Final[str] = "application/json"

        # Validation patterns
        PATTERN_EMAIL: Final[str] = (
            r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}"
            r"[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
        )
        PATTERN_URL: Final[str] = (
            r"^https?://"
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
            r"localhost|"
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
            r"(?::\d+)?"
            r"(?:/?|[/?]\S+)$"
        )
        PATTERN_PHONE_NUMBER: Final[str] = r"^\+?[\d\s\-\(\)]{10,20}$"
        PATTERN_UUID: Final[str] = (
            r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
            r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
        )
        PATTERN_PATH: Final[str] = r'^[^<>"|?*\x00-\x1F]+$'

        # File extensions
        EXT_PYTHON: Final[str] = ".py"
        EXT_YAML: Final[str] = ".yaml"
        EXT_JSON: Final[str] = ".json"
        EXT_TOML: Final[str] = ".toml"
        EXT_XML: Final[str] = ".xml"
        EXT_TXT: Final[str] = ".txt"
        EXT_MD: Final[str] = ".md"

        # Directory names
        DIR_CONFIG: Final[str] = "config"
        DIR_PLUGINS: Final[str] = "plugins"
        DIR_LOGS: Final[str] = "logs"
        DIR_DATA: Final[str] = "data"
        DIR_TEMP: Final[str] = "temp"

    class Performance:
        """Performance thresholds and operational limits."""

        # Database connection defaults
        DEFAULT_DB_POOL_SIZE: Final[int] = 10
        MIN_DB_POOL_SIZE: Final[int] = 1
        MAX_DB_POOL_SIZE: Final[int] = 100

        # Operation limits
        MAX_RETRY_ATTEMPTS_LIMIT: Final[int] = 10
        DEFAULT_TIMEOUT_LIMIT: Final[int] = 300
        MIN_CURRENT_STEP: Final[int] = 0

        # Timing and delays
        DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
        DEFAULT_MAX_DELAY_SECONDS: Final[float] = 60.0
        DEFAULT_INITIAL_DELAY_SECONDS: Final[float] = 1.0

        # Batch processing constants
        class BatchProcessing:
            """Batch processing configuration constants."""

            DEFAULT_SIZE: Final[int] = 1000
            MAX_ITEMS: Final[int] = 10000
            MAX_VALIDATION_SIZE: Final[int] = 1000

        # Alias for backward compatibility
        MAX_BATCH_SIZE: Final[int] = 10000

        # Pagination and processing
        DEFAULT_TIME_RANGE_SECONDS: Final[int] = 3600
        DEFAULT_TTL_SECONDS: Final[int] = 3600

        # Field validation constraints
        DEFAULT_VERSION: Final[int] = 1
        MIN_VERSION: Final[int] = 1

        # Pagination - defaults for page processing
        DEFAULT_PAGE_SIZE: Final[int] = 10  # Default page size for processing

        # Memory and resource thresholds
        HIGH_MEMORY_THRESHOLD_BYTES: Final[int] = 1073741824  # 1 GB in bytes

        # Operation constraints
        MAX_TIMEOUT_SECONDS: Final[int] = 600
        MAX_BATCH_OPERATIONS: Final[int] = 1000
        MAX_OPERATION_NAME_LENGTH: Final[int] = 100

        # Data structure constants
        EXPECTED_TUPLE_LENGTH: Final[int] = 2
        DEFAULT_EMPTY_STRING: Final[str] = ""

        # Module-specific performance thresholds
        CLI_PERFORMANCE_CRITICAL_MS: Final[float] = 10000.0

    class Reliability:
        """Reliability thresholds for retry and circuit breaker logic."""

        MAX_RETRY_ATTEMPTS: Final[int] = 3  # Usage count: 1
        DEFAULT_MAX_RETRIES: Final[int] = 3  # Usage count: 1 (referenced in models.py)
        DEFAULT_RETRY_DELAY_SECONDS: Final[int] = 1  # Default delay between retries
        RETRY_BACKOFF_BASE: Final[float] = (
            2.0  # Base multiplier for exponential backoff
        )
        RETRY_BACKOFF_MAX: Final[float] = (
            60.0  # Maximum delay between retries in seconds
        )
        RETRY_COUNT_MIN: Final[int] = 1  # Minimum retry attempts allowed
        DEFAULT_BACKOFF_STRATEGY: Final[str] = "exponential"
        DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
        DEFAULT_RECOVERY_TIMEOUT: Final[int] = 60
        DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0

        # Rate limiting constants
        DEFAULT_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
        DEFAULT_RATE_LIMIT_MAX_REQUESTS: Final[int] = 100

        # Circuit breaker constants
        DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
        DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[int] = 60  # seconds
        DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD: Final[int] = 3  # successes to close

        class CircuitBreakerState(StrEnum):
            """Circuit breaker state machine states.

            Modern enum-based implementation for type-safe state management.
            Replaces string literals with proper enum for validation and IDE support.
            """

            CLOSED = "closed"  # Normal operation - requests allowed
            OPEN = "open"  # Failing - requests blocked
            HALF_OPEN = "half_open"  # Testing recovery - limited requests

            @classmethod
            def validate(cls, value: str) -> bool:
                """Validate if value is a valid circuit breaker state.

                Args:
                    value: State value to validate

                Returns:
                    bool: True if valid state, False otherwise

                """
                return value in cls.__members__.values()

    class Security:
        """Security and authentication constants."""

        JWT_DEFAULT_ALGORITHM: Final[str] = "HS256"
        CREDENTIAL_BCRYPT_ROUNDS: Final[int] = 12

    class Logging:
        """Logging configuration constants."""

        # Log level constants
        DEBUG: Final[str] = "DEBUG"
        INFO: Final[str] = "INFO"
        WARNING: Final[str] = "WARNING"
        ERROR: Final[str] = "ERROR"
        CRITICAL: Final[str] = "CRITICAL"

        # Log level defaults (use Config.LogLevel StrEnum for type-safe access)
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

        # Output format defaults
        JSON_OUTPUT_DEFAULT: Final[bool] = False
        DEFAULT_FORMAT: Final[str] = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        STRUCTURED_OUTPUT: Final[bool] = True

        # Source code tracking
        INCLUDE_SOURCE: Final[bool] = True
        VERBOSITY: Final[str] = "compact"  # Logging verbosity (compact, detailed, full)

        # File logging defaults
        MAX_FILE_SIZE: Final[int] = 10485760  # 10 MB in bytes
        BACKUP_COUNT: Final[int] = 5

        # Console output defaults
        CONSOLE_ENABLED: Final[bool] = True
        CONSOLE_COLOR_ENABLED: Final[bool] = True

        # Performance tracking
        TRACK_PERFORMANCE: Final[bool] = False
        TRACK_TIMING: Final[bool] = False

        # Context and correlation
        INCLUDE_CONTEXT: Final[bool] = True
        INCLUDE_CORRELATION_ID: Final[bool] = True
        MAX_CONTEXT_KEYS: Final[int] = 50  # Maximum context keys to include in logs

        # Security
        MASK_SENSITIVE_DATA: Final[bool] = True

    class Cqrs:
        """CQRS pattern constants and configuration."""

        # CQRS handler type StrEnum for type safety
        class HandlerType(StrEnum):
            """CQRS handler type enumeration for type-safe handler identification."""

            COMMAND = "command"
            QUERY = "query"
            EVENT = "event"
            OPERATION = "operation"
            SAGA = "saga"

        # Handler type constants using StrEnum values
        DEFAULT_HANDLER_TYPE: HandlerType = HandlerType.COMMAND
        COMMAND_HANDLER_TYPE: HandlerType = HandlerType.COMMAND
        QUERY_HANDLER_TYPE: HandlerType = HandlerType.QUERY
        EVENT_HANDLER_TYPE: HandlerType = HandlerType.EVENT
        OPERATION_HANDLER_TYPE: HandlerType = HandlerType.OPERATION
        SAGA_HANDLER_TYPE: HandlerType = HandlerType.SAGA

        # Handler mode Literal type for type-safe annotations
        HandlerModeLiteral: TypeAlias = Literal[
            "command",
            "query",
            "event",
            "operation",
            "saga",
        ]

        # Message type Literal types for type-safe annotations
        CommandMessageTypeLiteral: TypeAlias = Literal["command"]
        QueryMessageTypeLiteral: TypeAlias = Literal["query"]
        EventMessageTypeLiteral: TypeAlias = Literal["event"]

        # Service metric type Literal for type-safe annotations
        ServiceMetricTypeLiteral: TypeAlias = Literal[
            "performance",
            "errors",
            "throughput",
            "latency",
            "availability",
        ]

        # Processing mode StrEnum
        class ProcessingMode(StrEnum):
            """Processing mode enumeration for type-safe mode identification."""

            BATCH = "batch"
            STREAM = "stream"
            PARALLEL = "parallel"
            SEQUENTIAL = "sequential"

        # Processing status StrEnum
        class ProcessingStatus(StrEnum):
            """Processing status enumeration for type-safe status tracking."""

            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"
            CANCELLED = "cancelled"

        # Validation level StrEnum
        class ValidationLevel(StrEnum):
            """Validation level enumeration for type-safe validation modes."""

            STRICT = "strict"
            LENIENT = "lenient"
            STANDARD = "standard"

        # Processing phase StrEnum
        class ProcessingPhase(StrEnum):
            """Processing phase enumeration for type-safe phase tracking."""

            PREPARE = "prepare"
            EXECUTE = "execute"
            VALIDATE = "validate"
            COMPLETE = "complete"

        # Bind type StrEnum
        class BindType(StrEnum):
            """Bind type enumeration for type-safe bind operations."""

            TEMPORARY = "temporary"
            PERMANENT = "permanent"

        # Merge strategy StrEnum
        class MergeStrategy(StrEnum):
            """Merge strategy enumeration for type-safe merge operations."""

            REPLACE = "replace"
            UPDATE = "update"
            MERGE_DEEP = "merge_deep"

        # Status StrEnum
        class Status(StrEnum):
            """Common status enumeration for type-safe status tracking."""

            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"
            COMPENSATING = "compensating"

        # Health status StrEnum
        class HealthStatus(StrEnum):
            """Health status enumeration for monitoring."""

            HEALTHY = "healthy"
            DEGRADED = "degraded"
            UNHEALTHY = "unhealthy"

        # Token type StrEnum
        class TokenType(StrEnum):
            """Token type enumeration for authentication."""

            BEARER = "bearer"
            API_KEY = "api_key"
            JWT = "jwt"

        # Notification status StrEnum
        class NotificationStatus(StrEnum):
            """Notification status enumeration."""

            PENDING = "pending"
            SENT = "sent"
            FAILED = "failed"

        # Token status StrEnum
        class TokenStatus(StrEnum):
            """Token status enumeration."""

            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"

        # Circuit breaker status StrEnum
        class CircuitBreakerStatus(StrEnum):
            """Circuit breaker status enumeration."""

            IDLE = "idle"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"

        # Batch status StrEnum
        class BatchStatus(StrEnum):
            """Batch processing status enumeration."""

            PENDING = "pending"
            PROCESSING = "processing"
            COMPLETED = "completed"
            FAILED = "failed"

        # Export status StrEnum
        class ExportStatus(StrEnum):
            """Export status enumeration."""

            PENDING = "pending"
            PROCESSING = "processing"
            COMPLETED = "completed"
            FAILED = "failed"

        # Operation status StrEnum
        class OperationStatus(StrEnum):
            """Operation status enumeration."""

            SUCCESS = "success"
            FAILURE = "failure"
            PARTIAL = "partial"

        # Serialization format StrEnum
        class SerializationFormat(StrEnum):
            """Serialization format enumeration."""

            JSON = "json"
            YAML = "yaml"
            TOML = "toml"
            MSGPACK = "msgpack"

        # Compression StrEnum
        class Compression(StrEnum):
            """Compression type enumeration."""

            NONE = "none"
            GZIP = "gzip"
            BZIP2 = "bzip2"
            LZ4 = "lz4"

        # Aggregation StrEnum
        class Aggregation(StrEnum):
            """Aggregation function enumeration."""

            SUM = "sum"
            AVG = "avg"
            MIN = "min"
            MAX = "max"
            COUNT = "count"

        # Action StrEnum
        class Action(StrEnum):
            """Action type enumeration."""

            GET = "get"
            CREATE = "create"
            UPDATE = "update"
            DELETE = "delete"
            LIST = "list"

        # Persistence level StrEnum
        class PersistenceLevel(StrEnum):
            """Persistence level enumeration."""

            MEMORY = "memory"
            DISK = "disk"
            DISTRIBUTED = "distributed"

        # Target format StrEnum
        class TargetFormat(StrEnum):
            """Target format enumeration."""

            FULL = "full"
            COMPACT = "compact"
            MINIMAL = "minimal"

        # Warning level StrEnum
        class WarningLevel(StrEnum):
            """Warning level enumeration."""

            NONE = "none"
            WARN = "warn"
            ERROR = "error"

        # Output format StrEnum
        class OutputFormat(StrEnum):
            """Output format enumeration."""

            DICT = "dict"
            JSON = "json"

        # Mode StrEnum
        class Mode(StrEnum):
            """Mode enumeration for various operations."""

            VALIDATION = "validation"
            SERIALIZATION = "serialization"

        # Registration status StrEnum
        class RegistrationStatus(StrEnum):
            """Registration status enumeration."""

            ACTIVE = "active"
            INACTIVE = "inactive"

        # Type aliases using StrEnum classes (ModelLiteral removed - use StrEnum directly)
        # HandlerModeSimple is a subset of HandlerType for simple command/query operations
        # Since Python doesn't support enum subsets natively, we use the full HandlerType
        HandlerModeSimple = HandlerType

        # Command/Query defaults
        DEFAULT_COMMAND_TYPE: Final[str] = "generic_command"
        DEFAULT_TIMESTAMP: Final[str] = ""  # Empty string for uninitialized timestamps

        # Timeout constants
        DEFAULT_TIMEOUT: Final[int] = 30000  # milliseconds
        MIN_TIMEOUT: Final[int] = 1000  # milliseconds
        MAX_TIMEOUT: Final[int] = 300000  # milliseconds (5 minutes)
        DEFAULT_COMMAND_TIMEOUT: Final[int] = 0  # 0 means no timeout override

        # Retry constants
        DEFAULT_RETRIES: Final[int] = 0
        MIN_RETRIES: Final[int] = 0
        MAX_RETRIES: Final[int] = 5
        DEFAULT_MAX_COMMAND_RETRIES: Final[int] = 0  # 0 means no retries

        # Pagination constants
        DEFAULT_PAGE_SIZE: Final[int] = 10
        MAX_PAGE_SIZE: Final[int] = 1000

        # Validation constants
        DEFAULT_MAX_VALIDATION_ERRORS: Final[int] = (
            10  # Maximum validation errors to report
        )
        DEFAULT_MINIMUM_THROUGHPUT: Final[int] = 10  # Minimum throughput threshold

        # Pipeline execution defaults
        DEFAULT_PARALLEL_EXECUTION: Final[bool] = (
            False  # Pipeline parallel execution default
        )
        DEFAULT_STOP_ON_ERROR: Final[bool] = True  # Pipeline error handling default

        # Error constants for CQRS operations
        CQRS_OPERATION_FAILED: Final[str] = "CQRS_OPERATION_FAILED"
        COMMAND_VALIDATION_FAILED: Final[str] = "COMMAND_VALIDATION_FAILED"
        QUERY_VALIDATION_FAILED: Final[str] = "QUERY_VALIDATION_FAILED"
        HANDLER_CONFIG_INVALID: Final[str] = "HANDLER_CONFIG_INVALID"
        COMMAND_PROCESSING_FAILED: Final[str] = "COMMAND_PROCESSING_FAILED"

    class Context:
        """Context management constants."""

        # Scope literals for context management
        SCOPE_GLOBAL: Final[str] = "global"  # Global scope across entire application
        SCOPE_REQUEST: Final[str] = "request"  # Request-specific scope
        SCOPE_SESSION: Final[str] = "session"  # Session-specific scope
        SCOPE_TRANSACTION: Final[str] = "transaction"  # Transaction-specific scope

        # Correlation ID configuration
        CORRELATION_ID_PREFIX: Final[str] = (
            "flext-"  # Standard prefix for correlation IDs
        )
        CORRELATION_ID_LENGTH: Final[int] = (
            12  # Standard length for correlation ID suffix
        )

        # Context operation timeouts and limits
        DEFAULT_CONTEXT_TIMEOUT: Final[int] = (
            30  # Default timeout for context operations (seconds)
        )
        MAX_CONTEXT_DEPTH: Final[int] = 10  # Maximum nesting depth for contexts
        MAX_CONTEXT_SIZE: Final[int] = 1000  # Maximum number of items in a context

        # Time conversion constants
        MILLISECONDS_PER_SECOND: Final[int] = 1000  # Conversion factor for elapsed time

        # Context export formats
        EXPORT_FORMAT_JSON: Final[str] = "json"
        EXPORT_FORMAT_DICT: Final[str] = "dict"

        # Metadata field names (centralized to avoid duplication)
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

        # Service lifecycle timeouts
        TIMEOUT_SECONDS: Final[int] = 30
        MIN_TIMEOUT_SECONDS: Final[int] = 1
        MAX_TIMEOUT_SECONDS: Final[int] = 300

        # Caching defaults
        MAX_CACHE_SIZE: Final[int] = 100

    class Dispatcher:
        """Message dispatcher constants."""

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

    class Pagination:
        """Pagination configuration constants."""

        DEFAULT_PAGE_NUMBER: Final[int] = 1  # Default starting page (1-based)
        DEFAULT_PAGE_SIZE: Final[int] = 10  # Default page size
        MAX_PAGE_SIZE: Final[int] = 1000  # Maximum allowed page size
        MIN_PAGE_SIZE: Final[int] = 1  # Minimum page size
        MIN_PAGE_NUMBER: Final[int] = 1  # Minimum page number
        MAX_PAGE_NUMBER: Final[int] = 10000  # Maximum page number

    class Mixins:
        """Constants for mixin operations."""

        # Field Names
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

        # Default States
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

        # Status Constants for Scripts and Diagnostics
        STATUS_PASS = "PASS"  # nosec B105 - Not a password, status constant
        STATUS_FAIL = "FAIL"
        STATUS_NO_TARGET = "NO_TARGET"
        STATUS_SKIP = "SKIP"
        STATUS_UNKNOWN = "UNKNOWN"

        # Default identifiers
        IDENTIFIER_UNKNOWN = "unknown"
        IDENTIFIER_DEFAULT = "default"
        IDENTIFIER_ANONYMOUS = "anonymous"
        IDENTIFIER_GUEST = "guest"
        IDENTIFIER_SYSTEM = "system"

        # Common method names
        METHOD_HANDLE = "handle"
        METHOD_PROCESS = "process"
        METHOD_EXECUTE = "execute"
        METHOD_PROCESS_COMMAND = "process_command"

        # Authentication types
        AUTH_BEARER = "bearer"
        AUTH_API_KEY = "api_key"
        AUTH_JWT = "jwt"

        # CQRS handler types
        HANDLER_COMMAND = "command"
        HANDLER_QUERY = "query"
        METHOD_VALIDATE = "validate"

        # Validation Types (simple string constants)
        VALIDATION_BASIC = "basic"
        VALIDATION_STRICT = "strict"
        VALIDATION_CUSTOM = "custom"

        # Serialization Defaults
        DEFAULT_JSON_INDENT = 2
        DEFAULT_ENCODING = "utf-8"
        DEFAULT_SORT_KEYS = False
        DEFAULT_ENSURE_ASCII = False

        # Boolean string representations
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

    class FlextWeb:
        """HTTP protocol constants."""

        HTTP_STATUS_MIN: Final[int] = 100
        HTTP_STATUS_MAX: Final[int] = 599

    class Processing:
        """Processing pipeline constants."""

        DEFAULT_MAX_WORKERS: Final[int] = 4  # Default maximum worker threads
        DEFAULT_BATCH_SIZE: Final[int] = 1000  # Default batch size for processing
        MAX_BATCH_SIZE: Final[int] = 10000  # Maximum batch size for validation


__all__ = ["FlextConstants"]
