"""Layer 0: Foundation constants for the entire FLEXT ecosystem.

This module provides the foundational constants that ALL other flext_core modules depend on.
As Layer 0 (pure Python), it has ZERO dependencies on other flext_core modules,
making it safe to import from anywhere without circular dependency risks.

**ARCHITECTURE HIERARCHY**:
- Layer 0: constants.py, typings.py (pure Python, no flext_core imports)
- Layer 0.5: runtime.py (imports Layer 0, exposes external libraries)
- Layer 1+: All other modules (import Layer 0 and 0.5)

**KEY FEATURES**:
- Error codes for exception categorization (50+ codes)
- Configuration defaults (timeouts, environments, network settings)
- Validation patterns (email, URL, UUID, phone - used by runtime.py)
- Logging constants (levels, formats)
- Platform constants (HTTP, encodings, paths)
- Message and display constants
- Telemetry and metrics defaults
- Complete immutability with typing.Final

**DEPENDENCIES**: ZERO flext_core imports (pure Python stdlib only)
**USED BY**: ALL flext_core modules and 32+ ecosystem projects

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import (
    ClassVar,
    Final,
    Literal,
)

from flext_core.__version__ import __version__


class FlextConstants:
    """Ecosystem-wide constant definitions for FLEXT foundation (Layer 0).

    FlextConstants provides the single source of truth for all constants
    across the entire FLEXT ecosystem. As Layer 0, this module has ZERO
    dependencies on other flext_core modules, making it safe to import
    from anywhere without circular dependency risks.

    **ARCHITECTURE ROLE**: Layer 0 - Pure Python Foundation
        - NO dependencies on other flext_core modules
        - Imported by runtime.py (Layer 0.5) for validation patterns
        - Imported by ALL higher-level modules (loggings, config, models, etc.)
        - Foundation for 32+ dependent ecosystem projects

    **PROVIDES**:
        - Error codes for exception categorization (50+ codes)
        - Configuration defaults (timeout, environment, network)
        - Validation patterns (email, URL, UUID, phone) - consumed by runtime.py
        - Logging constants (levels, formats, defaults)
        - Platform constants (HTTP, encodings, paths)
        - Message and display constants
        - Telemetry and metrics defaults
        - Complete immutability with typing.Final

    **PATTERN**: Namespace class with nested classes for organization
        - typing.Final for immutable constant definitions
        - enum.StrEnum for string enumeration types
        - typing.ClassVar for class-level constants
        - Pure Python stdlib only (no external dependencies)
        - Layer 0 foundation - imported by all other modules

    **How to use**: Access constants via nested namespaces
        ```python
        from flext_core import FlextConstants

        # Example 1: Access error codes for exception handling
        error_code = FlextConstants.Errors.VALIDATION_ERROR
        raise FlextException("Invalid input", error_code=error_code)

        # Example 2: Use configuration defaults
        timeout = FlextConstants.Config.DEFAULT_TIMEOUT
        max_workers = FlextConstants.Config.DEFAULT_MAX_WORKERS
        environment = FlextConstants.Config.DEFAULT_ENVIRONMENT

        # Example 3: Apply validation limits
        if len(name) < FlextConstants.Validation.MIN_NAME_LENGTH:
            return FlextResult[str].fail("Name too short")

        # Example 4: Use network constants for validation
        if not (
            FlextConstants.Network.MIN_PORT <= port <= FlextConstants.Network.MAX_PORT
        ):
            return FlextResult[int].fail("Invalid port number")

        # Example 5: Access logging constants
        log_level = FlextConstants.Logging.DEFAULT_LEVEL
        log_format = FlextConstants.Logging.DEFAULT_FORMAT

        # Example 6: Use message constants for user feedback
        success_msg = FlextConstants.Messages.OPERATION_SUCCESS
        failure_msg = FlextConstants.Messages.OPERATION_FAILURE
        ```

    Attributes:
        CONSTANTS_VERSION (str): Constants module version.
        PROJECT_PREFIX (str): Project prefix for environment vars.
        PROJECT_NAME (str): Full project name string.
        Core: Core identifiers (NAME, VERSION, DEFAULT_VERSION).
        Network: Network defaults (ports, timeouts).
        Validation: Validation limits (length, format rules).
        Errors: Error codes for exception categorization.
        Messages: User-facing message constants.
        Config: Configuration defaults and limits.
        Platform: Platform-specific constants (OS, paths).
        Logging: Logging configuration constants.
        Container: DI container defaults.
        Processing: Processing pipeline constants.
        Security: Security and encryption constants.

    Note:
        Layer 0 foundation with ZERO dependencies. All constants are
        immutable using typing.Final. This is the single source of
        truth for the entire FLEXT ecosystem. Constants guaranteed
        stable throughout 1.x series. Import only FlextConstants,
        never individual nested classes.

    Warning:
        Never modify constants at runtime - all are Final. Never
        create local constant copies - always import from this
        module. Constants must remain backward compatible across
        1.x series. Breaking constant changes require major version.

    Example:
        Complete constant usage pattern:

        >>> from flext_core import FlextConstants
        >>> error_code = FlextConstants.Errors.VALIDATION_ERROR
        >>> print(error_code)
        VALIDATION_ERROR
        >>> timeout = FlextConstants.Config.DEFAULT_TIMEOUT
        >>> print(timeout)
        30

    See Also:
        FlextExceptions: For error code usage in exceptions.
        FlextConfig: For runtime configuration management.
        FlextTypes: For type definitions and aliases.

    """

    PROJECT_PREFIX = "FLEXT_CORE"
    CONSTANTS_VERSION: Final[str] = __version__  # Hardcoded to match Core.VERSION
    PROJECT_NAME: Final[str] = "flext-core"

    # Direct access support - backward compatible with deprecation warnings
    _deprecation_warnings_shown: ClassVar[set[str]] = set()

    def __class_getitem__(cls, key: str) -> object:
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
            >>> FlextConstants.Config.DEFAULT_TIMEOUT
            30

        """
        # Parse nested key like "Errors.VALIDATION_ERROR"
        parts = key.split(".")
        value: object = cls

        try:
            for part in parts:
                value = getattr(value, part)
            return value
        except AttributeError as e:
            msg = f"Constant path '{key}' not found in FlextConstants"
            raise AttributeError(msg) from e

    class Core:
        """Core identifiers hardened for the 1.0.0 release cycle."""

        NAME: Final[str] = "FLEXT"  # Usage count: 1
        VERSION: Final[str] = __version__
        DEFAULT_VERSION: Final[str] = __version__

        # Core-specific overrides
        SYSTEM_NAME: Final[str] = "flext-core"
        MAJOR_VERSION: Final[int] = 1
        MINOR_VERSION: Final[int] = 0
        PATCH_VERSION: Final[int] = 0

        # Semantic zero and initial values
        ZERO: Final[int] = 0  # Semantic zero for counters/initialization
        INITIAL_TIME: Final[float] = (
            0.0  # Initial timestamp value  # Initial timestamp value
        )

    class Network:
        """Network defaults shared across dispatcher-aligned services."""

        MIN_PORT: Final[int] = 1  # Usage count: 4
        MAX_PORT: Final[int] = 65535  # Usage count: 4
        TOTAL_TIMEOUT: Final[int] = 60  # Usage count: 0
        DEFAULT_TIMEOUT: Final[int] = 30  # Usage count: 4
        DEFAULT_MAX_RETRIES: Final[int] = (
            3  # Default retry attempts for network operations
        )
        DEFAULT_CONNECTION_POOL_SIZE: Final[int] = 10  # Usage count: 1
        MAX_CONNECTION_POOL_SIZE: Final[int] = 100  # Usage count: 1

    class Validation:
        """Validation guardrails referenced in modernization docs.

        Usage example:
            from flext_core import FlextConstants, FlextResult

            def validate_user_name(name: str) -> FlextResult[str]:
                if len(name) < FlextConstants.Validation.MIN_NAME_LENGTH:
                    return FlextResult[str].fail(
                        f"Name must be at least {FlextConstants.Validation.MIN_NAME_LENGTH} characters"
                    )
                if len(name) > FlextConstants.Validation.MAX_NAME_LENGTH:
                    return FlextResult[str].fail(
                        f"Name cannot exceed {FlextConstants.Validation.MAX_NAME_LENGTH} characters"
                    )
                return FlextResult[str].ok(name)

            def validate_email(email: str) -> FlextResult[str]:
                if len(email) > FlextConstants.Validation.MAX_EMAIL_LENGTH:
                    return FlextResult[str].fail("Email too long")
                parts = email.split("@")
                if len(parts) != FlextConstants.Validation.EMAIL_PARTS_COUNT:
                    return FlextResult[str].fail("Invalid email format")
                return FlextResult[str].ok(email)

            def validate_percentage(value: float) -> FlextResult[float]:
                if value < FlextConstants.Validation.MIN_PERCENTAGE:
                    return FlextResult[float].fail("Percentage cannot be negative")
                if value > FlextConstants.Validation.MAX_PERCENTAGE:
                    return FlextResult[float].fail("Percentage cannot exceed 100")
                return FlextResult[float].ok(value)
        """

        MIN_NAME_LENGTH: Final[int] = 2  # Usage count: 1
        MAX_NAME_LENGTH: Final[int] = 100  # Usage count: 0
        MIN_SERVICE_NAME_LENGTH: Final[int] = 2  # Usage count: 0
        MAX_EMAIL_LENGTH: Final[int] = 254  # Usage count: 0
        EMAIL_PARTS_COUNT: Final[int] = 2  # Expected parts when splitting email by @
        MIN_PERCENTAGE: Final[float] = 0.0  # Usage count: 0
        MAX_PERCENTAGE: Final[float] = 100.0  # Usage count: 0
        MIN_SECRET_KEY_LENGTH: Final[int] = 32  # Usage count: 0

        # Phone number validation
        MIN_PHONE_DIGITS: Final[int] = 10  # Minimum phone number length
        MIN_USERNAME_LENGTH: Final[int] = 3  # Minimum username length for validation
        MAX_AGE: Final[int] = 150  # Maximum valid age for persons
        MIN_AGE: Final[int] = 0  # Minimum valid age for persons
        PREVIEW_LENGTH: Final[int] = 50  # Maximum length for string previews/truncation
        DISCOUNT_THRESHOLD: Final[int] = 100  # Minimum total for discount eligibility
        DISCOUNT_RATE: Final[float] = 0.05  # 5% discount rate
        SLOW_OPERATION_THRESHOLD: Final[float] = (
            0.1  # Threshold for slow operation warning (seconds)
        )
        RESOURCE_LIMIT_MIN: Final[int] = 50  # Minimum resource limit
        FILTER_THRESHOLD: Final[int] = 5  # General filter threshold for comparisons
        RETRY_COUNT_MIN: Final[int] = 2  # Minimum retry attempts
        RETRY_COUNT_MAX: Final[int] = 3  # Maximum retry attempts
        WORKERS_TEST_COUNT: Final[int] = 8  # Test-specific worker count
        TIMEOUT_TEST_SECONDS: Final[int] = 60  # Test-specific timeout

    class Errors:
        """Canonical error codes surfaced in telemetry narratives.

        Error codes define the standard taxonomy for error classification
        across the FLEXT ecosystem. Constants with 0 usage are part of the
        stable 1.0.0 API contract and reserved for:
        - Future flext-* library implementations
        - Third-party ecosystem extensions
        - Backward compatibility guarantees
        - Public API stability requirements

        Usage pattern:
            raise FlextException(
                message="Validation failed",
                error_code=FlextConstants.Errors.VALIDATION_ERROR
            )
        """

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

        # System errors (reserved for critical failures)
        EXCEPTION_ERROR: Final[str] = (
            "EXCEPTION_ERROR"  # Reserved for exception wrapping
        )
        CRITICAL_ERROR: Final[str] = (
            "CRITICAL_ERROR"  # Reserved for system failures  # Usage count: 1
        )

        # Error category literals for type safety
        class CategoryLiteral:
            """Error category literals for type annotations."""

            VALIDATION: Final[str] = "validation"
            NETWORK: Final[str] = "network"
            DATABASE: Final[str] = "database"
            AUTH: Final[str] = "auth"
            SYSTEM: Final[str] = "system"
            UNKNOWN: Final[str] = "unknown"

        # Error severity literals for type safety
        class SeverityLiteral:
            """Error severity literals for type annotations."""

            LOW: Final[str] = "low"
            MEDIUM: Final[str] = "medium"
            HIGH: Final[str] = "high"
            CRITICAL: Final[str] = "critical"

        # Service type literals for type safety
        class ServiceTypeLiteral:
            """Service type literals for type annotations."""

            INSTANCE: Final[str] = "instance"
            FACTORY: Final[str] = "factory"
            SINGLETON: Final[str] = "singleton"

        # Service lifecycle literals for type safety
        class ServiceLifecycleLiteral:
            """Service lifecycle literals for type annotations."""

            INITIALIZING: Final[str] = "initializing"
            READY: Final[str] = "ready"
            RUNNING: Final[str] = "running"
            STOPPING: Final[str] = "stopping"
            STOPPED: Final[str] = "stopped"
            ERROR: Final[str] = "error"

        # Service protocol literals for type safety
        class ServiceProtocolLiteral:
            """Service protocol literals for type annotations."""

            HTTP: Final[str] = "http"
            GRPC: Final[str] = "grpc"
            WEBSOCKET: Final[str] = "websocket"
            MESSAGE_QUEUE: Final[str] = "message_queue"

        # Environment literals for type safety
        class EnvironmentLiteral:
            """Environment literals for type annotations."""

            DEVELOPMENT: Final[str] = "development"
            STAGING: Final[str] = "staging"
            PRODUCTION: Final[str] = "production"
            TEST: Final[str] = "test"
            LOCAL: Final[str] = "local"

        # Log level literals for type safety
        class LogLevelLiteral:
            """Log level literals for type annotations."""

            DEBUG: Final[str] = "DEBUG"
            INFO: Final[str] = "INFO"
            WARNING: Final[str] = "WARNING"
            ERROR: Final[str] = "ERROR"
            CRITICAL: Final[str] = "CRITICAL"

        # Output format literals for type safety
        class OutputFormatLiteral:
            """Output format literals for type annotations."""

            JSON: Final[str] = "json"
            YAML: Final[str] = "yaml"
            TABLE: Final[str] = "table"
            CSV: Final[str] = "csv"
            TEXT: Final[str] = "text"
            XML: Final[str] = "xml"

        # Serialization format literals for type safety
        class SerializationFormatLiteral:
            """Serialization format literals for type annotations."""

            JSON: Final[str] = "json"
            YAML: Final[str] = "yaml"
            TOML: Final[str] = "toml"
            INI: Final[str] = "ini"
            XML: Final[str] = "xml"

    class Messages:
        """User-facing validation and failure messages.

        Usage example:
            from flext_core import FlextConstants, FlextResult, FlextException

            def process_user_input(data: dict) -> FlextResult[FlextTypes.Dict]:
                if not data:
                    return FlextResult[FlextTypes.Dict].fail(
                        FlextConstants.Messages.NULL_DATA,
                        error_code="VALIDATION_ERROR"
                    )

                if "name" not in data or not data["name"]:
                    return FlextResult[FlextTypes.Dict].fail(
                        FlextConstants.Messages.VALUE_EMPTY,
                        error_code="VALIDATION_ERROR"
                    )

                try:
                    processed = transform_data(data)
                    return FlextResult[FlextTypes.Dict].ok(processed)
                except Exception as e:
                    return FlextResult[FlextTypes.Dict].fail(
                        FlextConstants.Messages.OPERATION_FAILED,
                        error_code="OPERATION_ERROR"
                    )

            def validate_type(value: any, expected_type: type) -> FlextResult[None]:
                if not isinstance(value, expected_type):
                    return FlextResult[None].fail(
                        FlextConstants.Messages.TYPE_MISMATCH,
                        error_code="TYPE_ERROR"
                    )
                return FlextResult[None].ok(None)
        """

        TYPE_MISMATCH: Final[str] = "Type mismatch"  # Usage count: 2
        SERVICE_NAME_EMPTY: Final[str] = (
            "Service name cannot be empty"  # Usage count: 0
        )
        OPERATION_FAILED: Final[str] = "Operation failed"  # Usage count: 0
        INVALID_INPUT: Final[str] = "Invalid input"  # Usage count: 0
        VALUE_EMPTY: Final[str] = "Value cannot be empty"  # Usage count: 0
        VALIDATION_FAILED: Final[str] = "Validation failed"  # Usage count: 0
        UNKNOWN_ERROR: Final[str] = "Unknown error"  # Usage count: 0
        NULL_DATA: Final[str] = (
            "Data cannot be null"  # Usage count: 0  # Usage count: 0
        )
        REDACTED_SECRET: Final[str] = (
            "***REDACTED***"  # For hiding sensitive data in logs
        )

    class Entities:
        """Entity validation prompts reused across services."""

        ENTITY_ID_EMPTY: Final[str] = "Entity ID cannot be empty"  # Usage count: 0

    class Defaults:
        """Baseline defaults called out in onboarding docs."""

        TIMEOUT: Final[int] = 30  # Usage count: 18
        PAGE_SIZE: Final[int] = 100  # Usage count: 2
        TIMEOUT_SECONDS: Final[int] = 30  # Default timeout for operations
        ENVIRONMENT: Final[str] = "development"  # Default environment  # Usage count: 2

        # Cache defaults
        CACHE_TTL: Final[int] = 300  # Cache time-to-live in seconds (5 minutes)
        MAX_CACHE_SIZE: Final[int] = 100  # Maximum cache entries

        # Message and operation defaults
        MAX_MESSAGE_LENGTH: Final[int] = 100  # Maximum message length for truncation
        DEFAULT_MIDDLEWARE_ORDER: Final[int] = 0  # Default middleware execution order
        OPERATION_TIMEOUT_SECONDS: Final[int] = 30  # Default operation timeout

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

        # Encoding and processing defaults
        DEFAULT_ENCODING: Final[str] = "utf-8"  # Default character encoding
        DEFAULT_BATCH_SIZE: Final[int] = 1000  # Default batch size for processing

        # Security and validation constants
        MIN_TOKEN_LENGTH: Final[int] = (
            8  # Minimum length for security tokens and passwords
        )
        MAX_TIMEOUT_SECONDS: Final[int] = 3600  # Maximum timeout in seconds (1 hour)
        MAX_ERROR_DISPLAY: Final[int] = (
            5  # Maximum errors to display in batch processing
        )
        MAX_REGEX_PATTERN_LENGTH: Final[int] = (
            1000  # Maximum regex pattern length to prevent ReDoS
        )

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
        DEFAULT_ENVIRONMENT: Final[str] = "development"
        DOTENV_FILES: Final[list[str]] = [
            ".env",
            ".internal.invalid",
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

        class LogLevel(StrEnum):
            """Standard log levels for centralized logging configuration."""

            DEBUG = "DEBUG"
            INFO = "INFO"
            WARNING = "WARNING"
            ERROR = "ERROR"
            CRITICAL = "CRITICAL"

        PROFILE_BATCH_JOB: Final[str] = "batch_job"
        PROFILE_MICROSERVICE: Final[str] = "microservice"

        # Timeout defaults
        DEFAULT_TIMEOUT: Final[int] = 30  # Default timeout in seconds

        class ConfigSource(StrEnum):
            """Enumerate configuration origins supported by FlextConfig."""

            FILE = "file"  # Usage count: 0
            ENVIRONMENT = "env"  # Usage count: 0
            CLI = "cli"  # Usage count: 0
            DEFAULT = "default"  # Usage count: 0
            DOTENV = "dotenv"  # Usage count: 0

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

        class WorkspaceStatus(StrEnum):
            """Enumeration capturing core workspace states."""

            INITIALIZING = "initializing"
            READY = "ready"
            ERROR = "error"
            MAINTENANCE = "maintenance"

    class Platform:
        """Platform defaults referenced by CLI and adapter packages.

        Usage example:
            from flext_core import FlextConstants, FlextResult

            # HTTP client configuration
            def build_api_url(path: str) -> str:
                return (
                    f"{FlextConstants.Platform.PROTOCOL_HTTPS}"
                    f"{FlextConstants.Platform.DEFAULT_HOST}:"
                    f"{FlextConstants.Platform.FLEXT_API_PORT}{path}"
                )

            # Database connection
            def get_postgres_url(host: str, db_name: str) -> str:
                port = FlextConstants.Platform.POSTGRES_DEFAULT_PORT
                return f"{FlextConstants.Platform.DB_SCHEME_POSTGRESQL}{host}:{port}/{db_name}"

            # HTTP headers
            def create_request_headers(api_key: str) -> dict:
                return {
                    FlextConstants.Platform.HEADER_CONTENT_TYPE: FlextConstants.Platform.MIME_TYPE_JSON,
                    FlextConstants.Platform.HEADER_API_KEY: api_key,
                    FlextConstants.Platform.HEADER_USER_AGENT: "FLEXT/1.0.0"
                }

            # HTTP status validation
            def is_success_status(status_code: int) -> bool:
                return (
                    FlextConstants.Platform.HTTP_STATUS_OK <= status_code < 300
                )

            # LDAP connection
            def build_ldap_url(host: str, use_ssl: bool = False) -> str:
                protocol = (
                    FlextConstants.Platform.PROTOCOL_LDAPS if use_ssl
                    else FlextConstants.Platform.PROTOCOL_LDAP
                )
                port = (
                    FlextConstants.Platform.LDAPS_DEFAULT_PORT if use_ssl
                    else FlextConstants.Platform.LDAP_DEFAULT_PORT
                )
                return f"{protocol}{host}:{port}"
        """

        FLEXT_API_PORT: Final[int] = 8000
        DEFAULT_HOST: Final[str] = "localhost"
        LOOPBACK_IP: Final[str] = "127.0.0.1"  # Localhost IP address

        # Database defaults
        POSTGRES_DEFAULT_PORT: Final[int] = 5432
        MYSQL_DEFAULT_PORT: Final[int] = 3306
        REDIS_DEFAULT_PORT: Final[int] = 6379
        MONGODB_DEFAULT_PORT: Final[int] = 27017
        DATABASE_DEFAULT_PORT: Final[int] = 1521  # Oracle database default port
        DEFAULT_HTTP_PORT: Final[int] = 8080  # Alternative HTTP port

        # HTTP status code validation ranges
        MIN_HTTP_STATUS_CODE: Final[int] = 200
        MAX_HTTP_STATUS_CODE: Final[int] = 599

        # HTTP methods
        HTTP_METHOD_GET: Final[str] = "GET"
        HTTP_METHOD_POST: Final[str] = "POST"
        HTTP_METHOD_PUT: Final[str] = "PUT"
        HTTP_METHOD_DELETE: Final[str] = "DELETE"
        HTTP_METHOD_PATCH: Final[str] = "PATCH"
        HTTP_METHOD_HEAD: Final[str] = "HEAD"
        HTTP_METHOD_OPTIONS: Final[str] = "OPTIONS"

        VALID_HTTP_METHODS: Final[set[str]] = {
            HTTP_METHOD_GET,
            HTTP_METHOD_POST,
            HTTP_METHOD_PUT,
            HTTP_METHOD_DELETE,
            HTTP_METHOD_PATCH,
            HTTP_METHOD_HEAD,
            HTTP_METHOD_OPTIONS,
        }

        # Common MIME types
        MIME_TYPE_JSON: Final[str] = "application/json"
        MIME_TYPE_HTML: Final[str] = "text/html"
        MIME_TYPE_PLAIN: Final[str] = "text/plain"
        MIME_TYPE_XML: Final[str] = "application/xml"

        # Protocol prefixes
        PROTOCOL_HTTP: Final[str] = "http://"
        PROTOCOL_HTTPS: Final[str] = "https://"
        PROTOCOL_TCP: Final[str] = "tcp://"
        PROTOCOL_UDP: Final[str] = "udp://"
        PROTOCOL_LDAP: Final[str] = "ldap://"
        PROTOCOL_LDAPS: Final[str] = "ldaps://"

        # LDAP protocol constants
        LDAP_DEFAULT_PORT: Final[int] = 389
        LDAPS_DEFAULT_PORT: Final[int] = 636

        # LDAP search scope constants
        LDAP_SCOPE_BASE: Final[str] = "base"
        LDAP_SCOPE_LEVEL: Final[str] = "level"
        LDAP_SCOPE_SUBTREE: Final[str] = "subtree"

        # LDAP modify operation constants
        LDAP_MODIFY_ADD: Final[str] = "add"
        LDAP_MODIFY_DELETE: Final[str] = "delete"
        LDAP_MODIFY_REPLACE: Final[str] = "replace"

        # LDAP authentication constants
        LDAP_AUTH_SIMPLE: Final[str] = "simple"
        LDAP_AUTH_SASL: Final[str] = "sasl"

        # LDAP attribute constants
        LDAP_ATTR_ALL: Final[str] = "*"
        LDAP_ATTR_OBJECT_CLASS: Final[str] = "objectClass"
        LDAP_ATTR_DISTINGUISHED_NAME: Final[str] = "distinguishedName"

        # Output formats
        FORMAT_JSON: Final[str] = "json"
        FORMAT_XML: Final[str] = "xml"
        FORMAT_YAML: Final[str] = "yaml"
        FORMAT_TABLE: Final[str] = "table"
        FORMAT_CSV: Final[str] = "csv"

        # File extensions
        EXT_PYTHON: Final[str] = ".py"
        EXT_JSON: Final[str] = ".json"
        EXT_YAML: Final[str] = ".yaml"
        EXT_YML: Final[str] = ".yml"
        EXT_TOML: Final[str] = ".toml"
        EXT_TXT: Final[str] = ".txt"
        EXT_LOG: Final[str] = ".log"
        EXT_CSV: Final[str] = ".csv"
        EXT_XML: Final[str] = ".xml"
        EXT_MD: Final[str] = ".md"

        # Database types and drivers
        DB_SQLITE: Final[str] = "sqlite"
        DB_POSTGRESQL: Final[str] = "postgresql"
        DB_MYSQL: Final[str] = "mysql"
        DB_ORACLE: Final[str] = "oracle"
        DB_MONGODB: Final[str] = "mongodb"
        DB_REDIS: Final[str] = "redis"

        # Database URL schemes
        DB_SCHEME_SQLITE: Final[str] = "sqlite:///"
        DB_SCHEME_POSTGRESQL: Final[str] = "postgresql://"
        DB_SCHEME_MYSQL: Final[str] = "mysql://"
        DB_SCHEME_ORACLE: Final[str] = "oracle://"
        DB_SCHEME_MONGODB: Final[str] = "mongodb://"
        DB_SCHEME_REDIS: Final[str] = "redis://"

        # Environment variable names
        ENV_PYTEST_CURRENT_TEST: Final[str] = "PYTEST_CURRENT_TEST"
        ENV_LOG_LEVEL: Final[str] = "LOG_LEVEL"
        ENV_DEBUG: Final[str] = "DEBUG"
        ENV_ENVIRONMENT: Final[str] = "ENVIRONMENT"
        ENV_DATABASE_URL: Final[str] = "DATABASE_URL"

        # Common directory names
        DIR_LOGS: Final[str] = "logs"
        DIR_DATA: Final[str] = "data"
        DIR_CONFIG: Final[str] = "config"
        DIR_TEMP: Final[str] = "temp"
        DIR_CACHE: Final[str] = "cache"

        # Configuration delimiters
        ENV_NESTED_DELIMITER: Final[str] = "__"
        ENV_PREFIX: Final[str] = "FLEXT_"
        ENV_FILE_DEFAULT: Final[str] = ".env"

        # Validation patterns for runtime type guards
        PATTERN_SECURITY_LEVEL: Final[str] = "^(low|standard|high|critical)$"
        PATTERN_DATA_CLASSIFICATION: Final[str] = (
            "^(public|internal|confidential|restricted)$"
        )

        # Email validation (RFC 5322 simplified) - used by FlextRuntime.is_valid_email()
        PATTERN_EMAIL: Final[str] = (
            r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}"
            r"[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
        )

        # URL validation (HTTP/HTTPS) - used by FlextRuntime.is_valid_url()
        PATTERN_URL: Final[str] = (
            r"^https?://"
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
            r"localhost|"
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
            r"(?::\d+)?"
            r"(?:/?|[/?]\S+)$"
        )

        # Phone number validation (international format with separators) - used by FlextRuntime.is_valid_phone()
        PATTERN_PHONE_NUMBER: Final[str] = r"^\+?[\d\s\-\(\)]{10,20}$"

        # UUID validation (with/without hyphens) - used by FlextRuntime.is_valid_uuid()
        PATTERN_UUID: Final[str] = (
            r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
            r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
        )

        # File path validation - used by FlextRuntime.is_valid_path()
        # Allows Windows paths (C:\...) and Unix paths (/home/...)
        PATTERN_PATH: Final[str] = r'^[^<>"|?*\x00-\x1F]+$'

        # HTTP headers
        HEADER_CONTENT_TYPE: Final[str] = "Content-Type"
        HEADER_AUTHORIZATION: Final[str] = "Authorization"
        HEADER_USER_AGENT: Final[str] = "User-Agent"
        HEADER_ACCEPT: Final[str] = "Accept"
        HEADER_CACHE_CONTROL: Final[str] = "Cache-Control"
        HEADER_API_KEY: Final[str] = "X-API-Key"
        HEADER_REQUEST_ID: Final[str] = "X-Request-ID"
        HEADER_CORRELATION_ID: Final[str] = "X-Correlation-ID"

        # HTTP status codes
        HTTP_STATUS_OK: Final[int] = 200
        HTTP_STATUS_CREATED: Final[int] = 201
        HTTP_STATUS_BAD_REQUEST: Final[int] = 400
        HTTP_STATUS_UNAUTHORIZED: Final[int] = 401
        HTTP_STATUS_FORBIDDEN: Final[int] = 403
        HTTP_STATUS_NOT_FOUND: Final[int] = 404
        HTTP_STATUS_INTERNAL_ERROR: Final[int] = 500

        # HTTP status ranges
        MIN_HTTP_STATUS_RANGE: Final[int] = 100
        MAX_HTTP_STATUS_RANGE: Final[int] = 599

    class Observability:
        """Observability defaults consumed by FlextLogger."""

        DEFAULT_LOG_LEVEL: Final[str] = "INFO"  # Usage count: 0

    class Performance:
        """Performance thresholds and operational limits for FLEXT services."""

        # Connection and timeout settings
        CONNECTION_TIMEOUT: Final[int] = 30  # seconds
        SOCKET_TIMEOUT: Final[int] = 30  # seconds
        SUBPROCESS_TIMEOUT: Final[int] = 300  # 5 minutes for build/test operations
        SUBPROCESS_TIMEOUT_SHORT: Final[int] = 180  # 3 minutes for quick operations

        # Database connection defaults
        DEFAULT_DB_POOL_SIZE: Final[int] = 10
        MIN_DB_POOL_SIZE: Final[int] = 1
        MAX_DB_POOL_SIZE: Final[int] = 100

        # Operation limits
        MAX_RETRY_ATTEMPTS_LIMIT: Final[int] = 10
        DEFAULT_TIMEOUT_LIMIT: Final[int] = 300

        # Timing and delays
        DEFAULT_DELAY_SECONDS: Final[float] = 1.0
        DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0

        # Cache and parallel execution defaults
        DEFAULT_CACHE_SIZE: Final[int] = 100  # Default cache size for CqrsCache
        DEFAULT_EMPTY_STRING: Final[str] = ""  # Safe empty string constant
        DEFAULT_MAX_PARALLEL: Final[int] = 4  # Default parallel execution threads
        DEFAULT_MAX_DELAY_SECONDS: Final[float] = 60.0
        DEFAULT_INITIAL_DELAY_SECONDS: Final[float] = 1.0
        DEFAULT_RECOVERY_TIMEOUT: Final[int] = 60
        DEFAULT_FALLBACK_DELAY: Final[float] = 0.1

        # Backward compatibility aliases for batch processing
        DEFAULT_BATCH_SIZE: Final[int] = 1000  # Alias for BatchProcessing.DEFAULT_SIZE
        MAX_BATCH_SIZE_VALIDATION: Final[int] = (
            10000  # Maximum batch size for validation
        )

        # Batch processing constants - consolidated from duplicates
        class BatchProcessing:
            """Batch processing configuration constants - consolidated from duplicates."""

            # Core batch sizes (consolidated from previous duplicates)
            DEFAULT_SIZE: Final[int] = 1000  # Main batch size (was DEFAULT_BATCH_SIZE)
            SMALL_SIZE: Final[int] = (
                100  # Small batch size (was DEFAULT_BATCH_SIZE_SMALL)
            )
            STREAM_SIZE: Final[int] = (
                100  # Stream batch size (was DEFAULT_STREAM_BATCH_SIZE)
            )

            # Validation limits
            MAX_VALIDATION_SIZE: Final[int] = (
                1000  # Maximum batch size for validation (was MAX_BATCH_SIZE_VALIDATION)
            )
            MAX_ITEMS: Final[int] = 10000  # Maximum batch items (existing)

        # Pagination and processing (duplicated batch constants removed)
        DEFAULT_PAGE_SIZE: Final[int] = 10
        DEFAULT_PAGE_NUMBER: Final[int] = 1
        DEFAULT_TIME_RANGE_SECONDS: Final[int] = 3600  # 1 hour
        DEFAULT_TTL_SECONDS: Final[int] = 3600  # 1 hour default TTL

        # Field validation constraints
        DEFAULT_VERSION: Final[int] = 1  # Default version for entities
        MIN_VERSION: Final[int] = 1  # Minimum version number
        DEFAULT_SKIP: Final[int] = 0  # Default skip value for pagination
        MIN_SKIP: Final[int] = 0  # Minimum skip value
        DEFAULT_TAKE: Final[int] = 10  # Default take value for pagination
        MIN_TAKE: Final[int] = 1  # Minimum take value
        MAX_TAKE: Final[int] = 100  # Maximum take value
        DEFAULT_CURRENT_STEP: Final[int] = 0  # Default current step
        MIN_CURRENT_STEP: Final[int] = 0  # Minimum current step

        # String length constraints
        CURRENCY_CODE_LENGTH: Final[int] = 3  # ISO currency code length
        MIN_NAME_LENGTH: Final[int] = 1  # Minimum name length

        # Numeric validation constraints
        DEFAULT_PRIORITY: Final[int] = 0  # Default priority value
        MIN_PRIORITY: Final[int] = 0  # Minimum priority value
        DEFAULT_ROLLOUT_PERCENTAGE: Final[float] = 0.0  # Default rollout percentage
        MIN_ROLLOUT_PERCENTAGE: Final[float] = 0.0  # Minimum rollout percentage
        MAX_ROLLOUT_PERCENTAGE: Final[float] = 100.0  # Maximum rollout percentage
        CURRENCY_DECIMAL_PLACES: Final[int] = 2  # Standard currency decimal places

        # Validation limits
        MAX_METADATA_SIZE: Final[int] = 10000  # Maximum metadata size in characters
        CRITICAL_DURATION_MS: Final[int] = (
            5000  # Critical performance threshold (5 seconds)
        )
        CRITICAL_USAGE_PERCENT: Final[int] = 90  # Critical CPU/memory usage threshold
        MAX_TIMEOUT_SECONDS: Final[int] = 600  # Maximum timeout (10 minutes)
        MAX_BATCH_OPERATIONS: Final[int] = 1000  # Maximum batch operations
        MAX_OPERATION_NAME_LENGTH: Final[int] = 100  # Maximum operation name length
        MAX_DIMENSIONS: Final[int] = 20  # Maximum dimensions for metrics
        PARALLEL_THRESHOLD: Final[int] = 100  # Threshold for parallel processing

        # Data structure constants
        EXPECTED_TUPLE_LENGTH: Final[int] = 2
        EXPECTED_PAIR_LENGTH: Final[int] = 2

        # Rate limiting and throttling
        DEFAULT_RATE_LIMIT_WINDOW: Final[int] = 60  # seconds
        DEFAULT_REQUEST_TIMEOUT: Final[int] = 30  # seconds

        # Module-specific performance thresholds
        CLI_PERFORMANCE_WARNING_MS: Final[float] = 2000.0  # 2 seconds
        CLI_PERFORMANCE_CRITICAL_MS: Final[float] = 10000.0  # 10 seconds
        AUTH_PERFORMANCE_WARNING_MS: Final[float] = 1000.0  # 1 second
        AUTH_PERFORMANCE_CRITICAL_MS: Final[float] = 5000.0  # 5 seconds
        HIGH_MEMORY_THRESHOLD_BYTES: Final[int] = (
            100 * 1024 * 1024
        )  # 100MB  # 100MB  # 100MB

    class Reliability:
        """Reliability thresholds backing retry guidance."""

        MAX_RETRY_ATTEMPTS: Final[int] = 3  # Usage count: 1
        DEFAULT_MAX_RETRIES: Final[int] = 3  # Usage count: 1 (referenced in models.py)
        DEFAULT_RETRY_DELAY_SECONDS: Final[int] = 1  # Default delay between retries
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
        LINEAR_BACKOFF_FACTOR: Final[float] = (
            1.0  # Linear backoff (no exponential growth)
        )

        # Rate limiting constants
        DEFAULT_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60  # 1 minute window
        DEFAULT_RATE_LIMIT_MAX_REQUESTS: Final[int] = 100  # Max requests per window

        # Circuit breaker constants
        DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5  # Open after failures
        DEFAULT_CIRCUIT_BREAKER_TIMEOUT_SECONDS: Final[int] = 60  # Recovery time
        DEFAULT_CIRCUIT_BREAKER_RECOVERY: Final[float] = (
            60.0  # Recovery timeout (float version)
        )
        ALTERNATIVE_RECOVERY_TIMEOUT: Final[float] = (
            30.0  # Alternative recovery timeout
        )

    class Logging:
        """Logging configuration constants for FlextLogger and FlextConfig.

        Provides default values for logging configuration across the FLEXT ecosystem,
        ensuring consistent logging behavior.
        """

        # Log level constants
        DEBUG: Final[str] = "DEBUG"
        INFO: Final[str] = "INFO"
        WARNING: Final[str] = "WARNING"
        ERROR: Final[str] = "ERROR"
        CRITICAL: Final[str] = "CRITICAL"

        # Log level defaults
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

    class Security:
        """Security-related constants for authentication and authorization.

        Usage example:
            from flext_core import FlextConstants, FlextResult
            from datetime import datetime, timedelta
            import bcrypt

            # JWT token generation with appropriate expiry
            def create_access_token(user_id: str, short_lived: bool = False) -> dict:
                expiry_minutes = (
                    FlextConstants.Security.SHORT_JWT_EXPIRY_MINUTES if short_lived
                    else FlextConstants.Security.DEFAULT_JWT_EXPIRY_MINUTES
                )
                return {
                    "user_id": user_id,
                    "exp": datetime.utcnow() + timedelta(minutes=expiry_minutes),
                    "type": "access"
                }

            # Password validation
            def validate_password(password: str) -> FlextResult[str]:
                if len(password) < FlextConstants.Security.MIN_PASSWORD_LENGTH:
                    return FlextResult[str].fail(
                        f"Password must be at least {FlextConstants.Security.MIN_PASSWORD_LENGTH} characters"
                    )
                if len(password) > FlextConstants.Security.MAX_PASSWORD_LENGTH:
                    return FlextResult[str].fail(
                        f"Password cannot exceed {FlextConstants.Security.MAX_PASSWORD_LENGTH} characters"
                    )
                return FlextResult[str].ok(password)

            # BCrypt password hashing
            def hash_password(password: str) -> str:
                rounds = FlextConstants.Security.DEFAULT_BCRYPT_ROUNDS
                salt = bcrypt.gensalt(rounds=rounds)
                return bcrypt.hashpw(password.encode(), salt).decode()

            # Session management
            def create_session(user_id: str, extended: bool = False) -> dict:
                duration_hours = (
                    FlextConstants.Security.MAX_SESSION_EXTENSION_HOURS if extended
                    else FlextConstants.Security.DEFAULT_SESSION_HOURS
                )
                return {
                    "user_id": user_id,
                    "expires_at": datetime.utcnow() + timedelta(hours=duration_hours),
                    "extended": extended
                }
        """

        # JWT token expiry limits
        MAX_JWT_EXPIRY_MINUTES: Final[int] = 43200  # 30 days in minutes
        DEFAULT_JWT_EXPIRY_MINUTES: Final[int] = 1440  # 24 hours in minutes

        # Session management
        MAX_SESSION_EXTENSION_HOURS: Final[int] = 168  # 7 days maximum
        DEFAULT_SESSION_HOURS: Final[int] = 24  # 24 hours default

        # Password and hash validation
        MIN_BCRYPT_HASH_LENGTH: Final[int] = 32  # Minimum bcrypt hash length
        MIN_PASSWORD_LENGTH: Final[int] = 8  # Minimum password length
        MAX_PASSWORD_LENGTH: Final[int] = 128  # Maximum password length
        MAX_DAYS_FOR_MONTH_ADDITION: Final[int] = 28  # Safe days for month calculations

        # JWT token expiry limits - Additional short-term option
        SHORT_JWT_EXPIRY_MINUTES: Final[int] = 60  # 1 hour for short sessions

        # BCrypt configuration
        DEFAULT_BCRYPT_ROUNDS: Final[int] = 12
        MIN_BCRYPT_ROUNDS: Final[int] = 10  # Minimum acceptable rounds
        MAX_BCRYPT_ROUNDS: Final[int] = 15  # Maximum reasonable rounds

        # Default JWT secret for development/testing
        DEFAULT_JWT_SECRET: Final[str] = "default-jwt-secret-change-in-production"

    class Environment:
        """Environment enumerations used by configuration profiles."""

        class ConfigEnvironment(StrEnum):
            """Enumerate core deployment environments in docs."""

            DEVELOPMENT = "development"  # Usage count: 1
            STAGING = "staging"  # Usage count: 0
            PRODUCTION = "production"  # Usage count: 1
            TESTING = "test"  # Usage count: 0

        class ValidationLevel(StrEnum):
            """Validation strictness tiers adopted by tooling."""

            STRICT = "strict"  # Usage count: 0
            NORMAL = "normal"  # Usage count: 0
            RELAXED = "relaxed"  # Usage count: 0

    class Cqrs:
        """CQRS (Command Query Responsibility Segregation) constants."""

        # Handler types - Using Literal for type safety
        DEFAULT_HANDLER_TYPE: Final = "command"
        COMMAND_HANDLER_TYPE: Final = "command"

        # CQRS handler type literals for type annotations
        class HandlerTypeLiteral:
            """CQRS handler type literals for type annotations."""

            COMMAND: Final[str] = "command"
            QUERY: Final[str] = "query"
            EVENT: Final[str] = "event"
            SAGA: Final[str] = "saga"

        # Processing mode literals for type annotations
        class ProcessingModeLiteral:
            """Processing mode literals for type annotations."""

            BATCH: Final[str] = "batch"
            STREAM: Final[str] = "stream"
            PARALLEL: Final[str] = "parallel"
            SEQUENTIAL: Final[str] = "sequential"

        # Processing status literals for type annotations
        class ProcessingStatusLiteral:
            """Processing status literals for type annotations."""

            PENDING: Final[str] = "pending"
            RUNNING: Final[str] = "running"
            COMPLETED: Final[str] = "completed"
            FAILED: Final[str] = "failed"
            CANCELLED: Final[str] = "cancelled"

        # Validation level literals for type annotations
        class ValidationLevelLiteral:
            """Validation level literals for type annotations."""

            STRICT: Final[str] = "strict"
            LENIENT: Final[str] = "lenient"
            STANDARD: Final[str] = "standard"

        # Processing phase literals for type annotations
        class ProcessingPhaseLiteral:
            """Processing phase literals for type annotations."""

            PREPARE: Final[str] = "prepare"
            EXECUTE: Final[str] = "execute"
            VALIDATE: Final[str] = "validate"
            COMPLETE: Final[str] = "complete"

        # Model-specific literal types (moved from models.py)
        class BindTypeLiteral:
            """Bind type literals for model annotations."""

            TEMPORARY: Final[str] = "temporary"
            PERMANENT: Final[str] = "permanent"

        class MergeStrategyLiteral:
            """Merge strategy literals for model annotations."""

            REPLACE: Final[str] = "replace"
            UPDATE: Final[str] = "update"
            MERGE_DEEP: Final[str] = "merge_deep"

        class StatusLiteral:
            """Common status literals for model annotations."""

            PENDING: Final[str] = "pending"
            RUNNING: Final[str] = "running"
            COMPLETED: Final[str] = "completed"
            FAILED: Final[str] = "failed"
            COMPENSATING: Final[str] = "compensating"

        class HealthStatusLiteral:
            """Health status literals for monitoring."""

            HEALTHY: Final[str] = "healthy"
            DEGRADED: Final[str] = "degraded"
            UNHEALTHY: Final[str] = "unhealthy"

        class TokenTypeLiteral:
            """Token type literals for authentication."""

            BEARER: Final[str] = "bearer"
            API_KEY: Final[str] = "api_key"
            JWT: Final[str] = "jwt"

        class CircuitBreakerStateLiteral:
            """Circuit breaker state literals."""

            CLOSED: Final[str] = "closed"
            OPEN: Final[str] = "open"
            HALF_OPEN: Final[str] = "half_open"

        class NotificationStatusLiteral:
            """Notification status literals."""

            PENDING: Final[str] = "pending"
            SENT: Final[str] = "sent"
            FAILED: Final[str] = "failed"

        class TokenStatusLiteral:
            """Token status literals."""

            PENDING: Final[str] = "pending"
            RUNNING: Final[str] = "running"
            COMPLETED: Final[str] = "completed"
            FAILED: Final[str] = "failed"

        class CircuitBreakerStatusLiteral:
            """Circuit breaker status literals."""

            IDLE: Final[str] = "idle"
            RUNNING: Final[str] = "running"
            COMPLETED: Final[str] = "completed"
            FAILED: Final[str] = "failed"

        class BatchStatusLiteral:
            """Batch processing status literals."""

            PENDING: Final[str] = "pending"
            PROCESSING: Final[str] = "processing"
            COMPLETED: Final[str] = "completed"
            FAILED: Final[str] = "failed"

        class ExportStatusLiteral:
            """Export status literals."""

            PENDING: Final[str] = "pending"
            PROCESSING: Final[str] = "processing"
            COMPLETED: Final[str] = "completed"
            FAILED: Final[str] = "failed"

        class OperationStatusLiteral:
            """Operation status literals."""

            SUCCESS: Final[str] = "success"
            FAILURE: Final[str] = "failure"
            PARTIAL: Final[str] = "partial"

        class SerializationFormatLiteral:
            """Serialization format literals."""

            JSON: Final[str] = "json"
            YAML: Final[str] = "yaml"
            TOML: Final[str] = "toml"
            MSGPACK: Final[str] = "msgpack"

        class CompressionLiteral:
            """Compression type literals."""

            NONE: Final[str] = "none"
            GZIP: Final[str] = "gzip"
            BZIP2: Final[str] = "bzip2"
            LZ4: Final[str] = "lz4"

        class AggregationLiteral:
            """Aggregation function literals."""

            SUM: Final[str] = "sum"
            AVG: Final[str] = "avg"
            MIN: Final[str] = "min"
            MAX: Final[str] = "max"
            COUNT: Final[str] = "count"

        class ActionLiteral:
            """Action type literals."""

            GET: Final[str] = "get"
            CREATE: Final[str] = "create"
            UPDATE: Final[str] = "update"
            DELETE: Final[str] = "delete"
            LIST: Final[str] = "list"

        class PersistenceLevelLiteral:
            """Persistence level literals."""

            MEMORY: Final[str] = "memory"
            DISK: Final[str] = "disk"
            DISTRIBUTED: Final[str] = "distributed"

        class TargetFormatLiteral:
            """Target format literals."""

            FULL: Final[str] = "full"
            COMPACT: Final[str] = "compact"
            MINIMAL: Final[str] = "minimal"

        class WarningLevelLiteral:
            """Warning level literals."""

            NONE: Final[str] = "none"
            WARN: Final[str] = "warn"
            ERROR: Final[str] = "error"

        class OutputFormatLiteral:
            """Output format literals."""

            DICT: Final[str] = "dict"
            JSON: Final[str] = "json"

        class ModeLiteral:
            """Mode literals for various operations."""

            VALIDATION: Final[str] = "validation"
            SERIALIZATION: Final[str] = "serialization"

        class RegistrationStatusLiteral:
            """Registration status literals."""

            ACTIVE: Final[str] = "active"
            INACTIVE: Final[str] = "inactive"

        QUERY_HANDLER_TYPE: Final = "query"
        EVENT_HANDLER_TYPE: Final = "event"
        SAGA_HANDLER_TYPE: Final = "saga"

        # Command/Query defaults
        DEFAULT_COMMAND_TYPE: Final[str] = (
            ""  # Empty string for unspecified command type
        )
        DEFAULT_TIMESTAMP: Final[str] = ""  # Empty string for uninitialized timestamps
        DEFAULT_PRIORITY: Final[int] = 0  # Default priority level
        MAX_PRIORITY: Final[int] = 100  # Maximum priority value
        MIN_PRIORITY: Final[int] = 0  # Minimum priority value

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

        # FlextResult error messages
        FIRST_ARG_FAILED_MSG: Final[str] = "First argument failed"
        SECOND_ARG_FAILED_MSG: Final[str] = "Second argument failed"

    class Context:
        """Context management constants for FlextContext operations.

        Centralized constants for context scope management, correlation IDs,
        timeouts, and context depth limits. Used by FlextContext and related
        context management utilities across the ecosystem.

        Usage example:
            from flext_core import FlextConstants, FlextContext

            context = FlextContext()
            context.set("user_id", "123", scope=FlextConstants.Context.SCOPE_GLOBAL)
            context.set("request_id", "req-456", scope=FlextConstants.Context.SCOPE_REQUEST)

            correlation_id = f"{FlextConstants.Context.CORRELATION_ID_PREFIX}{uuid.uuid4().hex[:8]}"
        """

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

    class Container:
        """Constants for FlextContainer dependency injection.

        Provides default values for container configuration including
        worker threads, caching, and service lifecycle management.
        """

        # Worker thread configuration
        DEFAULT_WORKERS: Final[int] = 4  # Default thread pool size for async operations
        MAX_WORKERS: Final[int] = 4  # Alias for DEFAULT_WORKERS (config.py uses this)
        MIN_WORKERS: Final[int] = 1  # Minimum worker threads allowed

        # Service lifecycle timeouts
        TIMEOUT_SECONDS: Final[int] = 30  # Default timeout for container operations
        MIN_TIMEOUT_SECONDS: Final[int] = 1  # Minimum timeout in seconds
        MAX_TIMEOUT_SECONDS: Final[int] = 300  # Maximum timeout in seconds (5 minutes)
        DEFAULT_STARTUP_TIMEOUT: Final[int] = 30  # Seconds
        DEFAULT_SHUTDOWN_TIMEOUT: Final[int] = 10  # Seconds

        # Caching defaults
        ENABLE_SERVICE_CACHE: Final[bool] = True
        MAX_CACHE_SIZE: Final[int] = 100

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

    class Pagination:
        """Pagination constants referenced by FlextModels.Pagination."""

        DEFAULT_PAGE_NUMBER: Final[int] = 1  # Default starting page (1-based)
        DEFAULT_PAGE_SIZE: Final[int] = 10  # Default page size
        MAX_PAGE_SIZE: Final[int] = 1000  # Maximum allowed page size
        MIN_PAGE_SIZE: Final[int] = 1  # Minimum page size

    class Mixins:
        """Constants for FlextMixins operations.

        USAGE: Centralized constants for mixin field names, states, and defaults.
        OPTIMIZATION: Eliminates magic strings and provides type-safe constants.
        """

        # Field Names
        FIELD_ID = "id"
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
        STATUS_PASS = "PASS"  # Not a password, status constant
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

        # Log Levels
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

        # Boolean string representations
        BOOL_TRUE_STRINGS: ClassVar[set[str]] = {"true", "1", "yes", "on", "enabled"}
        BOOL_FALSE_STRINGS: ClassVar[set[str]] = {"false", "0", "no", "off", "disabled"}
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

    class Http:
        """HTTP protocol constants for client and server operations.

        Shared HTTP primitives for flext-api (client) and flext-web (server).
        Eliminates duplication by providing single source of truth for HTTP constants.

        USAGE: Foundation for HTTP client and server implementations.
        OPTIMIZATION: Zero duplication - all HTTP constants centralized here.

        Usage example:
            from flext_core import FlextConstants

            # Status code validation
            if status_code in range(
                FlextConstants.Http.HTTP_SUCCESS_MIN,
                FlextConstants.Http.HTTP_SUCCESS_MAX + 1
            ):
                return "Success response"

            # Method validation
            if method in FlextConstants.Http.SAFE_METHODS:
                return "Safe HTTP method"

            # Port validation
            if port == FlextConstants.Http.HTTPS_PORT:
                return "Using HTTPS"
        """

        # HTTP Status Code Ranges
        HTTP_STATUS_MIN: Final[int] = 100
        HTTP_STATUS_MAX: Final[int] = 599
        HTTP_INFORMATIONAL_MIN: Final[int] = 100
        HTTP_INFORMATIONAL_MAX: Final[int] = 199
        HTTP_SUCCESS_MIN: Final[int] = 200
        HTTP_SUCCESS_MAX: Final[int] = 299
        HTTP_REDIRECTION_MIN: Final[int] = 300
        HTTP_REDIRECTION_MAX: Final[int] = 399
        HTTP_CLIENT_ERROR_MIN: Final[int] = 400
        HTTP_CLIENT_ERROR_MAX: Final[int] = 499
        HTTP_SERVER_ERROR_MIN: Final[int] = 500
        HTTP_SERVER_ERROR_MAX: Final[int] = 599

        # Common HTTP Status Codes
        HTTP_OK: Final[int] = 200
        HTTP_CREATED: Final[int] = 201
        HTTP_ACCEPTED: Final[int] = 202
        HTTP_NO_CONTENT: Final[int] = 204
        HTTP_MULTIPLE_CHOICES: Final[int] = 300
        HTTP_MOVED_PERMANENTLY: Final[int] = 301
        HTTP_FOUND: Final[int] = 302
        HTTP_BAD_REQUEST: Final[int] = 400
        HTTP_UNAUTHORIZED: Final[int] = 401
        HTTP_FORBIDDEN: Final[int] = 403
        HTTP_NOT_FOUND: Final[int] = 404
        HTTP_METHOD_NOT_ALLOWED: Final[int] = 405
        HTTP_CONFLICT: Final[int] = 409
        HTTP_INTERNAL_SERVER_ERROR: Final[int] = 500
        HTTP_NOT_IMPLEMENTED: Final[int] = 501
        HTTP_BAD_GATEWAY: Final[int] = 502
        HTTP_SERVICE_UNAVAILABLE: Final[int] = 503

        # HTTP Methods
        class Method:
            """HTTP method constants."""

            GET: Final[str] = "GET"
            POST: Final[str] = "POST"
            PUT: Final[str] = "PUT"
            DELETE: Final[str] = "DELETE"
            PATCH: Final[str] = "PATCH"
            HEAD: Final[str] = "HEAD"
            OPTIONS: Final[str] = "OPTIONS"
            TRACE: Final[str] = "TRACE"
            CONNECT: Final[str] = "CONNECT"

        # HTTP Method Sets
        SAFE_METHODS: ClassVar[set[str]] = {"GET", "HEAD", "OPTIONS"}
        IDEMPOTENT_METHODS: ClassVar[set[str]] = {
            "GET",
            "HEAD",
            "PUT",
            "DELETE",
            "OPTIONS",
        }
        METHODS_WITH_BODY: ClassVar[set[str]] = {"POST", "PUT", "PATCH"}

        # HTTP Literals for type safety
        class MethodLiteral:
            """HTTP method literals for type annotations."""

            GET: Final[str] = "GET"
            POST: Final[str] = "POST"
            PUT: Final[str] = "PUT"
            DELETE: Final[str] = "DELETE"
            PATCH: Final[str] = "PATCH"
            HEAD: Final[str] = "HEAD"
            OPTIONS: Final[str] = "OPTIONS"
            TRACE: Final[str] = "TRACE"
            CONNECT: Final[str] = "CONNECT"

        # HTTP Ports
        HTTP_PORT: Final[int] = 80
        HTTPS_PORT: Final[int] = 443
        HTTPS_ALT_PORT: Final[int] = 8443

        # Content Types
        class ContentType:
            """HTTP content type constants."""

            JSON: Final[str] = "application/json"
            XML: Final[str] = "application/xml"
            FORM: Final[str] = "application/x-www-form-urlencoded"
            MULTIPART: Final[str] = "multipart/form-data"
            TEXT: Final[str] = "text/plain"
            HTML: Final[str] = "text/html"
            BINARY: Final[str] = "application/octet-stream"
            PDF: Final[str] = "application/pdf"
            CSV: Final[str] = "text/csv"

        # Common HTTP Headers
        AUTHORIZATION_HEADER: Final[str] = "Authorization"
        CONTENT_TYPE_HEADER: Final[str] = "Content-Type"
        ACCEPT_HEADER: Final[str] = "Accept"
        USER_AGENT_HEADER: Final[str] = "User-Agent"
        CONTENT_LENGTH_HEADER: Final[str] = "Content-Length"

        # Default Header Values
        DEFAULT_USER_AGENT: Final[str] = "FLEXT/1.0.0"
        DEFAULT_CONTENT_TYPE: Final[str] = "application/json"
        DEFAULT_ACCEPT: Final[str] = "application/json"

    class Web:
        """Web application constants for client and server operations.

        Shared web primitives for flext-web and web-based integrations.
        Provides timeout and configuration constants for web applications.
        """

        class Timeout:
            """Web timeout configuration constants."""

            DEFAULT_TIMEOUT: Final[int] = 30  # Default web request timeout in seconds
            TOTAL_TIMEOUT: Final[int] = 300  # Maximum total timeout for web operations
            CONNECT_TIMEOUT: Final[int] = 10  # Connection timeout for web requests
            READ_TIMEOUT: Final[int] = 30  # Read timeout for web responses

    class Batch:
        """Batch processing constants for data operations.

        Shared batch processing primitives for all flext modules requiring
        batch operations. Provides size limits and configuration constants.
        """

        class Default:
            """Default batch processing configuration constants."""

            DEFAULT_SIZE: Final[int] = 1000  # Standard batch size for processing
            SMALL_SIZE: Final[int] = 100  # Small batch size for limited operations
            LARGE_SIZE: Final[int] = 10000  # Large batch size for bulk operations

    class Processing:
        """Processing pipeline constants for batch operations and workers."""

        DEFAULT_MAX_WORKERS: Final[int] = 4  # Default maximum worker threads
        DEFAULT_BATCH_SIZE: Final[int] = 1000  # Default batch size for processing

    class Paths:
        """File system paths for core operations."""

        CONFIG_DIR: Final[str] = "config"
        LOGS_DIR: Final[str] = "logs"
        CACHE_DIR: Final[str] = "cache"
        TEMP_DIR: Final[str] = "tmp"

    # =================================================================
    # TYPE ALIASES (Python 3.13+ Literal types from FlextConstants)
    # =================================================================

    # Bind type literals - reference nested class attributes correctly
    BindType = Literal[
        "temporary",
        "permanent",
    ]

    # Merge strategy literals - reference nested class attributes correctly
    MergeStrategy = Literal[
        "replace",
        "update",
        "merge_deep",
    ]

    # Status literals - reference nested class attributes correctly
    Status = Literal[
        "pending",
        "running",
        "completed",
        "failed",
        "compensating",
    ]

    # Health status literals - reference nested class attributes correctly
    HealthStatus = Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    # Token type literals - reference nested class attributes correctly
    TokenType = Literal[
        "bearer",
        "api_key",
        "jwt",
    ]

    # Notification status literals - reference nested class attributes correctly
    NotificationStatus = Literal[
        "pending",
        "sent",
        "failed",
    ]

    # Token status literals - reference nested class attributes correctly
    TokenStatus = Literal[
        "pending",
        "running",
        "completed",
        "failed",
    ]

    # Batch status literals - reference nested class attributes correctly
    BatchStatus = Literal[
        "pending",
        "processing",
        "completed",
        "failed",
    ]

    # Circuit breaker status literals - reference nested class attributes correctly
    CircuitBreakerStatus = Literal[
        "idle",
        "running",
        "completed",
        "failed",
    ]

    # Circuit breaker state literals - reference nested class attributes correctly
    CircuitBreakerState = Literal[
        "closed",
        "open",
        "half_open",
    ]


__all__ = ["FlextConstants"]
