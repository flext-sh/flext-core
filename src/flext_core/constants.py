"""Layer 0: Foundation constants backing the FLEXT-Core 1.0.0 modernization plan.

This module provides the foundational constants for the entire FLEXT ecosystem.
As Layer 0, it has NO dependencies and serves as the basis for all other modules.

Dependency Layer: 0 (Foundation - No Dependencies)
Used by: All other FlextCore modules and ecosystem projects

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Final


class FlextConstants:
    """Ecosystem-wide constant definitions for FLEXT foundation.

    FlextConstants provides the single source of truth for all constants
    across the entire FLEXT ecosystem. As Layer 0, this module has NO
    dependencies and serves as the foundation for all 32+ dependent
    projects. All constants are immutable using typing.Final.

    **Function**: Central constant definitions for ecosystem
        - Define error codes for exception categorization (50+ codes)
        - Provide configuration defaults (timeout, environment, etc.)
        - Establish validation limits (string length, ports, etc.)
        - Define logging constants (levels, formats, defaults)
        - Provide platform constants (OS, encoding, paths)
        - Support message and display constants
        - Enable telemetry and metrics defaults
        - Guarantee immutability with typing.Final

    **Uses**: Pure Python foundation (Layer 0 - no dependencies)
        - typing.Final for immutable constant definitions
        - enum.StrEnum for string enumeration types
        - typing.ClassVar for class-level constants
        - No external dependencies (pure Python stdlib)
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

    **TODO**: Enhanced constants for 1.0.0+ releases
        - [ ] Add runtime constant validation on module load
        - [ ] Implement dynamic constant loading from config
        - [ ] Support environment variable overrides for constants
        - [ ] Add constant versioning for API compatibility
        - [ ] Implement constant deprecation warnings
        - [ ] Add constant usage tracking and metrics
        - [ ] Support constant documentation generation
        - [ ] Implement constant type validation at runtime
        - [ ] Add constant groups for feature flags
        - [ ] Support constant internationalization (i18n)

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

    # Project metadata (REQUIRED - Foundation Library)
    CONSTANTS_VERSION = "1.0.0"
    PROJECT_PREFIX = "FLEXT_CORE"
    PROJECT_NAME = "FLEXT Core Foundation"

    class Core:
        """Core identifiers hardened for the 1.0.0 release cycle."""

        NAME: Final[str] = "FLEXT"  # Usage count: 1
        VERSION: Final[str] = "0.9.0"  # Usage count: 8
        DEFAULT_VERSION: Final[str] = "1.0.0"  # Default version for components

    class Network:
        """Network defaults shared across dispatcher-aligned services."""

        MIN_PORT: Final[int] = 1  # Usage count: 4
        MAX_PORT: Final[int] = 65535  # Usage count: 4
        TOTAL_TIMEOUT: Final[int] = 60  # Usage count: 0
        DEFAULT_TIMEOUT: Final[int] = 30  # Usage count: 4

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
        MIN_PHONE_DIGITS: Final[int] = (
            10  # Minimum phone number length  # Minimum phone number length
        )

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
        CONFIG_ERROR: Final[str] = "CONFIG_ERROR"  # Usage count: 1
        GENERIC_ERROR: Final[str] = "GENERIC_ERROR"  # Usage count: 3
        COMMAND_PROCESSING_FAILED: Final[str] = (
            "COMMAND_PROCESSING_FAILED"  # Usage count: 4
        )
        UNKNOWN_ERROR: Final[str] = "UNKNOWN_ERROR"  # Usage count: 1

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
        TIMEOUT_ERROR: Final[str] = "TIMEOUT_ERROR"  # Reserved for async operations
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

    class Messages:
        """User-facing validation and failure messages.

        Usage example:
            from flext_core import FlextConstants, FlextResult, FlextException

            def process_user_input(data: dict) -> FlextResult[dict]:
                if not data:
                    return FlextResult[dict].fail(
                        FlextConstants.Messages.NULL_DATA,
                        error_code="VALIDATION_ERROR"
                    )

                if "name" not in data or not data["name"]:
                    return FlextResult[dict].fail(
                        FlextConstants.Messages.VALUE_EMPTY,
                        error_code="VALIDATION_ERROR"
                    )

                try:
                    processed = transform_data(data)
                    return FlextResult[dict].ok(processed)
                except Exception as e:
                    return FlextResult[dict].fail(
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
        DEFAULT_ENVIRONMENT: Final[str] = "development"  # Usage count: 0
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

        FLEXT_API_PORT: Final[int] = 8000  # Usage count: 4
        DEFAULT_HOST: Final[str] = "localhost"  # Usage count: 0
        LOOPBACK_IP: Final[str] = "127.0.0.1"  # Localhost IP address

        # Database defaults
        POSTGRES_DEFAULT_PORT: Final[int] = 5432
        MYSQL_DEFAULT_PORT: Final[int] = 3306
        REDIS_DEFAULT_PORT: Final[int] = 6379
        MONGODB_DEFAULT_PORT: Final[int] = 27017
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

        # Validation patterns
        PATTERN_SECURITY_LEVEL: Final[str] = "^(low|standard|high|critical)$"
        PATTERN_DATA_CLASSIFICATION: Final[str] = (
            "^(public|internal|confidential|restricted)$"
        )
        PATTERN_EMAIL: Final[str] = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        PATTERN_PHONE_NUMBER: Final[str] = r"^\+?[1-9]\d{1,14}$"
        PATTERN_UUID: Final[str] = (
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )

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
        DEFAULT_MAX_DELAY_SECONDS: Final[float] = 60.0
        DEFAULT_INITIAL_DELAY_SECONDS: Final[float] = 1.0
        DEFAULT_RECOVERY_TIMEOUT: Final[int] = 60
        DEFAULT_FALLBACK_DELAY: Final[float] = 0.1

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

        # Rate limiting constants
        DEFAULT_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60  # 1 minute window
        DEFAULT_RATE_LIMIT_MAX_REQUESTS: Final[int] = 100  # Max requests per window

        # Circuit breaker constants
        DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5  # Open after failures
        DEFAULT_CIRCUIT_BREAKER_TIMEOUT_SECONDS: Final[int] = 60  # Recovery time

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
        DEFAULT_BCRYPT_ROUNDS: Final[int] = 12  # Secure default for password hashing
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
        QUERY_HANDLER_TYPE: Final = "query"
        EVENT_HANDLER_TYPE: Final = "event"
        SAGA_HANDLER_TYPE: Final = "saga"

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
        """Comprehensive logging configuration constants for FLEXT ecosystem.

        Provides centralized logging defaults, levels, formats, and configuration
        options used across all FLEXT modules and subprojects.

        Usage example:
            from flext_core import FlextConstants, FlextLogger

            # Configure logger with environment-specific settings
            def setup_logger(environment: str) -> FlextLogger:
                if environment == "production":
                    config = FlextConstants.Logging.Environment.PRODUCTION
                    level = FlextConstants.Logging.DEFAULT_LEVEL_PRODUCTION
                elif environment == "development":
                    config = FlextConstants.Logging.Environment.DEVELOPMENT
                    level = FlextConstants.Logging.DEFAULT_LEVEL_DEVELOPMENT
                else:
                    config = FlextConstants.Logging.Environment.STAGING
                    level = FlextConstants.Logging.DEFAULT_LEVEL

                logger = FlextLogger(__name__)
                logger.setLevel(level)
                return logger

            # Log with structured output and performance tracking
            logger = FlextLogger(__name__)
            if FlextConstants.Logging.TRACK_PERFORMANCE:
                start_time = time.time()
                # ... operation ...
                duration_ms = (time.time() - start_time) * 1000

                if duration_ms > FlextConstants.Logging.PERFORMANCE_THRESHOLD_CRITICAL:
                    logger.warning(
                        FlextConstants.Logging.Messages.PERFORMANCE_CRITICAL,
                        extra={"operation": "data_processing", "duration": duration_ms}
                    )

            # Module-specific logging level
            db_logger = FlextLogger("flext_db_oracle")
            db_logger.setLevel(FlextConstants.Logging.ModuleLevels.FLEXT_DB_ORACLE)

            # Structured logging with context
            if FlextConstants.Logging.INCLUDE_CONTEXT:
                logger.info(
                    FlextConstants.Logging.Messages.API_REQUEST,
                    extra={
                        "method": "POST",
                        "endpoint": "/api/users",
                        "correlation_id": "abc-123"
                    }
                )
        """

        # Log Levels - Standard hierarchy
        TRACE: Final[str] = "TRACE"
        DEBUG: Final[str] = "DEBUG"
        INFO: Final[str] = "INFO"
        WARNING: Final[str] = "WARNING"
        ERROR: Final[str] = "ERROR"
        CRITICAL: Final[str] = "CRITICAL"

        # Default log level for different environments
        DEFAULT_LEVEL: Final[str] = "INFO"
        DEFAULT_LEVEL_DEVELOPMENT: Final[str] = "DEBUG"
        DEFAULT_LEVEL_PRODUCTION: Final[str] = "WARNING"
        DEFAULT_LEVEL_TESTING: Final[str] = "INFO"

        # Valid log levels set
        VALID_LEVELS: Final[set[str]] = {
            "TRACE",
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        # Log Formatting
        DEFAULT_FORMAT: Final[str] = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        DETAILED_FORMAT: Final[str] = (
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
        JSON_FORMAT: Final[str] = "json"
        SIMPLE_FORMAT: Final[str] = "%(levelname)s - %(message)s"

        # Date formatting
        DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
        DATE_FORMAT_WITH_MS: Final[str] = "%Y-%m-%d %H:%M:%S.%f"

        # FlextLogger optimization constants for Pydantic models
        INCLUDE_SOURCE: Final[bool] = True
        STRUCTURED_OUTPUT: Final[bool] = True
        VERBOSITY: Final[str] = "detailed"
        VALID_VERBOSITY_LEVELS: Final[set[str]] = {"compact", "detailed", "full"}

        # Output configuration
        JSON_OUTPUT_DEFAULT: Final[None] = None  # Auto-detect
        JSON_OUTPUT_ENABLED: Final[bool] = True
        JSON_OUTPUT_DISABLED: Final[bool] = False

        # Performance tracking constants
        TRACK_PERFORMANCE: Final[bool] = True
        TRACK_MEMORY: Final[bool] = False
        TRACK_TIMING: Final[bool] = True
        ENABLE_TRACING: Final[bool] = False  # Distributed tracing disabled by default
        PERFORMANCE_THRESHOLD_WARNING: Final[float] = 1000.0  # milliseconds
        PERFORMANCE_THRESHOLD_CRITICAL: Final[float] = 5000.0  # milliseconds

        # Log rotation and file management
        MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB
        BACKUP_COUNT: Final[int] = 5
        ROTATION_WHEN: Final[str] = "midnight"
        MAX_CONTEXT_KEYS: Final[int] = 100  # Maximum context keys per log entry

        # Console output configuration
        CONSOLE_ENABLED: Final[bool] = True
        CONSOLE_DISABLED: Final[bool] = False
        CONSOLE_COLOR_ENABLED: Final[bool] = True
        CONSOLE_COLOR_DISABLED: Final[bool] = False

        # File output configuration
        FILE_ENABLED: Final[bool] = True
        FILE_DISABLED: Final[bool] = False
        DEFAULT_LOG_FILE: Final[str] = "flext.log"
        ERROR_LOG_FILE: Final[str] = "flext_error.log"

        # Network logging (syslog, remote logging)
        SYSLOG_ENABLED: Final[bool] = False
        REMOTE_LOGGING_ENABLED: Final[bool] = False
        DEFAULT_SYSLOG_FACILITY: Final[str] = "local0"

        # Context and metadata
        INCLUDE_CONTEXT: Final[bool] = True
        INCLUDE_CORRELATION_ID: Final[bool] = True
        INCLUDE_USER_ID: Final[bool] = False
        INCLUDE_SESSION_ID: Final[bool] = False

        # Filtering and sampling
        ENABLE_FILTERING: Final[bool] = True
        DEFAULT_SAMPLE_RATE: Final[float] = 1.0  # 100% sampling
        HIGH_VOLUME_SAMPLE_RATE: Final[float] = 0.1  # 10% sampling for high-volume logs

        # Security and compliance
        MASK_SENSITIVE_DATA: Final[bool] = True
        REDACT_PASSWORDS: Final[bool] = True
        REDACT_API_KEYS: Final[bool] = True
        REDACT_CREDIT_CARDS: Final[bool] = True

        # Environment-specific overrides
        class Environment:
            """Environment-specific logging configuration overrides."""

            DEVELOPMENT: Final[dict[str, object]] = {
                "level": "DEBUG",
                "console_enabled": True,
                "file_enabled": True,
                "structured_output": False,
                "include_source": True,
                "track_performance": True,
            }

            STAGING: Final[dict[str, object]] = {
                "level": "INFO",
                "console_enabled": True,
                "file_enabled": True,
                "structured_output": True,
                "include_source": True,
                "track_performance": True,
            }

            PRODUCTION: Final[dict[str, object]] = {
                "level": "WARNING",
                "console_enabled": False,
                "file_enabled": True,
                "structured_output": True,
                "include_source": False,
                "track_performance": False,
            }

            TESTING: Final[dict[str, object]] = {
                "level": "INFO",
                "console_enabled": True,
                "file_enabled": False,
                "structured_output": False,
                "include_source": True,
                "track_performance": False,
            }

        # Module-specific logging levels
        class ModuleLevels:
            """Default log levels for different FLEXT modules."""

            # Core modules
            FLEXT_CORE: Final[str] = "INFO"
            FLEXT_CONFIG: Final[str] = "WARNING"
            FLEXT_LOGGER: Final[str] = "WARNING"
            FLEXT_CONTAINER: Final[str] = "INFO"

            # API modules
            FLEXT_API: Final[str] = "INFO"
            FLEXT_WEB: Final[str] = "INFO"
            FLEXT_GRPC: Final[str] = "INFO"

            # Data modules
            FLEXT_DB_ORACLE: Final[str] = "WARNING"
            FLEXT_LDAP: Final[str] = "WARNING"
            FLEXT_LDIF: Final[str] = "WARNING"
            FLEXT_MELTANO: Final[str] = "INFO"

            # CLI modules
            FLEXT_CLI: Final[str] = "INFO"

            # Auth modules
            FLEXT_AUTH: Final[str] = "WARNING"

            # Target modules
            FLEXT_TARGET_ORACLE: Final[str] = "WARNING"
            FLEXT_TARGET_LDAP: Final[str] = "WARNING"
            FLEXT_TARGET_LDIF: Final[str] = "WARNING"

            # Tap modules
            FLEXT_TAP_ORACLE: Final[str] = "WARNING"
            FLEXT_TAP_LDAP: Final[str] = "WARNING"
            FLEXT_TAP_LDIF: Final[str] = "WARNING"

            # DBT modules
            FLEXT_DBT_ORACLE: Final[str] = "WARNING"
            FLEXT_DBT_LDAP: Final[str] = "WARNING"
            FLEXT_DBT_LDIF: Final[str] = "WARNING"

        # Log message templates
        class Messages:
            """Standard log message templates."""

            # Configuration messages
            CONFIG_LOADED: Final[str] = "Configuration loaded from {source}"
            CONFIG_VALIDATED: Final[str] = "Configuration validation completed"
            CONFIG_ERROR: Final[str] = "Configuration error: {error}"

            # Service messages
            SERVICE_STARTED: Final[str] = "Service {service_name} started"
            SERVICE_STOPPED: Final[str] = "Service {service_name} stopped"
            SERVICE_ERROR: Final[str] = "Service {service_name} error: {error}"

            # Database messages
            DB_CONNECTED: Final[str] = "Database connected to {database}"
            DB_DISCONNECTED: Final[str] = "Database disconnected from {database}"
            DB_QUERY_EXECUTED: Final[str] = "Database query executed: {query_type}"
            DB_ERROR: Final[str] = "Database error: {error}"

            # API messages
            API_REQUEST: Final[str] = "API request: {method} {endpoint}"
            API_RESPONSE: Final[str] = "API response: {status_code} {endpoint}"
            API_ERROR: Final[str] = "API error: {error}"

            # Authentication messages
            AUTH_SUCCESS: Final[str] = "Authentication successful for user {user}"
            AUTH_FAILED: Final[str] = "Authentication failed for user {user}"
            AUTH_ERROR: Final[str] = "Authentication error: {error}"

            # Performance messages
            PERFORMANCE_WARNING: Final[str] = (
                "Performance warning: {operation} took {duration}ms"
            )
            PERFORMANCE_CRITICAL: Final[str] = (
                "Performance critical: {operation} took {duration}ms"
            )

            # Error messages
            ERROR_OCCURRED: Final[str] = "Error occurred: {error}"
            CRITICAL_ERROR: Final[str] = "Critical error: {error}"
            VALIDATION_ERROR_MESSAGE: Final[str] = (
                "Validation error: {error}"  # Template referencing Errors.VALIDATION_ERROR
            )

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


__all__ = ["FlextConstants"]
