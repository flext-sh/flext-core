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
    ClassVar,
    Final,
    Literal,
)

from flext_core import __version__


class FlextConstants:
    """Foundation constants for the FLEXT ecosystem.

    Provides immutable constants organized in namespaces for configuration,
    validation, error handling, and system defaults. All constants use
    typing.Final for immutability and serve as the single source of truth.

    The class is organized into nested namespaces:
    - Core: Basic identifiers and version information
    - Network: Network-related defaults and limits
    - Validation: Input validation limits and patterns
    - Errors: Error codes for exception categorization
    - Messages: User-facing message templates
    - Config: Configuration defaults and limits
    - Platform: Platform-specific constants (HTTP, database, file types)
    - Logging: Logging configuration and levels
    - Security: Security and authentication constants

    Usage:
        >>> from flext_core import FlextConstants
        >>> error_code = FlextConstants.Errors.VALIDATION_ERROR
        >>> timeout = FlextConstants.Config.DEFAULT_TIMEOUT
    """

    """Core identifiers."""
    NAME: Final[str] = "FLEXT"
    VERSION: Final[str] = __version__

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
            >>> FlextConstants.Config.DEFAULT_TIMEOUT
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

    @classmethod
    def __class_getitem__(cls, key: str) -> object:
        """Class-level access for type annotations."""
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
        """Core identifiers and version information."""

        NAME: Final[str] = "FLEXT"
        VERSION: Final[str] = __version__
        ZERO: Final[int] = 0
        INITIAL_TIME: Final[float] = 0.0

    class Network:
        """Network-related defaults and limits."""

        MIN_PORT: Final[int] = 1  # Usage count: 4
        MAX_PORT: Final[int] = 65535  # Usage count: 4
        TOTAL_TIMEOUT: Final[int] = 60  # Usage count: 0
        DEFAULT_TIMEOUT: Final[int] = 30  # Usage count: 4
        DEFAULT_MAX_RETRIES: Final[int] = (
            3  # Default retry attempts for network operations
        )
        DEFAULT_CONNECTION_POOL_SIZE: Final[int] = 10  # Usage count: 1
        MAX_CONNECTION_POOL_SIZE: Final[int] = 100  # Usage count: 1
        MAX_CONNECTIONS: Final[int] = 100  # Maximum HTTP connections

    class Validation:
        """Input validation limits and patterns."""

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

        # System errors (reserved for critical failures)
        EXCEPTION_ERROR: Final[str] = (
            "EXCEPTION_ERROR"  # Reserved for exception wrapping
        )
        CRITICAL_ERROR: Final[str] = (
            "CRITICAL_ERROR"  # Reserved for system failures  # Usage count: 1
        )
        NONEXISTENT_ERROR: Final[str] = "NONEXISTENT_ERROR"

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
        """User-facing message templates."""

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
        """Entity-related constants."""

        ENTITY_ID_EMPTY: Final[str] = "Entity ID cannot be empty"  # Usage count: 0

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

    class Limits:
        """Upper bounds for resource usage and validation."""

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
        """Regex patterns for validation."""

        PATTERN_MESSAGE_ID_UNKNOWN: Final[str] = "unknown"
        PATTERN_MESSAGE_ROUTER: Final[str] = "message_router"
        PATTERN_STATUS_PROCESSED: Final[str] = "processed"

    class Config:
        """Configuration defaults and limits."""

        # Configuration validation constants (moved from FlextConfig.Constants)
        SEMANTIC_VERSION_MIN_PARTS: Final[int] = 3
        HIGH_TIMEOUT_THRESHOLD: Final[int] = 120
        MIN_WORKERS_FOR_HIGH_TIMEOUT: Final[int] = 4
        MAX_WORKERS_THRESHOLD: Final[int] = 50

        # Feature flags
        DEFAULT_ENABLE_CACHING: Final[bool] = True
        DEFAULT_ENABLE_METRICS: Final[bool] = False
        DEFAULT_ENABLE_TRACING: Final[bool] = False

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
        MIN_DATE_LENGTH: Final[int] = 10  # Minimum date format length (YYYY-MM-DD)
        MIN_TOKEN_EXPIRY_BUFFER: Final[int] = (
            60  # Minimum token expiry buffer in seconds
        )
        MAX_FUNCTIONS_THRESHOLD: Final[int] = (
            10  # Maximum functions before complexity increase
        )
        MAX_AST_DEPTH_THRESHOLD: Final[int] = (
            5  # Maximum AST depth before complexity increase
        )
        MAX_LINE_COUNT_THRESHOLD: Final[int] = (
            200  # Maximum lines before complexity increase
        )
        COMPLEXITY_SCORE_THRESHOLD: Final[float] = (
            0.7  # Threshold for high complexity modules
        )

        class ConfigSource(StrEnum):
            """Enumerate configuration origins supported by FlextConfig."""

            FILE = "file"  # Usage count: 0
            ENVIRONMENT = "env"  # Usage count: 0
            CLI = "cli"  # Usage count: 0

        class Environment(StrEnum):
            """Environment types for configuration."""

            DEVELOPMENT = "development"
            STAGING = "staging"
            PRODUCTION = "production"
            TESTING = "testing"
            LOCAL = "local"

    class Enums:
        """Shared enumeration types."""

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
        """Platform-specific constants for HTTP, database, and file types."""

        # Environment variable prefix for configuration
        ENV_PREFIX: Final[str] = "FLEXT_"
        ENV_FILE_DEFAULT: Final[str] = ".env"
        ENV_NESTED_DELIMITER: Final[str] = "__"

        FLEXT_API_PORT: Final[int] = 8000
        DEFAULT_HOST: Final[str] = "localhost"
        LOOPBACK_IP: Final[str] = "127.0.0.1"  # Localhost IP address
        DEFAULT_HTTP_PORT: Final[int] = 8080  # Alternative HTTP port

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

        # Common directory names
        DIR_LOGS: Final[str] = "logs"
        DIR_DATA: Final[str] = "data"
        DIR_CONFIG: Final[str] = "config"
        DIR_TEMP: Final[str] = "temp"
        DIR_CACHE: Final[str] = "cache"

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

    class Observability:
        """Observability and logging constants."""

        DEFAULT_LOG_LEVEL: Final[str] = "INFO"  # Usage count: 0

    class Performance:
        """Performance thresholds and operational limits."""

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

    class Security:
        """Security and authentication constants."""

        MAX_JWT_EXPIRY_MINUTES: Final[int] = 43200  # 30 days maximum
        DEFAULT_JWT_EXPIRY_MINUTES: Final[int] = 60  # 1 hour default

    class Logging:
        """Logging configuration constants."""

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

    class Cqrs:
        """CQRS pattern constants and configuration."""

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

        class ModelLiteral:
            """Model-related literal types."""

            # Handler types
            COMMAND: Final[str] = "command"
            QUERY: Final[str] = "query"
            EVENT: Final[str] = "event"
            SAGA: Final[str] = "saga"

            # Status types
            ACTIVE: Final[str] = "active"
            INACTIVE: Final[str] = "inactive"

            # Action types
            ACQUIRE: Final[str] = "acquire"
            RELEASE: Final[str] = "release"
            CHECK: Final[str] = "check"
            LIST_OBJECTS: Final[str] = "FlextTypes.List"

            # Warning types
            WARN_NONE: Final[str] = "none"
            WARN_WARN: Final[str] = "warn"
            WARN_ERROR: Final[str] = "error"

            # Mode types
            VALIDATION: Final[str] = "validation"
            SERIALIZATION: Final[str] = "serialization"

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

    class Container:
        """Dependency injection container constants."""

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
        """HTTP protocol constants."""

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
        """Web application constants."""

        DEFAULT_TIMEOUT: Final[int] = 30  # Default web request timeout in seconds
        TOTAL_TIMEOUT: Final[int] = 300  # Maximum total timeout for web operations
        CONNECT_TIMEOUT: Final[int] = 10  # Connection timeout for web requests
        READ_TIMEOUT: Final[int] = 30  # Read timeout for web responses

    class Batch:
        """Batch processing constants."""

        DEFAULT_SIZE: Final[int] = 1000  # Standard batch size for processing
        SMALL_SIZE: Final[int] = 100  # Small batch size for limited operations
        LARGE_SIZE: Final[int] = 10000  # Large batch size for bulk operations

    class Processing:
        """Processing pipeline constants."""

        DEFAULT_MAX_WORKERS: Final[int] = 4  # Default maximum worker threads
        DEFAULT_BATCH_SIZE: Final[int] = 1000  # Default batch size for processing
        MAX_BATCH_SIZE: Final[int] = 10000  # Maximum batch size for validation

    # Status literals - reference nested class attributes correctly
    Status = Literal[
        "pending",
        "running",
        "completed",
        "failed",
        "compensating",
    ]

    # Circuit breaker state literals - reference nested class attributes correctly
    CircuitBreakerState = Literal[
        "closed",
        "open",
        "half_open",
    ]

    # Model literal types - reference nested class attributes correctly
    HandlerType = Literal[
        "command",
        "query",
        "event",
        "saga",
    ]

    HandlerMode = Literal[
        "command",
        "query",
        "event",
        "saga",
    ]

    HandlerModeSimple = Literal[
        "command",
        "query",
    ]

    Compression = Literal[
        "none",
        "gzip",
        "bzip2",
        "lz4",
    ]

    # Error handling literals (from FlextTypes.ErrorHandling)
    ErrorCategory = Literal[
        "validation", "network", "database", "auth", "system", "unknown"
    ]
    ErrorSeverity = Literal["low", "medium", "high", "critical"]

    # Service literals (from FlextTypes.Service)
    ServiceType = Literal["instance", "factory", "singleton"]
    ServiceLifecycleState = Literal[
        "initializing", "ready", "running", "stopping", "stopped", "error"
    ]
    ServiceProtocol = Literal["http", "grpc", "websocket", "message_queue"]

    # Context literals (from FlextTypes.Context)
    ContextScope = Literal["global", "request", "session", "transaction"]
    ContextExportFormat = Literal["json", "dict"]

    # Logging literals (from FlextTypes.Logging)
    LoggingLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # Processing literals (from FlextTypes.Processing)
    ProcessingOutputFormat = Literal["json", "yaml", "table", "csv", "text", "xml"]
    ProcessingSerializationFormat = Literal["json", "yaml", "toml", "ini", "xml"]
    ProcessingCompressionFormat = Literal["gzip", "bzip2", "xz", "lzma"]

    # Project literals (from FlextTypes.Project)
    ProjectType = Literal[
        "flext-core",
        "flext-api",
        "flext-ldap",
        "flext-ldif",
        "flext-cli",
        "flext-auth",
        "flext-web",
        "flext-db-oracle",
        "flext-meltano",
        "flext-observability",
        "flext-grpc",
        "flext-oracle-wms",
        "flext-oracle-oic",
        "flext-target-ldap",
        "flext-target-ldif",
        "flext-target-oracle",
        "flext-target-oracle-oic",
        "flext-target-oracle-wms",
        "flext-tap-ldap",
        "flext-tap-ldif",
        "flext-tap-oracle",
        "flext-tap-oracle-oic",
        "flext-tap-oracle-wms",
        "flext-dbt-ldap",
        "flext-dbt-ldif",
        "flext-dbt-oracle",
        "flext-dbt-oracle-wms",
        "algar-oud-mig",
        "gruponos-meltano-native",
        "flexcore",
    ]
    ProjectStatus = Literal["active", "inactive", "deprecated", "archived"]

    # Workflow literals (from FlextTypes.Workflow)
    WorkflowProcessingStatus = Literal[
        "pending", "queued", "running", "completed", "failed", "cancelled"
    ]
    WorkflowProcessingMode = Literal["batch", "stream", "parallel", "sequential"]
    WorkflowValidationLevel = Literal["strict", "lenient", "standard"]
    WorkflowProcessingPhase = Literal["prepare", "execute", "validate", "complete"]
    WorkflowHandlerType = Literal["command", "query", "event", "processor"]
    WorkflowStatus = Literal[
        "created", "started", "running", "paused", "completed", "failed", "cancelled"
    ]
    WorkflowStepStatus = Literal[
        "pending", "running", "completed", "failed", "skipped", "cancelled"
    ]

    # CQRS literals (from FlextTypes.Cqrs)
    CqrsMode = Literal["command", "query", "event", "saga"]

    # Workspace literals (from FlextTypes.Workspace)
    WorkspaceStatus = Literal[
        "active", "inactive", "maintenance", "deprecated", "archived"
    ]


__all__ = ["FlextConstants"]
