"""FLEXT Core Constants - Foundation Layer Constants Management.

Centralized constants providing the foundational values used throughout all 32 projects
in the FLEXT ecosystem. Establishes standardized error codes, environment settings,
validation patterns, and system defaults that ensure consistent behavior across
the entire architectural stack.

Module Role in Architecture:
    Foundation Layer → Constants → All Other Layers

    This module provides essential constants that enable:
    - Standardized error codes across all FlextResult operations
    - Environment-aware configuration for multi-deployment scenarios
    - Validation patterns for data integrity across ecosystem projects
    - System defaults for timeouts, pagination, and performance tuning
    - Version management for ecosystem compatibility tracking

Constant Categories by Usage:
    Error Codes: Standardized classification for FlextResult error handling
    Environment Types: Development, staging, production deployment contexts
    Log Levels: Structured logging priorities aligned with enterprise monitoring
    Field Types: Data validation constants for domain model validation
    Timeout Values: Performance-tuned defaults for network and processing operations
    Validation Patterns: Regex patterns for email, URLs, and data format validation

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Error codes, environment types, validation patterns
    🔄 Enhancement: Observability constants (Enhancement Priority 2)
    📋 TODO Integration: Plugin architecture constants (Plugin Priority 3)

Ecosystem Integration:
    - Used by all 32 projects for consistent error classification
    - Enables standardized monitoring and alerting across deployments
    - Provides configuration defaults that work across different project types
    - Supports version compatibility checking for ecosystem updates

Usage Patterns Across Projects:
    # Standardized error handling in Singer taps/targets
    if connection_failed:
        return FlextResult.fail(ERROR_CODES["CONNECTION_ERROR"])

    # Environment-aware configuration in services
    if FlextEnvironment.PRODUCTION:
        timeout = DEFAULT_TIMEOUT * 2

    # Validation in domain entities
    if not EMAIL_PATTERN.match(email):
        return FlextResult.fail(ERROR_CODES["VALIDATION_ERROR"])

Design Principles:
    - Single source of truth eliminates constant duplication
    - Type-safe enums provide IDE support and runtime validation
    - Backward compatibility ensures smooth ecosystem updates
    - Performance-optimized for high-frequency access patterns
    - Hierarchical organization supports different access patterns

Quality Standards:
    - All constants must be used by multiple ecosystem projects
    - Error codes must align with FlextResult failure patterns
    - Environment constants must support production deployment needs
    - Validation patterns must be tested across real-world data

See Also:
    docs/python-module-organization.md: Constants architecture guidance
    src/flext_core/exceptions.py: Related error hierarchy implementation

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

# =============================================================================
# ERROR CODES - sem underscore conforme diretrizes
# =============================================================================

ERROR_CODES = {
    "GENERIC_ERROR": "GENERIC_ERROR",
    "VALIDATION_ERROR": "VALIDATION_ERROR",
    "TYPE_ERROR": "TYPE_ERROR",
    "UNWRAP_ERROR": "UNWRAP_ERROR",
    "NULL_DATA_ERROR": "NULL_DATA_ERROR",
    "EXPECTATION_ERROR": "EXPECTATION_ERROR",
    "INVALID_ARGUMENT": "INVALID_ARGUMENT",
    "OPERATION_ERROR": "OPERATION_ERROR",
    "FILTER_ERROR": "FILTER_ERROR",
    "EXCEPTION_ERROR": "EXCEPTION_ERROR",
    "MULTIPLE_ERRORS": "MULTIPLE_ERRORS",
    "CONTEXT_ERROR": "CONTEXT_ERROR",
    "CONDITIONAL_ERROR": "CONDITIONAL_ERROR",
    "SIDE_EFFECT_ERROR": "SIDE_EFFECT_ERROR",
    "CHAIN_ERROR": "CHAIN_ERROR",
    "MAP_ERROR": "MAP_ERROR",
    "BIND_ERROR": "BIND_ERROR",
    # New specialized error codes
    "BUSINESS_RULE_ERROR": "BUSINESS_RULE_ERROR",
    "RETRY_ERROR": "RETRY_ERROR",
    "CIRCUIT_BREAKER_ERROR": "CIRCUIT_BREAKER_ERROR",
    "CONCURRENCY_ERROR": "CONCURRENCY_ERROR",
    "RESOURCE_ERROR": "RESOURCE_ERROR",
    "SERIALIZATION_ERROR": "SERIALIZATION_ERROR",
    "DATABASE_ERROR": "DATABASE_ERROR",
    "API_ERROR": "API_ERROR",
    "EVENT_ERROR": "EVENT_ERROR",
    "TIMEOUT_ERROR": "TIMEOUT_ERROR",
    "SECURITY_ERROR": "SECURITY_ERROR",
    "CONFIGURATION_ERROR": "CONFIGURATION_ERROR",
    # Additional error codes used in exceptions module
    "AUTH_ERROR": "AUTH_ERROR",
    "CONNECTION_ERROR": "CONNECTION_ERROR",
    "CRITICAL_ERROR": "CRITICAL_ERROR",
    "EXTERNAL_ERROR": "EXTERNAL_ERROR",
    "MIGRATION_ERROR": "MIGRATION_ERROR",
    "PERMISSION_ERROR": "PERMISSION_ERROR",
    "PROCESSING_ERROR": "PROCESSING_ERROR",
    "CHAINED_ERROR": "CHAINED_ERROR",
    "RETRYABLE_ERROR": "RETRYABLE_ERROR",
    "UNKNOWN_ERROR": "UNKNOWN_ERROR",
    # Operation-specific error codes
    "FALLBACK_FAILURE": "FALLBACK_FAILURE",
    "OPERATION_AND_FALLBACK_FAILURE": "OPERATION_AND_FALLBACK_FAILURE",
    "OPERATION_FAILURE": "OPERATION_FAILURE",
    "NOT_FOUND": "NOT_FOUND",
    "ALREADY_EXISTS": "ALREADY_EXISTS",
    "CONFIG_ERROR": "CONFIG_ERROR",
}

# =============================================================================
# MESSAGES - sem underscore conforme diretrizes
# =============================================================================

MESSAGES = {
    "UNKNOWN_ERROR": "Unknown error occurred",
    "FILTER_FAILED": "Filter condition not met",
    "VALIDATION_FAILED": "Validation failed",
    "OPERATION_FAILED": "Operation failed",
    "UNWRAP_FAILED": "Cannot unwrap failed result",
    "NULL_DATA": "Result data is None",
    "INVALID_INPUT": "Invalid input provided",
    "TYPE_MISMATCH": "Type mismatch error",
    # Consolidated empty validation messages
    "ENTITY_ID_EMPTY": "Entity ID cannot be empty",
    "SERVICE_NAME_EMPTY": "Service name cannot be empty",
    "NAME_EMPTY": "Name cannot be empty",
    "MESSAGE_EMPTY": "Message cannot be empty",
    "EVENT_TYPE_EMPTY": "Event type cannot be empty",
    "VALUE_EMPTY": "Value cannot be empty",
    # New specialized error messages
    "BUSINESS_RULE_VIOLATED": "Business rule violated",
    "RETRY_EXHAUSTED": "Retry attempts exhausted",
    "CIRCUIT_BREAKER_OPEN": "Circuit breaker is open",
    "CONCURRENT_MODIFICATION": "Concurrent modification detected",
    "RESOURCE_UNAVAILABLE": "Resource is unavailable",
    "SERIALIZATION_FAILED": "Serialization failed",
    "DATABASE_CONNECTION_FAILED": "Database connection failed",
    "API_CALL_FAILED": "API call failed",
    "EVENT_PROCESSING_FAILED": "Event processing failed",
    "OPERATION_TIMEOUT": "Operation timed out",
    "SECURITY_VIOLATION": "Security violation detected",
    "CONFIGURATION_INVALID": "Configuration is invalid",
}

# =============================================================================
# STATUS CODES - sem underscore conforme diretrizes
# =============================================================================

STATUS_CODES = {
    "SUCCESS": "SUCCESS",
    "FAILURE": "FAILURE",
    "PENDING": "PENDING",
    "PROCESSING": "PROCESSING",
    "CANCELLED": "CANCELLED",
}

# =============================================================================
# LOG LEVELS - eliminado para usar apenas FlextLogLevel Enum
# =============================================================================

# LOG_LEVELS removido - usar FlextLogLevel.get_numeric_levels() para compatibilidade

# =============================================================================
# VALIDATION RULES - sem underscore conforme diretrizes
# =============================================================================

VALIDATION_RULES = {
    "REQUIRED": "REQUIRED",
    "OPTIONAL": "OPTIONAL",
    "NULLABLE": "NULLABLE",
    "NON_EMPTY": "NON_EMPTY",
}

# =============================================================================
# ENUMS - sem underscore conforme diretrizes
# =============================================================================


class FlextLogLevel(Enum):
    """Log level constants for structured logging system.

    Provides standard log level constants for use throughout the FLEXT Core
    library. These levels correspond to standard logging practices with
    consistent naming and priority ordering.

    Log Level Hierarchy (lowest to highest priority):
        - TRACE: Most verbose debugging information for detailed tracing
        - DEBUG: General debugging information for development
        - INFO: General information about application flow
        - WARNING: Potentially harmful situations that don't stop execution
        - ERROR: Error events that allow application to continue
        - CRITICAL: Very severe error events that may abort the application

    Usage:
        logger.set_level(FlextLogLevel.DEBUG)
        if level == FlextLogLevel.ERROR:
            handle_error()
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRACE = "TRACE"

    def __hash__(self) -> int:
        """Hash based on enum value."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Support comparison with string values for test compatibility."""
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    @classmethod
    def get_numeric_levels(cls) -> dict[str, int]:
        """Get numeric level values for logging compatibility.

        Returns:
            Dictionary mapping level names to numeric values

        """
        return {
            "CRITICAL": 50,
            "ERROR": 40,
            "WARNING": 30,
            "INFO": 20,
            "DEBUG": 10,
            "TRACE": 5,
        }

    def get_numeric_value(self) -> int:
        """Get numeric value for this log level.

        Returns:
            Numeric priority value for this level

        """
        return self.get_numeric_levels()[self.value]


class FlextEnvironment(Enum):
    """Environment constants for application deployment contexts.

    Defines standard environment types for application configuration and
    deployment strategies. Used throughout the system for environment-specific
    behavior and configuration management.

    Environment Types:
        - DEVELOPMENT: Local development environment with debug features
        - STAGING: Pre-production environment for testing and validation
        - PRODUCTION: Live production environment with optimized settings
        - TESTING: Automated testing environment with test-specific configuration

    Usage:
        if get_environment() == FlextEnvironment.PRODUCTION:
            enable_performance_optimizations()
    """

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    STAGING = "staging"
    TESTING = "testing"


class FlextFieldType(Enum):
    """Field type constants for data validation and field definitions.

    Defines standard field types for validation, schema definition, and
    data processing operations. Used by validation system and field
    definition components for type consistency.

    Supported Field Types:
        - STRING: Text and string data with length constraints
        - INTEGER: Whole number values with range validation
        - FLOAT: Decimal number values with precision handling
        - BOOLEAN: True/false boolean values
        - DATE: Date values without time component
        - DATETIME: Date and time values with timezone support
        - UUID: Universally unique identifier format
        - EMAIL: Email address format with validation

    Usage:
        field = FlextFieldCore(
            field_type=FlextFieldType.EMAIL,
            pattern=EMAIL_PATTERN
        )
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    UUID = "uuid"
    EMAIL = "email"


# =============================================================================
# SEMANTIC CONSTANTS BASE - Clean Architecture Foundation
# =============================================================================


class FlextSemanticConstants:
    """Semantic constants base providing structured namespace management.

    Modern Python 3.13 constants system using class composition and semantic prefixes.
    Enables consistent constants management across the 32-project FLEXT ecosystem
    with fallback support and alias management for backward compatibility.

    Architecture:
        - Semantic prefix grouping for logical organization
        - Class composition for complex constant hierarchies
        - Type-safe access patterns with IDE support
        - Fallback and alias system for migration compatibility
        - Extensible base for project-specific constants
        - Zero namespace pollution with clean exports

    Design Principles:
        - Single source of truth for shared constants
        - Semantic grouping over alphabetical organization
        - Composition over inheritance for complex structures
        - Backward compatibility through alias system
        - Performance-optimized for high-frequency access
    """

    class Core:
        """Core system constants - fundamental values."""

        VERSION = "0.9.0"
        NAME = "flext-core"
        ECOSYSTEM_SIZE = 32
        PYTHON_MIN_VERSION = "3.13"

    class Errors:
        """Error classification constants - standardized error codes."""

        # Railway-oriented programming error codes
        GENERIC_ERROR = "GENERIC_ERROR"
        VALIDATION_ERROR = "VALIDATION_ERROR"
        TYPE_ERROR = "TYPE_ERROR"
        UNWRAP_ERROR = "UNWRAP_ERROR"
        NULL_DATA_ERROR = "NULL_DATA_ERROR"
        EXPECTATION_ERROR = "EXPECTATION_ERROR"
        INVALID_ARGUMENT = "INVALID_ARGUMENT"
        OPERATION_ERROR = "OPERATION_ERROR"
        FILTER_ERROR = "FILTER_ERROR"
        EXCEPTION_ERROR = "EXCEPTION_ERROR"
        MULTIPLE_ERRORS = "MULTIPLE_ERRORS"
        CONTEXT_ERROR = "CONTEXT_ERROR"
        CONDITIONAL_ERROR = "CONDITIONAL_ERROR"
        SIDE_EFFECT_ERROR = "SIDE_EFFECT_ERROR"
        CHAIN_ERROR = "CHAIN_ERROR"
        MAP_ERROR = "MAP_ERROR"
        BIND_ERROR = "BIND_ERROR"

        # Business logic error codes
        BUSINESS_RULE_ERROR = "BUSINESS_RULE_ERROR"
        RETRY_ERROR = "RETRY_ERROR"
        CIRCUIT_BREAKER_ERROR = "CIRCUIT_BREAKER_ERROR"
        CONCURRENCY_ERROR = "CONCURRENCY_ERROR"
        RESOURCE_ERROR = "RESOURCE_ERROR"
        SERIALIZATION_ERROR = "SERIALIZATION_ERROR"

        # Infrastructure error codes
        DATABASE_ERROR = "DATABASE_ERROR"
        API_ERROR = "API_ERROR"
        EVENT_ERROR = "EVENT_ERROR"
        TIMEOUT_ERROR = "TIMEOUT_ERROR"
        SECURITY_ERROR = "SECURITY_ERROR"
        CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
        AUTH_ERROR = "AUTH_ERROR"
        CONNECTION_ERROR = "CONNECTION_ERROR"
        CRITICAL_ERROR = "CRITICAL_ERROR"
        EXTERNAL_ERROR = "EXTERNAL_ERROR"
        MIGRATION_ERROR = "MIGRATION_ERROR"
        PERMISSION_ERROR = "PERMISSION_ERROR"
        PROCESSING_ERROR = "PROCESSING_ERROR"
        CHAINED_ERROR = "CHAINED_ERROR"
        RETRYABLE_ERROR = "RETRYABLE_ERROR"
        UNKNOWN_ERROR = "UNKNOWN_ERROR"

        # Operation-specific error codes
        FALLBACK_FAILURE = "FALLBACK_FAILURE"
        OPERATION_AND_FALLBACK_FAILURE = "OPERATION_AND_FALLBACK_FAILURE"
        OPERATION_FAILURE = "OPERATION_FAILURE"
        NOT_FOUND = "NOT_FOUND"
        ALREADY_EXISTS = "ALREADY_EXISTS"
        CONFIG_ERROR = "CONFIG_ERROR"

    class Messages:
        """User-facing message constants - standardized messages."""

        # Generic operation messages
        UNKNOWN_ERROR = "Unknown error occurred"
        FILTER_FAILED = "Filter condition not met"
        VALIDATION_FAILED = "Validation failed"
        OPERATION_FAILED = "Operation failed"
        UNWRAP_FAILED = "Cannot unwrap failed result"
        NULL_DATA = "Result data is None"
        INVALID_INPUT = "Invalid input provided"
        TYPE_MISMATCH = "Type mismatch error"

        # Entity validation messages
        ENTITY_ID_EMPTY = "Entity ID cannot be empty"
        SERVICE_NAME_EMPTY = "Service name cannot be empty"
        NAME_EMPTY = "Name cannot be empty"
        MESSAGE_EMPTY = "Message cannot be empty"
        EVENT_TYPE_EMPTY = "Event type cannot be empty"
        VALUE_EMPTY = "Value cannot be empty"

        # Business operation messages
        BUSINESS_RULE_VIOLATED = "Business rule violated"
        RETRY_EXHAUSTED = "Retry attempts exhausted"
        CIRCUIT_BREAKER_OPEN = "Circuit breaker is open"
        CONCURRENT_MODIFICATION = "Concurrent modification detected"
        RESOURCE_UNAVAILABLE = "Resource is unavailable"
        SERIALIZATION_FAILED = "Serialization failed"
        DATABASE_CONNECTION_FAILED = "Database connection failed"
        API_CALL_FAILED = "API call failed"
        EVENT_PROCESSING_FAILED = "Event processing failed"
        OPERATION_TIMEOUT = "Operation timed out"
        SECURITY_VIOLATION = "Security violation detected"
        CONFIGURATION_INVALID = "Configuration is invalid"

    class Status:
        """Status indicator constants - operation and process states."""

        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"
        PENDING = "PENDING"
        PROCESSING = "PROCESSING"
        CANCELLED = "CANCELLED"

    class Validation:
        """Validation rule constants - data integrity rules."""

        REQUIRED = "REQUIRED"
        OPTIONAL = "OPTIONAL"
        NULLABLE = "NULLABLE"
        NON_EMPTY = "NON_EMPTY"

    class Patterns:
        """Regex validation patterns - data format validation."""

        EMAIL = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        UUID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        URL = r"^https?://[^\s/$.?#].[^\s]*$"
        IDENTIFIER = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        SERVICE_NAME = r"^[a-zA-Z][a-zA-Z0-9_-]*$"

    class Defaults:
        """Default value constants - system defaults."""

        TIMEOUT = 30
        RETRIES = 3
        PAGE_SIZE = 100
        LOG_LEVEL = "INFO"

        # Entity and operation defaults
        ENTITY_VERSION = 1
        MAX_DOMAIN_EVENTS = 1000
        MAX_ENTITY_VERSION = 999999
        CONFIG_VERSION = 1

    class Limits:
        """System limit constants - constraints and boundaries."""

        MAX_DOMAIN_EVENTS = 1000
        MAX_ENTITY_VERSION = 999999
        MAX_STRING_LENGTH = 10000
        MAX_LIST_SIZE = 10000
        MAX_ID_LENGTH = 255

    class Performance:
        """Performance constant - optimization and tuning."""

        CACHE_SIZE_SMALL = 100
        CACHE_SIZE_LARGE = 1000
        TIMEOUT_SHORT = 5000  # ms
        TIMEOUT_LONG = 30000  # ms


class FlextConstants(FlextSemanticConstants):
    """Consolidated constants providing single source of truth.

    Central repository for all system constants, patterns, defaults, and configuration
    values. Eliminates base constant modules following the "deliver more with much less"
    principle while maintaining comprehensive constant management.

    Architecture:
        - Single source of truth for all constant definitions across the system
        - No underscore prefixes on public objects for clean API access
        - Nested classes for logical grouping and namespace organization
        - Direct access patterns for frequently used constants
        - Legacy compatibility through wrapper classes for smooth migration
        - Enterprise-grade constant management with validation and documentation

    Constant Categories:
        - ERROR_CODES: Standardized error classification codes for exception handling
        - MESSAGES: Default error and status messages for user communication
        - STATUS_CODES: Operation and process status indicators for workflow management
        - LOG_LEVELS: Logging level definitions with numeric values for priority
        - VALIDATION_RULES: Field validation rule identifiers for data integrity
        - Regex patterns: Common validation patterns for data format verification
        - Default values: System defaults for timeouts, sizes, and configuration
        - Project metadata: Version, name, and project information for identification

    Nested Classes:
        - Prefixes: Common prefixes for naming conventions and code organization
        - LogLevels: Legacy logging level access for backward compatibility
        - Performance: Performance-related constants and thresholds for optimization
        - Defaults: Default values for entities and operations with sensible fallbacks
        - Limits: System limits and constraints for validation and security

    Enterprise Features:
        - Comprehensive error classification supporting operational monitoring
        - Performance-tuned constants optimized for enterprise workloads
        - Validation patterns ensuring data integrity across the system
        - Environment-aware constants for multi-deployment scenarios
        - Version management support for release tracking and compatibility

    Usage Patterns:
        # Direct access to frequently used constants
        version = FlextConstants.VERSION
        email_pattern = FlextConstants.EMAIL_PATTERN

        # Nested class access for grouped constants
        cache_size = FlextConstants.Performance.CACHE_SIZE_LARGE
        max_events = FlextConstants.Limits.MAX_DOMAIN_EVENTS

        # Error handling with standardized codes
        return FlextResult.fail(
            FlextConstants.MESSAGES["VALIDATION_FAILED"],
            error_code=FlextConstants.ERROR_CODES["VALIDATION_ERROR"]
        )

        # Environment-specific configuration
        if env == FlextConstants.PRODUCTION:
            configure_production_settings()

        # Performance optimization
        cache = LRUCache(maxsize=FlextConstants.Performance.CACHE_SIZE_LARGE)
    """

    # Legacy compatibility - dictionary access patterns
    ERROR_CODES: ClassVar[dict[str, str]] = {
        name: getattr(FlextSemanticConstants.Errors, name)
        for name in dir(FlextSemanticConstants.Errors)
        if not name.startswith("_")
    }

    MESSAGES: ClassVar[dict[str, str]] = {
        name: getattr(FlextSemanticConstants.Messages, name)
        for name in dir(FlextSemanticConstants.Messages)
        if not name.startswith("_")
    }

    STATUS_CODES: ClassVar[dict[str, str]] = {
        name: getattr(FlextSemanticConstants.Status, name)
        for name in dir(FlextSemanticConstants.Status)
        if not name.startswith("_")
    }

    VALIDATION_RULES: ClassVar[dict[str, str]] = {
        name: getattr(FlextSemanticConstants.Validation, name)
        for name in dir(FlextSemanticConstants.Validation)
        if not name.startswith("_")
    }

    # Direct access patterns - frequently used constants
    EMAIL_PATTERN = FlextSemanticConstants.Patterns.EMAIL
    UUID_PATTERN = FlextSemanticConstants.Patterns.UUID
    URL_PATTERN = FlextSemanticConstants.Patterns.URL
    IDENTIFIER_PATTERN = FlextSemanticConstants.Patterns.IDENTIFIER
    SERVICE_NAME_PATTERN = FlextSemanticConstants.Patterns.SERVICE_NAME

    # Default value patterns
    DEFAULT_TIMEOUT = FlextSemanticConstants.Defaults.TIMEOUT
    DEFAULT_RETRIES = FlextSemanticConstants.Defaults.RETRIES
    DEFAULT_PAGE_SIZE = FlextSemanticConstants.Defaults.PAGE_SIZE
    DEFAULT_LOG_LEVEL = FlextSemanticConstants.Defaults.LOG_LEVEL

    # Project metadata
    VERSION = FlextSemanticConstants.Core.VERSION
    NAME = FlextSemanticConstants.Core.NAME

    class Prefixes:
        """Common prefixes used in the system."""

        PRIVATE_PREFIX = "_"
        INTERNAL_PREFIX = "__"
        PUBLIC_PREFIX = ""

    # LogLevels class removida - usar FlextLogLevel Enum diretamente

    class Performance:
        """Performance-related constants."""

        CACHE_SIZE_SMALL = 100
        CACHE_SIZE_LARGE = 1000
        TIMEOUT_SHORT = 5000  # ms
        TIMEOUT_LONG = 30000  # ms

    class Defaults:
        """Default values for entities and operations."""

        ENTITY_VERSION = 1
        MAX_DOMAIN_EVENTS = 1000
        MAX_ENTITY_VERSION = 999999
        CONFIG_VERSION = 1

    class Limits:
        """System limits and constraints."""

        MAX_DOMAIN_EVENTS = 1000
        MAX_ENTITY_VERSION = 999999
        MAX_STRING_LENGTH = 10000
        MAX_LIST_SIZE = 10000
        MAX_ID_LENGTH = 255

    # =============================================================================
    # PLATFORM CONSTANTS - Single source of truth for entire FLEXT ecosystem
    # =============================================================================

    class Platform:
        """Platform-wide constants for the entire FLEXT ecosystem."""

        # Service Ports
        FLEXCORE_PORT = 8080
        FLEXT_SERVICE_PORT = 8081
        FLEXT_API_PORT = 8000
        FLEXT_WEB_PORT = 3000
        FLEXT_GRPC_PORT = 50051

        # Infrastructure Ports
        POSTGRESQL_PORT = 5433
        REDIS_PORT = 6380
        MONITORING_PORT = 9090
        METRICS_PORT = 8090

        # Development Ports
        DEV_DB_PORT = 5432
        DEV_REDIS_PORT = 6379
        DEV_WEBHOOK_PORT = 8888

        # Hosts
        DEFAULT_HOST = "localhost"
        PRODUCTION_HOST = "localhost"  # Use specific host, not wildcard
        LOOPBACK_HOST = "127.0.0.1"

        # URLs
        DEFAULT_BASE_URL = f"http://{DEFAULT_HOST}"
        PRODUCTION_BASE_URL = "https://api.flext.io"

        # Database
        DB_MIN_CONNECTIONS = 1
        DB_MAX_CONNECTIONS = 10
        DB_CONNECTION_TIMEOUT = 30
        DB_QUERY_TIMEOUT = 60
        DEFAULT_POSTGRES_URL = (
            f"postgresql://flext:flext@{DEFAULT_HOST}:{POSTGRESQL_PORT}/flext"
        )
        DEFAULT_SQLITE_URL = "sqlite:///flext.db"

        # Cache
        REDIS_URL = f"redis://{DEFAULT_HOST}:{REDIS_PORT}/0"
        REDIS_TIMEOUT = 5
        CACHE_TTL_SHORT = 300  # 5 minutes
        CACHE_TTL_MEDIUM = 1800  # 30 minutes
        CACHE_TTL_LONG = 3600  # 1 hour
        CACHE_TTL_EXTENDED = 86400  # 24 hours

        # Security
        ACCESS_TOKEN_LIFETIME = 1800  # 30 minutes
        REFRESH_TOKEN_LIFETIME = 604800  # 7 days
        RATE_LIMIT_REQUESTS = 60  # requests per minute
        RATE_LIMIT_WINDOW = 60  # window in seconds
        MAX_LOGIN_ATTEMPTS = 5
        LOCKOUT_DURATION = 1800  # 30 minutes

        # Timeouts
        HTTP_CONNECT_TIMEOUT = 10
        HTTP_READ_TIMEOUT = 30
        HTTP_TOTAL_TIMEOUT = 60
        SERVICE_STARTUP_TIMEOUT = 30
        SERVICE_SHUTDOWN_TIMEOUT = 10

        # Validation
        MAX_NAME_LENGTH = 255
        MAX_DESCRIPTION_LENGTH = 1000
        MAX_FILE_SIZE = 10485760  # 10MB

        # Network
        MIN_PORT_NUMBER = 1
        MAX_PORT_NUMBER = 65535


# =============================================================================
# DIRECT ACCESS CONSTANTS - objetos sem underscore
# =============================================================================

# Core constants - acesso direto
VERSION = FlextConstants.VERSION
NAME = FlextConstants.NAME
DEFAULT_TIMEOUT = FlextConstants.DEFAULT_TIMEOUT
DEFAULT_RETRIES = FlextConstants.DEFAULT_RETRIES
DEFAULT_PAGE_SIZE = FlextConstants.DEFAULT_PAGE_SIZE
DEFAULT_LOG_LEVEL = FlextConstants.DEFAULT_LOG_LEVEL

# Patterns - acesso direto
EMAIL_PATTERN = FlextConstants.EMAIL_PATTERN
UUID_PATTERN = FlextConstants.UUID_PATTERN
URL_PATTERN = FlextConstants.URL_PATTERN
IDENTIFIER_PATTERN = FlextConstants.IDENTIFIER_PATTERN
SERVICE_NAME_PATTERN = FlextConstants.SERVICE_NAME_PATTERN

# =============================================================================
# LEGACY NESTED ACCESS PATTERNS (mantém interface existente)
# =============================================================================


class Project:
    """Project metadata constants for legacy compatibility.

    Backward compatibility class providing access to project metadata
    constants through legacy naming patterns.

    Legacy Usage:
        version = Project.VERSION
        name = Project.NAME
    """

    VERSION = FlextConstants.VERSION
    NAME = FlextConstants.NAME


class Environment:
    """Environment configuration constants for legacy compatibility.

    Backward compatibility class providing access to environment constants
    through legacy naming patterns and includes default environment setting.

    Legacy Usage:
        if environment == Environment.PRODUCTION:
            configure_production_settings()
    """

    PRODUCTION = FlextEnvironment.PRODUCTION
    DEVELOPMENT = FlextEnvironment.DEVELOPMENT
    STAGING = FlextEnvironment.STAGING
    TESTING = FlextEnvironment.TESTING
    DEFAULT = FlextEnvironment.DEVELOPMENT


class Defaults:
    """System default values for legacy compatibility.

    Backward compatibility class providing access to system default constants
    through legacy naming patterns for timeout, retry, and pagination settings.

    Legacy Usage:
        timeout = Defaults.TIMEOUT
        page_size = Defaults.PAGE_SIZE
    """

    TIMEOUT = FlextConstants.DEFAULT_TIMEOUT
    RETRIES = FlextConstants.DEFAULT_RETRIES
    PAGE_SIZE = FlextConstants.DEFAULT_PAGE_SIZE
    LOG_LEVEL = FlextConstants.DEFAULT_LOG_LEVEL


class Patterns:
    """Validation patterns for legacy compatibility.

    Backward compatibility class providing access to regex validation patterns
    through legacy naming conventions for common data format validation.

    Legacy Usage:
        if re.match(Patterns.EMAIL, email_address):
            process_email(email_address)
    """

    EMAIL = FlextConstants.EMAIL_PATTERN
    UUID = FlextConstants.UUID_PATTERN
    URL = FlextConstants.URL_PATTERN
    IDENTIFIER = FlextConstants.IDENTIFIER_PATTERN
    SERVICE_NAME = FlextConstants.SERVICE_NAME_PATTERN


# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_PAGE_SIZE",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "EMAIL_PATTERN",
    "ERROR_CODES",
    "IDENTIFIER_PATTERN",
    "MESSAGES",
    "NAME",
    "SERVICE_NAME_PATTERN",
    "STATUS_CODES",
    "URL_PATTERN",
    "UUID_PATTERN",
    "VALIDATION_RULES",
    "VERSION",
    "Defaults",
    "Environment",
    "FlextConstants",
    "FlextEnvironment",
    "FlextFieldType",
    "FlextLogLevel",
    "FlextSemanticConstants",
    "Patterns",
    "Project",
]
