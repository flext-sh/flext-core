"""Centralized constants and enumerations for the FLEXT core library.

Provides a comprehensive constant system for the FLEXT ecosystem, organized by
domain and functionality. Built for Python 3.13+ with strict typing enforcement
and hierarchical organization following Clean Architecture principles.

The module includes:
    - Hierarchical FlextConstants class organizing all system constants
    - Domain-specific constant categories (Core, Network, Validation, etc.)
    - Comprehensive error codes and status values
    - Type-safe enumerations for field types, log levels, and environments
    - Performance and infrastructure constants
    - Legacy compatibility layer for existing ecosystem

Examples:
    Basic usage with hierarchical constants::

        from flext_core.constants import FlextCoreConstants

        timeout = FlextCoreConstants.Defaults.TIMEOUT
        error_code = FlextCoreConstants.Errors.VALIDATION_ERROR
        log_level = FlextCoreConstants.Observability.DEFAULT_LOG_LEVEL

    Using enumerations for type safety::

        from flext_core.constants import FlextLogLevel, FlextEnvironment

        current_level = FlextLogLevel.INFO
        env = FlextEnvironment.PRODUCTION
        numeric_level = current_level.get_numeric_value()  # Returns 20

    Error handling with structured codes::

        from flext_core.constants import FlextCoreConstants, ERROR_CODES

        # Modern structured approach
        error = FlextCoreConstants.Errors.CONNECTION_ERROR
        message = FlextCoreConstants.Messages.DATABASE_CONNECTION_FAILED

        # Legacy flat mapping for backward compatibility
        legacy_error = ERROR_CODES["CONNECTION_ERROR"]

Note:
    This module enforces Python 3.13+ requirements and uses modern constant
    organization patterns. All constants are designed for strict type checking
    with mypy and pyright, following SOLID principles and Clean Architecture.
    Legacy flat constants are available for backward compatibility.

"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Final, override

# =============================================================================
# FLEXT HIERARCHICAL CONSTANTS SYSTEM - Organized by domain and functionality
# =============================================================================


class FlextCoreConstants:
    """Hierarchical constants system organizing FLEXT constants by domain and functionality.

    This class provides a structured organization of all constants used throughout
    the FLEXT ecosystem, grouped by domain and functionality for better
    maintainability, discoverability, and adherence to SOLID principles.

    The constants system is organized into the following domains:
        - Core: Fundamental system constants (name, version, architecture)
        - Network: Network-related constants (ports, protocols, timeouts)
        - Validation: Validation rules, patterns, and limits
        - Errors: Comprehensive error codes with structured hierarchy
        - Messages: User-facing and system messages
        - Status: Operation and entity status values
        - Patterns: Regular expressions for validation
        - Defaults: Default values for various system components
        - Limits: System limitations and boundaries
        - Performance: Performance-related constants and thresholds
        - Configuration: Configuration system constants
        - CLI: Command-line interface constants
        - Infrastructure: Database, cache, and external service constants
        - Models: Pydantic model configuration constants
        - Observability: Logging, monitoring, and tracing constants
        - Handlers: Handler system constants for CQRS patterns
        - Entities: Entity system constants for DDD patterns
        - ValidationSystem: Extended validation constants
        - InfrastructureMessages: Infrastructure layer messaging
        - Platform: FLEXT platform-specific constants

    Architecture Principles:
        - Single Responsibility: Each nested class has a single domain focus
        - Open/Closed: Easy to extend with new constant categories
        - Liskov Substitution: Consistent interface across all categories
        - Interface Segregation: Clients depend only on constants they use
        - Dependency Inversion: High-level constants don't depend on low-level details

    Examples:
        Using hierarchical constants for better organization::

            # Core system information
            app_name = FlextCoreConstants.Core.NAME
            version = FlextCoreConstants.Core.VERSION

            # Network configuration
            timeout = FlextCoreConstants.Network.DEFAULT_TIMEOUT
            port = FlextCoreConstants.Platform.FLEXT_SERVICE_PORT

            # Error handling
            error_code = FlextCoreConstants.Errors.VALIDATION_ERROR
            error_message = FlextCoreConstants.Messages.VALIDATION_FAILED

            # Performance tuning
            batch_size = FlextCoreConstants.Performance.DEFAULT_BATCH_SIZE
            threshold = FlextCoreConstants.Performance.SLOW_QUERY_THRESHOLD

        Type-safe constant access::

            # All constants are typed and validated
            max_retries: int = FlextCoreConstants.Defaults.MAX_RETRIES
            error_codes: dict[str, str] = FlextCoreConstants.Errors.MESSAGES
            log_levels: list[str] = FlextCoreConstants.Observability.LOG_LEVELS

    """

    # Class-level metadata for ecosystem compatibility
    ERROR_CODES: ClassVar[dict[str, str]] = {}  # Built at module level
    VERSION: Final[str] = "0.9.0"  # Legacy compatibility

    # =========================================================================
    # CORE CONSTANTS - Fundamental system constants
    # =========================================================================

    class Core:
        """Core fundamental constants used throughout the ecosystem.

        This class contains the most basic constants that form the foundation
        of the FLEXT system, including system identity, versioning, architecture
        information, and fundamental constraints following SOLID principles.

        Architecture Principles Applied:
            - Single Responsibility: Only core system metadata
            - Open/Closed: Easy to extend with new core constants
            - Interface Segregation: Core constants separated from operational ones
        """

        # System identity constants
        NAME: Final[str] = "FLEXT"
        VERSION: Final[str] = "0.9.0"
        ECOSYSTEM_SIZE: Final[int] = 33
        PYTHON_VERSION: Final[str] = "3.13+"
        ARCHITECTURE: Final[str] = "clean_architecture"

        # Code quality thresholds (replacing magic numbers)
        CONFIGURATION_ARGUMENT_INDEX_THRESHOLD: Final[int] = 2
        MAX_BRANCHES_ALLOWED: Final[int] = 12
        MAX_RETURN_STATEMENTS_ALLOWED: Final[int] = 6

        # Core patterns
        RAILWAY_PATTERN: Final[str] = "railway_oriented_programming"
        DI_PATTERN: Final[str] = "dependency_injection"
        DDD_PATTERN: Final[str] = "domain_driven_design"
        CQRS_PATTERN: Final[str] = "command_query_responsibility_segregation"

    # =========================================================================
    # NETWORK CONSTANTS - Network and connectivity constants
    # =========================================================================

    class Network:
        """Network and connectivity constants for the FLEXT ecosystem.

        This class contains network-related constants including port ranges,
        common service ports, protocol definitions, and networking defaults
        following separation of concerns.

        Architecture Principles Applied:
            - Single Responsibility: Only network-related constants
            - Interface Segregation: Network constants separated from application logic
        """

        # Port range boundaries
        MIN_PORT: Final[int] = 1
        MAX_PORT: Final[int] = 65535

        # Standard protocol ports
        HTTP_PORT: Final[int] = 80
        HTTPS_PORT: Final[int] = 443
        LDAP_PORT: Final[int] = 389
        LDAPS_PORT: Final[int] = 636

        # Protocol definitions
        HTTP_PROTOCOL: Final[str] = "http"
        HTTPS_PROTOCOL: Final[str] = "https"
        LDAP_PROTOCOL: Final[str] = "ldap"
        LDAPS_PROTOCOL: Final[str] = "ldaps"

        # Network timeouts (consolidated from various patterns)
        DEFAULT_TIMEOUT: Final[int] = 30
        CONNECTION_TIMEOUT: Final[int] = 10
        READ_TIMEOUT: Final[int] = 30
        TOTAL_TIMEOUT: Final[int] = 60

    # =========================================================================
    # VALIDATION CONSTANTS - Validation rules and limits
    # =========================================================================

    class Validation:
        """Validation constants for data integrity and business rules.

        This class contains validation-related constants including length limits,
        range boundaries, format requirements, and validation thresholds
        following the principle of explicit validation rules.

        Architecture Principles Applied:
            - Single Responsibility: Only validation constraints
            - Open/Closed: Easy to extend with new validation rules
            - Explicit Dependencies: Clear validation boundaries
        """

        # String length constraints
        MIN_SERVICE_NAME_LENGTH: Final[int] = 2
        MAX_SERVICE_NAME_LENGTH: Final[int] = 64
        MIN_SECRET_KEY_LENGTH: Final[int] = 32
        MIN_PASSWORD_LENGTH: Final[int] = 8
        MAX_PASSWORD_LENGTH: Final[int] = 128

        # Numeric range constraints
        MIN_PERCENTAGE: Final[float] = 0.0
        MAX_PERCENTAGE: Final[float] = 100.0
        MIN_PORT_NUMBER: Final[int] = 1
        MAX_PORT_NUMBER: Final[int] = 65535

        # Collection size limits
        MAX_LIST_SIZE: Final[int] = 10000
        MAX_BATCH_SIZE: Final[int] = 10000
        MIN_BATCH_SIZE: Final[int] = 1

        # File size limits
        MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB
        MAX_STRING_LENGTH: Final[int] = 1000

    # =========================================================================
    # ERROR CONSTANTS - Comprehensive error code hierarchy
    # =========================================================================

    class Errors:
        """Error codes and categorization for the FLEXT ecosystem.

        This class provides a comprehensive error code system organized by
        category with structured hierarchy following error handling best practices.
        Uses structured error codes with clear categorization for better
        error tracking, monitoring, and resolution.

        Error Code Structure:
            - FLEXT_XXXX: Structured numeric codes for key infrastructure errors
            - Category ranges: 1000-1999 (Business), 2000-2999 (Technical),
              3000-3999 (Validation), 4000-4999 (Security)
            - Legacy string codes: For backward compatibility with existing ecosystem

        Architecture Principles Applied:
            - Single Responsibility: Only error codes and categorization
            - Open/Closed: Easy to extend with new error categories
            - Liskov Substitution: Consistent error code interface
            - Dependency Inversion: Error codes don't depend on implementation details
        """

        # Error category ranges for structured error handling
        BUSINESS_ERROR_RANGE: Final[tuple[int, int]] = (1000, 1999)
        TECHNICAL_ERROR_RANGE: Final[tuple[int, int]] = (2000, 2999)
        VALIDATION_ERROR_RANGE: Final[tuple[int, int]] = (3000, 3999)
        SECURITY_ERROR_RANGE: Final[tuple[int, int]] = (4000, 4999)

        # Structured error codes (FLEXT_XXXX format)
        GENERIC_ERROR: Final[str] = "FLEXT_0001"
        VALIDATION_ERROR: Final[str] = "FLEXT_3001"
        BUSINESS_RULE_VIOLATION: Final[str] = "FLEXT_1001"
        AUTHORIZATION_DENIED: Final[str] = "FLEXT_4001"
        AUTHENTICATION_FAILED: Final[str] = "FLEXT_4002"
        RESOURCE_NOT_FOUND: Final[str] = "FLEXT_1004"
        DUPLICATE_RESOURCE: Final[str] = "FLEXT_1005"
        CONNECTION_ERROR: Final[str] = "FLEXT_2001"
        TIMEOUT_ERROR: Final[str] = "FLEXT_2002"
        CONFIGURATION_ERROR: Final[str] = "FLEXT_2003"
        SERIALIZATION_ERROR: Final[str] = "FLEXT_2004"
        EXTERNAL_SERVICE_ERROR: Final[str] = "FLEXT_2005"

        # Legacy compatibility codes (existing ecosystem dependencies)
        # Business Logic Errors
        BUSINESS_RULE_ERROR: Final[str] = (
            "BUSINESS_RULE_ERROR"  # Referenced by semantic.py
        )
        INVALID_ARGUMENT: Final[str] = "INVALID_ARGUMENT"
        EXPECTATION_ERROR: Final[str] = "EXPECTATION_ERROR"
        OPERATION_ERROR: Final[str] = "OPERATION_ERROR"
        OPERATION_FAILURE: Final[str] = "OPERATION_FAILURE"

        # Authentication & Authorization Errors
        AUTHENTICATION_ERROR: Final[str] = (
            "AUTHENTICATION_ERROR"  # Referenced by semantic.py
        )
        AUTH_ERROR: Final[str] = "AUTH_ERROR"
        PERMISSION_ERROR: Final[str] = "PERMISSION_ERROR"
        SECURITY_ERROR: Final[str] = "SECURITY_ERROR"

        # Technical System Errors
        TYPE_ERROR: Final[str] = "TYPE_ERROR"
        UNWRAP_ERROR: Final[str] = "UNWRAP_ERROR"
        NULL_DATA_ERROR: Final[str] = "NULL_DATA_ERROR"
        EXCEPTION_ERROR: Final[str] = "EXCEPTION_ERROR"
        CRITICAL_ERROR: Final[str] = "CRITICAL_ERROR"

        # Functional Programming Errors
        FILTER_ERROR: Final[str] = "FILTER_ERROR"
        MAP_ERROR: Final[str] = "MAP_ERROR"
        BIND_ERROR: Final[str] = "BIND_ERROR"
        CHAIN_ERROR: Final[str] = "CHAIN_ERROR"
        CHAINED_ERROR: Final[str] = "CHAINED_ERROR"

        # Resource & Infrastructure Errors
        RESOURCE_ERROR: Final[str] = "RESOURCE_ERROR"
        DATABASE_ERROR: Final[str] = "DATABASE_ERROR"
        API_ERROR: Final[str] = "API_ERROR"
        EXTERNAL_ERROR: Final[str] = "EXTERNAL_ERROR"
        CONFIG_ERROR: Final[str] = "CONFIG_ERROR"

        # Processing & Control Flow Errors
        PROCESSING_ERROR: Final[str] = "PROCESSING_ERROR"
        CONTEXT_ERROR: Final[str] = "CONTEXT_ERROR"
        CONDITIONAL_ERROR: Final[str] = "CONDITIONAL_ERROR"
        SIDE_EFFECT_ERROR: Final[str] = "SIDE_EFFECT_ERROR"
        MULTIPLE_ERRORS: Final[str] = "MULTIPLE_ERRORS"

        # Reliability Pattern Errors
        RETRY_ERROR: Final[str] = "RETRY_ERROR"
        RETRYABLE_ERROR: Final[str] = "RETRYABLE_ERROR"
        CIRCUIT_BREAKER_ERROR: Final[str] = "CIRCUIT_BREAKER_ERROR"
        FALLBACK_FAILURE: Final[str] = "FALLBACK_FAILURE"
        OPERATION_AND_FALLBACK_FAILURE: Final[str] = "OPERATION_AND_FALLBACK_FAILURE"

        # Domain & Event Errors
        EVENT_ERROR: Final[str] = "EVENT_ERROR"
        MIGRATION_ERROR: Final[str] = "MIGRATION_ERROR"
        CONCURRENCY_ERROR: Final[str] = "CONCURRENCY_ERROR"

        # Interface & CLI Errors
        CLI_ERROR: Final[str] = "CLI_ERROR"
        NOT_FOUND: Final[str] = "NOT_FOUND"
        ALREADY_EXISTS: Final[str] = "ALREADY_EXISTS"
        UNKNOWN_ERROR: Final[str] = "UNKNOWN_ERROR"

        # Error message mappings for structured error codes
        MESSAGES: Final[dict[str, str]] = {
            GENERIC_ERROR: "An error occurred",
            VALIDATION_ERROR: "Validation failed",
            BUSINESS_RULE_VIOLATION: "Business rule violation",
            AUTHORIZATION_DENIED: "Authorization denied",
            AUTHENTICATION_FAILED: "Authentication failed",
            RESOURCE_NOT_FOUND: "Resource not found",
            DUPLICATE_RESOURCE: "Resource already exists",
            CONNECTION_ERROR: "Connection failed",
            TIMEOUT_ERROR: "Operation timed out",
            CONFIGURATION_ERROR: "Configuration error",
            SERIALIZATION_ERROR: "Serialization failed",
            EXTERNAL_SERVICE_ERROR: "External service error",
        }

    # =========================================================================
    # MESSAGE CONSTANTS - User-facing and system messages
    # =========================================================================

    class Messages:
        """User-facing and system messages for the FLEXT ecosystem.

        This class contains standardized messages used throughout the system
        for user communication, logging, and error reporting following
        consistent messaging patterns.

        Architecture Principles Applied:
            - Single Responsibility: Only user-facing messages
            - Open/Closed: Easy to extend with new message categories
            - Interface Segregation: Messages separated by category
        """

        # Operation status messages
        SUCCESS: Final[str] = "Operation completed successfully"
        STARTED: Final[str] = "Operation started"
        COMPLETED: Final[str] = "Operation completed"
        FAILED: Final[str] = "Operation failed"
        VALIDATING: Final[str] = "Validating input"
        CONFIGURING: Final[str] = "Configuring system"

        # Error condition messages
        UNKNOWN_ERROR: Final[str] = "Unknown error occurred"
        FILTER_FAILED: Final[str] = "Filter condition not met"
        VALIDATION_FAILED: Final[str] = "Validation failed"
        OPERATION_FAILED: Final[str] = "Operation failed"
        UNWRAP_FAILED: Final[str] = "Cannot unwrap failed result"
        NULL_DATA: Final[str] = "Result data is None"
        INVALID_INPUT: Final[str] = "Invalid input provided"
        TYPE_MISMATCH: Final[str] = "Type mismatch error"

        # Entity validation messages
        ENTITY_ID_EMPTY: Final[str] = "Entity ID cannot be empty"
        SERVICE_NAME_EMPTY: Final[str] = "Service name cannot be empty"
        NAME_EMPTY: Final[str] = "Name cannot be empty"
        MESSAGE_EMPTY: Final[str] = "Message cannot be empty"
        EVENT_TYPE_EMPTY: Final[str] = "Event type cannot be empty"
        VALUE_EMPTY: Final[str] = "Value cannot be empty"

        # Business logic messages
        BUSINESS_RULE_VIOLATED: Final[str] = "Business rule violated"
        RETRY_EXHAUSTED: Final[str] = "Retry attempts exhausted"
        CIRCUIT_BREAKER_OPEN: Final[str] = "Circuit breaker is open"
        CONCURRENT_MODIFICATION: Final[str] = "Concurrent modification detected"

        # Infrastructure messages
        RESOURCE_UNAVAILABLE: Final[str] = "Resource is unavailable"
        SERIALIZATION_FAILED: Final[str] = "Serialization failed"
        DATABASE_CONNECTION_FAILED: Final[str] = "Database connection failed"
        API_CALL_FAILED: Final[str] = "API call failed"
        EVENT_PROCESSING_FAILED: Final[str] = "Event processing failed"
        OPERATION_TIMEOUT: Final[str] = "Operation timed out"
        SECURITY_VIOLATION: Final[str] = "Security violation detected"
        CONFIGURATION_INVALID: Final[str] = "Configuration is invalid"

    # =========================================================================
    # STATUS CONSTANTS - Operation and entity status values
    # =========================================================================

    class Status:
        """Status values for operations and entities in the FLEXT ecosystem.

        This class contains standardized status values used throughout the system
        for operation tracking, entity states, and workflow management.

        Architecture Principles Applied:
            - Single Responsibility: Only status value definitions
            - Interface Segregation: Status values separated by context
        """

        # Entity status values (lowercase for consistency)
        ACTIVE: Final[str] = "active"
        INACTIVE: Final[str] = "inactive"
        PENDING: Final[str] = "pending"
        COMPLETED: Final[str] = "completed"
        FAILED: Final[str] = "failed"
        RUNNING: Final[str] = "running"
        CANCELLED: Final[str] = "cancelled"

        # Operation status values (uppercase for emphasis)
        SUCCESS: Final[str] = "SUCCESS"
        FAILURE: Final[str] = "FAILURE"
        PROCESSING: Final[str] = "PROCESSING"

    # =========================================================================
    # PATTERN CONSTANTS - Regular expressions for validation
    # =========================================================================

    class Patterns:
        """Regular expression patterns for validation across the FLEXT ecosystem.

        This class contains comprehensive regex patterns used throughout the system
        for data validation, format checking, and input sanitization following
        security best practices and standard validation patterns.

        Architecture Principles Applied:
            - Single Responsibility: Only regex pattern definitions
            - Open/Closed: Easy to extend with new pattern categories
            - Interface Segregation: Patterns separated by domain
            - Security Focus: All patterns designed to prevent injection attacks
        """

        # Identifier patterns
        UUID_PATTERN: Final[str] = (
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        SLUG_PATTERN: Final[str] = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
        IDENTIFIER_PATTERN: Final[str] = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        SERVICE_NAME_PATTERN: Final[str] = r"^[a-zA-Z][a-zA-Z0-9_-]*$"

        # Authentication and user data patterns
        EMAIL_PATTERN: Final[str] = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        USERNAME_PATTERN: Final[str] = r"^[a-zA-Z0-9_-]{3,32}$"
        # Security pattern for credential strength validation (not a hardcoded password)
        CREDENTIAL_STRENGTH_PATTERN: Final[str] = (
            r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
        )

        # Network and address patterns
        IPV4_PATTERN: Final[str] = (
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        )
        HOSTNAME_PATTERN: Final[str] = (
            r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
            r"(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
        )
        URL_PATTERN: Final[str] = r"^https?://.+"

        # Version and release patterns
        SEMVER_PATTERN: Final[str] = (
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
            r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
            r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
            r"?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        )
        SEMANTIC_VERSION_PATTERN: Final[str] = r"^\d+\.\d+\.\d+(-\w+(\.\d+)?)?$"

    # =========================================================================
    # DEFAULT CONSTANTS - Default values for system components
    # =========================================================================

    class Defaults:
        """Default values for various system components in the FLEXT ecosystem.

        This class provides sensible default values used throughout the system
        for configuration, timeouts, pagination, and other operational parameters.
        Values are consolidated to eliminate duplication across components.

        Architecture Principles Applied:
            - Single Responsibility: Only default value definitions
            - DRY Principle: Eliminates duplication across components
            - Open/Closed: Easy to extend with new default categories
            - Configuration Over Convention: Explicit defaults instead of magic numbers
        """

        # Network and timeout defaults (consolidated from multiple sources)
        TIMEOUT: Final[int] = 30
        MAX_RETRIES: Final[int] = 3
        CONNECTION_TIMEOUT: Final[int] = 10

        # Pagination defaults
        PAGE_SIZE: Final[int] = 100
        MAX_PAGE_SIZE: Final[int] = 1000

        # Configuration system defaults (consolidated from Config pattern)
        CONFIG_TIMEOUT: Final[int] = 30
        CONFIG_RETRIES: Final[int] = 3

        # CLI interface defaults (consolidated from CLI pattern)
        CLI_HELP_WIDTH: Final[int] = 80
        CLI_TIMEOUT: Final[int] = 30

        # Model and validation defaults (consolidated from Pydantic pattern)
        MODEL_TIMEOUT: Final[int] = 30
        VALIDATION_TIMEOUT: Final[int] = 5

        # Database connection defaults
        DB_TIMEOUT: Final[int] = 30
        DB_POOL_SIZE: Final[int] = 10

        # Cache system defaults
        CACHE_TTL: Final[int] = 300  # 5 minutes
        CACHE_MAX_SIZE: Final[int] = 1000

    # =========================================================================
    # LIMIT CONSTANTS - System boundaries and constraints
    # =========================================================================

    class Limits:
        """System limits and boundaries for the FLEXT ecosystem.

        This class defines hard limits and constraints used throughout the system
        for resource protection, security boundaries, and operational constraints
        following security best practices and system stability requirements.

        Architecture Principles Applied:
            - Single Responsibility: Only system limit definitions
            - Security Focus: Limits prevent resource exhaustion attacks
            - Fail-Safe Defaults: Conservative limits for security and stability
        """

        # String and data size limits
        MAX_STRING_LENGTH: Final[int] = 1000
        MAX_LIST_SIZE: Final[int] = 10000
        MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB

        # Security and authentication limits
        MIN_PASSWORD_LENGTH: Final[int] = 8
        MAX_PASSWORD_LENGTH: Final[int] = 128

        # Processing and batch limits
        MAX_BATCH_SIZE: Final[int] = 10000
        MAX_THREADS: Final[int] = 100

        # Network limits (consolidated from Network class for security)
        MIN_PORT: Final[int] = 1
        MAX_PORT: Final[int] = 65535

    # =========================================================================
    # PERFORMANCE CONSTANTS - Performance tuning and monitoring
    # =========================================================================

    class Performance:
        """Performance-related constants for the FLEXT ecosystem.

        This class contains performance tuning parameters, thresholds, and
        monitoring intervals used throughout the system for optimal performance
        and resource utilization following performance engineering best practices.

        Architecture Principles Applied:
            - Single Responsibility: Only performance-related constants
            - Performance Focus: Tuned values for optimal system performance
            - Monitoring Integration: Constants support observability patterns
        """

        # Batch processing configuration
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        MAX_BATCH_SIZE: Final[int] = 10000
        BATCH_SIZE: Final[int] = 1000  # Alias for compatibility

        # Handler performance thresholds (in seconds)
        HANDLER_EXCELLENT_THRESHOLD: Final[float] = 0.1
        HANDLER_GOOD_THRESHOLD: Final[float] = 0.5
        HANDLER_FAIR_THRESHOLD: Final[float] = 1.0

        # Timeout configuration for performance operations
        TIMEOUT: Final[int] = 30  # Default timeout for performance operations
        KEEP_ALIVE_TIMEOUT: Final[int] = 60

        # Cache performance configuration
        CACHE_TTL: Final[int] = 300  # 5 minutes
        CACHE_MAX_SIZE: Final[int] = 1000

        # Connection pool configuration
        POOL_SIZE: Final[int] = 10
        MAX_CONNECTIONS: Final[int] = 100

        # Performance thresholds for monitoring and alerting
        SLOW_QUERY_THRESHOLD: Final[float] = 1.0  # seconds
        SLOW_REQUEST_THRESHOLD: Final[float] = 2.0  # seconds
        HIGH_MEMORY_THRESHOLD: Final[float] = 0.8  # 80%
        HIGH_CPU_THRESHOLD: Final[float] = 0.8  # 80%

        # Monitoring intervals
        METRICS_INTERVAL: Final[int] = 60  # seconds
        HEALTH_CHECK_INTERVAL: Final[int] = 30  # seconds

        # Time and size calculation constants (consolidated from utilities.py - SOLID SRP)
        SECONDS_PER_MINUTE: Final[int] = 60
        SECONDS_PER_HOUR: Final[int] = 3600
        BYTES_PER_KB: Final[int] = 1024

    # =========================================================================
    # CONFIGURATION CONSTANTS - Configuration system constants
    # =========================================================================

    class Configuration:
        """Configuration system constants for the FLEXT ecosystem.

        This class contains configuration management constants including provider
        priorities, file locations, environment types, and validation settings
        consolidated from various configuration patterns throughout the system.

        Architecture Principles Applied:
            - Single Responsibility: Only configuration system constants
            - Open/Closed: Easy to extend with new configuration sources
            - Dependency Inversion: Configuration abstractions independent of sources
        """

        # Configuration provider priority order (lower number = higher priority)
        CLI_PRIORITY: Final[int] = 1
        ENV_PRIORITY: Final[int] = 2
        DOTENV_PRIORITY: Final[int] = 3
        CONFIG_FILE_PRIORITY: Final[int] = 4
        CONSTANTS_PRIORITY: Final[int] = 5

        # Configuration file patterns and locations
        DOTENV_FILES: Final[list[str]] = [".env", ".internal.invalid", ".env.production"]
        CONFIG_FILES: Final[list[str]] = [
            "config.json",
            "config.yaml",
            "flext.config.json",
        ]

        # Environment type definitions
        ENVIRONMENTS: Final[list[str]] = [
            "development",
            "staging",
            "production",
            "test",
        ]
        DEFAULT_ENVIRONMENT: Final[str] = "development"

        # Configuration validation categories
        REQUIRED_FIELDS: Final[list[str]] = ["REQUIRED"]
        OPTIONAL_FIELDS: Final[list[str]] = ["OPTIONAL"]

    # =========================================================================
    # CLI CONSTANTS - Command-line interface constants
    # =========================================================================

    class Cli:
        """Command-line interface constants for the FLEXT ecosystem.

        This class contains CLI-related constants including argument patterns,
        output formats, exit codes, and interface standards consolidated from
        various CLI patterns throughout the system.

        Architecture Principles Applied:
            - Single Responsibility: Only CLI interface constants
            - Interface Segregation: CLI constants separated from application logic
            - Open/Closed: Easy to extend with new CLI features
        """

        # Standard CLI argument patterns
        HELP_ARGS: Final[list[str]] = ["--help", "-h"]
        VERSION_ARGS: Final[list[str]] = ["--version", "-v"]
        CONFIG_ARGS: Final[list[str]] = ["--config", "-c"]
        VERBOSE_ARGS: Final[list[str]] = ["--verbose", "-vv"]

        # CLI syntax prefixes
        LONG_PREFIX: Final[str] = "--"
        SHORT_PREFIX: Final[str] = "-"

        # Output format options
        OUTPUT_FORMATS: Final[list[str]] = ["json", "yaml", "table", "csv"]
        DEFAULT_OUTPUT_FORMAT: Final[str] = "table"

        # Standard exit codes following POSIX conventions
        SUCCESS_EXIT_CODE: Final[int] = 0
        ERROR_EXIT_CODE: Final[int] = 1
        INVALID_USAGE_EXIT_CODE: Final[int] = 2

    # =========================================================================
    # INFRASTRUCTURE CONSTANTS - Infrastructure and service constants
    # =========================================================================

    class Infrastructure:
        """Infrastructure constants for the FLEXT ecosystem.

        This class contains infrastructure-related constants including database
        ports, connection pools, network settings, and service configuration
        following infrastructure best practices.

        Architecture Principles Applied:
            - Single Responsibility: Only infrastructure configuration
            - Security Focus: Secure defaults and protocols
            - Scalability: Pool and connection management
        """

        # Standard database service ports
        DEFAULT_DB_PORT: Final[int] = 5432
        DEFAULT_ORACLE_PORT: Final[int] = 1521
        DEFAULT_REDIS_PORT: Final[int] = 6379
        DEFAULT_MYSQL_PORT: Final[int] = 3306
        DEFAULT_MONGODB_PORT: Final[int] = 27017

        # Connection pool configuration
        DEFAULT_POOL_SIZE: Final[int] = 10
        MAX_POOL_SIZE: Final[int] = 50
        MIN_POOL_SIZE: Final[int] = 1

        # Network and host configuration
        DEFAULT_HOST: Final[str] = "localhost"
        LOCALHOST_ALIASES: Final[list[str]] = ["localhost", "127.0.0.1", "::1"]
        DEFAULT_PROTOCOL: Final[str] = "http"
        SECURE_PROTOCOLS: Final[list[str]] = ["https", "wss", "ssl"]

    # =========================================================================
    # MODEL CONSTANTS - Pydantic model configuration constants
    # =========================================================================

    class Models:
        """Model system constants for the FLEXT ecosystem.

        This class contains Pydantic model configuration constants consolidated
        from various model patterns throughout the system for consistent
        model behavior and validation.

        Architecture Principles Applied:
            - Single Responsibility: Only model configuration constants
            - Consistency: Unified model behavior across ecosystem
            - Type Safety: Strong validation and serialization settings
        """

        # Validation behavior settings (consolidated from Pydantic patterns)
        VALIDATE_ASSIGNMENT: Final[bool] = True
        USE_ENUM_VALUES: Final[bool] = True
        STR_STRIP_WHITESPACE: Final[bool] = True

        # Serialization behavior settings (consolidated from Pydantic patterns)
        ARBITRARY_TYPES_ALLOWED: Final[bool] = True
        VALIDATE_DEFAULT: Final[bool] = True

        # Extra field handling modes (consolidated from Pydantic patterns)
        EXTRA_FORBID: Final[str] = "forbid"
        EXTRA_ALLOW: Final[str] = "allow"
        EXTRA_IGNORE: Final[str] = "ignore"

        # Supported field type definitions
        FIELD_TYPES: Final[list[str]] = [
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
        ]

        # Model lifecycle states
        MODEL_STATES: Final[list[str]] = ["draft", "valid", "invalid", "frozen"]

    # =========================================================================
    # OBSERVABILITY CONSTANTS - Logging, monitoring, and tracing constants
    # =========================================================================

    class Observability:
        """Observability constants for the FLEXT ecosystem.

        This class contains observability-related constants including logging levels,
        tracing configuration, metrics types, and monitoring headers consolidated
        from various observability patterns throughout the system.

        Architecture Principles Applied:
            - Single Responsibility: Only observability configuration
            - Monitoring Integration: Standards for distributed tracing
            - Consistency: Unified observability across all services
        """

        # Logging level hierarchy
        LOG_LEVELS: Final[list[str]] = [
            "TRACE",
            "DEBUG",
            "INFO",
            "WARN",
            "ERROR",
            "FATAL",
        ]
        DEFAULT_LOG_LEVEL: Final[str] = "INFO"

        # Trace level constant (consolidated from loggings.py - SOLID SRP)
        TRACE_LEVEL: Final[int] = 5

        # Observability type classifications
        SPAN_TYPES: Final[list[str]] = ["business", "technical", "error"]
        METRIC_TYPES: Final[list[str]] = ["counter", "histogram", "gauge"]
        ALERT_LEVELS: Final[list[str]] = ["info", "warning", "error", "critical"]

        # Distributed tracing headers (following OpenTelemetry standards)
        CORRELATION_ID_HEADER: Final[str] = "X-Correlation-ID"
        TRACE_ID_HEADER: Final[str] = "X-Trace-ID"

        # Serialization constants (consolidated from payload.py - SOLID SRP)
        FLEXT_SERIALIZATION_VERSION: Final[str] = "1.0.0"
        SERIALIZATION_FORMAT_JSON: Final[str] = "json"
        SERIALIZATION_FORMAT_JSON_COMPRESSED: Final[str] = "json_compressed"

    # NEW: Handler system constants (ADDED from string mapping analysis)
    class Handlers:
        """Handler system constants for command/query processing."""

        # Handler registration errors
        HANDLER_NOT_FOUND: Final[str] = "Handler not found"
        HANDLER_ALREADY_REGISTERED: Final[str] = "Handler already registered"
        HANDLER_NOT_CALLABLE: Final[str] = "Handler is not callable"
        NO_HANDLER_REGISTERED: Final[str] = "No handler registered for type"
        REGISTRY_NOT_FOUND: Final[str] = "No handlers registry found"

        # Permission and authorization
        MISSING_PERMISSION: Final[str] = "Missing permission"
        PERMISSION_DENIED: Final[str] = "Permission denied"
        AUTH_REQUIRED: Final[str] = "Authentication required"

        # Event processing
        EVENT_PROCESSING_FAILED: Final[str] = "Event processing failed"
        EVENT_HANDLER_FAILED: Final[str] = "Event handler failed"
        CHAIN_HANDLER_FAILED: Final[str] = "Chain handler failed"
        METRICS_COLLECTION_FAILED: Final[str] = "Metrics collection failed"

        # Handler validation errors
        REQUEST_CANNOT_BE_NONE: Final[str] = "Request cannot be None"
        MESSAGE_CANNOT_BE_NONE: Final[str] = "Message cannot be None"
        VALIDATION_FAILED: Final[str] = "Validation failed"
        EVENT_MISSING_TYPE: Final[str] = "Event missing event_type"
        NO_USER_IN_CONTEXT: Final[str] = "No user in context"
        INVALID_PERMISSIONS_FORMAT: Final[str] = "Invalid permissions format in context"
        HANDLER_NAME_EMPTY: Final[str] = "Handler name must be a non-empty string"
        INVALID_HANDLER_PROVIDED: Final[str] = "Invalid handler provided"
        HANDLER_NAME_MUST_BE_STRING: Final[str] = "Handler name must be a string"
        NO_HANDLER_COULD_PROCESS: Final[str] = "No handler could process the request"
        CHAIN_PROCESSING_FAILED: Final[str] = "Chain processing failed"
        NOT_IMPLEMENTED: Final[str] = "Not implemented"
        AUTHORIZATION_FAILED: Final[str] = "Authorization failed"
        QUERY_HANDLER_NOT_IMPLEMENTED: Final[str] = "Query handler not implemented"

        # Handler templates (for f-strings)
        HANDLER_NOT_FOUND_TEMPLATE: Final[str] = (
            "Handler '{name}' not found. Available: {available}"
        )
        MISSING_PERMISSION_TEMPLATE: Final[str] = "Missing permission: {permission}"
        EVENT_PROCESSING_FAILED_TEMPLATE: Final[str] = (
            "Event processing failed: {error}"
        )
        HANDLER_FAILED_TEMPLATE: Final[str] = "Handler failed: {error}"
        NO_HANDLER_FOR_TYPE_TEMPLATE: Final[str] = (
            "No handler registered for {type_name}"
        )
        CHAIN_HANDLER_FAILED_TEMPLATE: Final[str] = "Chain handler failed: {error}"
        METRICS_FAILED_TEMPLATE: Final[str] = "Metrics collection failed: {error}"
        NO_HANDLER_FOR_COMMAND_TEMPLATE: Final[str] = (
            "No handler registered for {command_type}"
        )
        NO_HANDLER_FOR_QUERY_TEMPLATE: Final[str] = (
            "No handler registered for {query_type}"
        )

    # NEW: Entity system constants (ADDED from string mapping analysis)
    class Entities:
        """Entity system constants for domain modeling."""

        # Entity validation
        ENTITY_ID_INVALID: Final[str] = "Invalid entity ID"
        ENTITY_ID_EMPTY: Final[str] = "Entity ID cannot be empty"
        ENTITY_NAME_EMPTY: Final[str] = "Entity name cannot be empty"
        ENTITY_VALIDATION_FAILED: Final[str] = "Entity validation failed"

        # Cache operations
        CACHE_KEY_TEMPLATE: Final[str] = "{class_name}:{id}"
        CACHE_KEY_HASH_TEMPLATE: Final[str] = "{class_name}:{hash}"

        # Entity operations
        OPERATION_LOG_TEMPLATE: Final[str] = "Operation: {operation}"
        ENTITY_CREATED: Final[str] = "Entity created"
        ENTITY_UPDATED: Final[str] = "Entity updated"
        ENTITY_DELETED: Final[str] = "Entity deleted"

        # Entity state
        ENTITY_ACTIVE: Final[str] = "Entity is active"
        ENTITY_INACTIVE: Final[str] = "Entity is inactive"
        ENTITY_STATE_INVALID: Final[str] = "Invalid entity state"

        # Entity templates
        INVALID_ENTITY_ID_TEMPLATE: Final[str] = "Invalid entity ID: {entity_id}"
        ENTITY_NOT_FOUND_TEMPLATE: Final[str] = "Entity not found: {entity_id}"
        ENTITY_OPERATION_TEMPLATE: Final[str] = "Entity {operation}: {entity_id}"

    # NEW: Validation system constants (EXPANDED from string mapping analysis)
    class ValidationSystem:
        """Extended validation constants beyond basic patterns."""

        # Error categories
        VALIDATION_ERROR_CATEGORY: Final[str] = "VALIDATION"
        BUSINESS_ERROR_CATEGORY: Final[str] = "BUSINESS"
        INFRASTRUCTURE_ERROR_CATEGORY: Final[str] = "INFRASTRUCTURE"
        CONFIGURATION_ERROR_CATEGORY: Final[str] = "CONFIGURATION"
        GENERAL_ERROR_CATEGORY: Final[str] = "GENERAL"

        # Value object validation
        VALUE_OBJECT_INVALID: Final[str] = "Value object validation failed"
        VALUE_OBJECT_EMPTY: Final[str] = "Value object cannot be empty"
        VALUE_OBJECT_FORMAT_INVALID: Final[str] = "Invalid value object format"

        # Type validation
        TYPE_MISMATCH: Final[str] = "Type mismatch error"
        TYPE_CONVERSION_FAILED: Final[str] = "Type conversion failed"
        TYPE_VALIDATION_FAILED: Final[str] = "Type validation failed"

        # Business rule validation
        BUSINESS_RULE_VIOLATED: Final[str] = "Business rule violated"
        DOMAIN_RULE_VIOLATION: Final[str] = "Domain rule violation"
        INVARIANT_VIOLATION: Final[str] = "Invariant violation"

        # Validation templates
        VALIDATION_ERROR_TEMPLATE: Final[str] = "Validation failed: {details}"
        BUSINESS_RULE_TEMPLATE: Final[str] = "Business rule violation: {rule}"
        TYPE_ERROR_TEMPLATE: Final[str] = (
            "Type error: expected {expected}, got {actual}"
        )

    # NEW: Infrastructure messaging constants (EXPANDED from string mapping analysis)
    class InfrastructureMessages:
        """Infrastructure layer messaging constants."""

        # Serialization
        SERIALIZATION_FAILED: Final[str] = "Serialization failed"
        DESERIALIZATION_FAILED: Final[str] = "Deserialization failed"
        SERIALIZATION_WARNING: Final[str] = "Serialization warning"

        # Delegation system
        DELEGATION_SUCCESS: Final[str] = "SUCCESS"
        DELEGATION_FAILED: Final[str] = "FAILED"
        DELEGATION_STATUS_TEMPLATE: Final[str] = "status: {status}"

        # Configuration
        CONFIG_LOADING: Final[str] = "Loading configuration"
        CONFIG_LOADED: Final[str] = "Configuration loaded"
        CONFIG_FAILED: Final[str] = "Configuration loading failed"
        CONFIG_FORMAT_JSON: Final[str] = "json"

        # Logging and monitoring
        OPERATION_STARTED: Final[str] = "Operation started"
        OPERATION_COMPLETED: Final[str] = "Operation completed"
        OPERATION_FAILED: Final[str] = "Operation failed"

        # Infrastructure templates
        SERIALIZATION_ERROR_TEMPLATE: Final[str] = "Serialization error: {error}"
        CONFIG_ERROR_TEMPLATE: Final[str] = "Configuration error: {details}"
        OPERATION_LOG_TEMPLATE: Final[str] = "Operation {operation}: {status}"

    # NEW: Platform constants (FLEXT specific infrastructure)
    class Platform:
        """Platform-wide constants."""

        # Service Ports
        FLEXCORE_PORT: Final[int] = 8080
        FLEXT_SERVICE_PORT: Final[int] = 8081
        FLEXT_API_PORT: Final[int] = 8000
        FLEXT_WEB_PORT: Final[int] = 3000
        FLEXT_GRPC_PORT: Final[int] = 50051

        # Infrastructure Ports
        POSTGRESQL_PORT: Final[int] = 5433
        REDIS_PORT: Final[int] = 6380
        MONITORING_PORT: Final[int] = 9090
        METRICS_PORT: Final[int] = 8090

        # Development Ports
        DEV_DB_PORT: Final[int] = 5432
        DEV_REDIS_PORT: Final[int] = 6379
        DEV_WEBHOOK_PORT: Final[int] = 8888

        # Hosts
        DEFAULT_HOST: Final[str] = "localhost"
        PRODUCTION_HOST: Final[str] = "localhost"  # Use specific host, not wildcard
        LOOPBACK_HOST: Final[str] = "127.0.0.1"

        # URLs
        DEFAULT_BASE_URL: Final[str] = f"http://{DEFAULT_HOST}"
        PRODUCTION_BASE_URL: Final[str] = "https://api.flext.io"

        # Database
        DB_MIN_CONNECTIONS: Final[int] = 1
        DB_MAX_CONNECTIONS: Final[int] = 10
        DB_CONNECTION_TIMEOUT: Final[int] = 30
        DB_QUERY_TIMEOUT: Final[int] = 60
        DEFAULT_POSTGRES_URL: Final[str] = (
            f"postgresql://flext:flext@{DEFAULT_HOST}:{POSTGRESQL_PORT}/flext"
        )
        DEFAULT_SQLITE_URL: Final[str] = "sqlite:///flext.db"

        # Cache
        REDIS_URL: Final[str] = f"redis://{DEFAULT_HOST}:{REDIS_PORT}/0"
        REDIS_TIMEOUT: Final[int] = 5
        CACHE_TTL_SHORT: Final[int] = 300  # 5 minutes
        CACHE_TTL_MEDIUM: Final[int] = 1800  # 30 minutes
        CACHE_TTL_LONG: Final[int] = 3600  # 1 hour
        CACHE_TTL_EXTENDED: Final[int] = 86400  # 24 hours

        # Security
        ACCESS_TOKEN_LIFETIME: Final[int] = 1800  # 30 minutes
        REFRESH_TOKEN_LIFETIME: Final[int] = 604800  # 7 days
        RATE_LIMIT_REQUESTS: Final[int] = 60  # requests per minute
        RATE_LIMIT_WINDOW: Final[int] = 60  # window in seconds
        MAX_LOGIN_ATTEMPTS: Final[int] = 5
        LOCKOUT_DURATION: Final[int] = 1800  # 30 minutes

        # Timeouts
        HTTP_CONNECT_TIMEOUT: Final[int] = 10
        HTTP_READ_TIMEOUT: Final[int] = 30
        HTTP_TOTAL_TIMEOUT: Final[int] = 60
        SERVICE_STARTUP_TIMEOUT: Final[int] = 30
        SERVICE_SHUTDOWN_TIMEOUT: Final[int] = 10

        # Validation
        MAX_NAME_LENGTH: Final[int] = 255
        MAX_DESCRIPTION_LENGTH: Final[int] = 1000
        MAX_FILE_SIZE: Final[int] = 10485760  # 10MB

        # Network
        MIN_PORT_NUMBER: Final[int] = 1
        MAX_PORT_NUMBER: Final[int] = 65535

    # Backward compatibility: Add commonly used constants as class attributes
    DEFAULT_TIMEOUT: Final[int] = Defaults.TIMEOUT

    # =============================================================================
    # ENUMS - Field types and other enums that modules import
    # =============================================================================

    class Enums:
        """Enums."""

        class FieldType(Enum):
            """Field type enumeration for validation and schema definition."""

            STRING = "string"
            INTEGER = "integer"
            FLOAT = "float"
            BOOLEAN = "boolean"
            DATE = "date"
            DATETIME = "datetime"
            UUID = "uuid"
            EMAIL = "email"

        class LogLevel(Enum):
            """Log level enumeration with numeric value support."""

            DEBUG = "DEBUG"
            INFO = "INFO"
            WARNING = "WARNING"
            ERROR = "ERROR"
            CRITICAL = "CRITICAL"
            TRACE = "TRACE"

            @override
            def __hash__(self) -> int:
                """Hash based on enum value."""
                return hash(self.value)

            @override
            def __eq__(self, other: object) -> bool:
                """Support comparison with string values."""
                if isinstance(other, str):
                    return self.value == other
                return super().__eq__(other)

            @classmethod
            def get_numeric_levels(cls) -> dict[str, int]:
                """Get numeric level values for logging compatibility."""
                return {
                    "CRITICAL": 50,
                    "ERROR": 40,
                    "WARNING": 30,
                    "INFO": 20,
                    "DEBUG": 10,
                    "TRACE": 5,
                }

            def get_numeric_value(self) -> int:
                """Get numeric value for this log level."""
                return self.get_numeric_levels()[self.value]

        class Environment(Enum):
            """Environment type enumeration."""

            DEVELOPMENT = "development"
            PRODUCTION = "production"
            STAGING = "staging"
            TESTING = "testing"

        class ConnectionType(Enum):
            """Connection type enumeration."""

            DATABASE = "database"
            REDIS = "redis"
            LDAP = "ldap"
            ORACLE = "oracle"
            POSTGRES = "postgres"
            REST_API = "rest_api"
            GRPC = "grpc"
            FILE = "file"
            STREAM = "stream"

        class DataFormat(Enum):
            """Data format enumeration."""

            JSON = "json"
            XML = "xml"
            CSV = "csv"
            LDIF = "ldif"
            YAML = "yaml"
            PARQUET = "parquet"
            AVRO = "avro"
            PROTOBUF = "protobuf"

        class OperationStatus(Enum):
            """Operation status enumeration."""

            PENDING = "pending"
            RUNNING = "running"
            COMPLETED = "completed"
            FAILED = "failed"
            CANCELLED = "cancelled"
            RETRYING = "retrying"

        class EntityStatus(Enum):
            """Entity status enumeration."""

            ACTIVE = "active"
            INACTIVE = "inactive"
            PENDING = "pending"
            DELETED = "deleted"
            SUSPENDED = "suspended"


FlextOperationStatus = FlextCoreConstants.Enums.OperationStatus
FlextLogLevel = FlextCoreConstants.Enums.LogLevel
FlextFieldType = FlextCoreConstants.Enums.FieldType
FlextEnvironment = FlextCoreConstants.Enums.Environment
FlextEntityStatus = FlextCoreConstants.Enums.EntityStatus
FlextDataFormat = FlextCoreConstants.Enums.DataFormat
FlextConnectionType = FlextCoreConstants.Enums.ConnectionType


# =============================================================================
# LEGACY CONSTANTS MOVED TO LEGACY.PY
# =============================================================================
# Legacy flat constants ERROR_CODES, MESSAGES, STATUS_CODES, VALIDATION_RULES,
# DEFAULT_TIMEOUT, DEFAULT_RETRIES, DEFAULT_PAGE_SIZE, VERSION, NAME, and
# pattern aliases have been moved to legacy.py with deprecation warnings.
#
# NEW USAGE: Use proper FlextConstants nested structure
#   FlextCoreConstants.Errors.VALIDATION_ERROR
#   FlextCoreConstants.Defaults.TIMEOUT
#   FlextCoreConstants.Patterns.EMAIL_PATTERN
#
# Use FlextConstants hierarchical structure for new code

# =============================================================================
# EXPORTS - Semantic constants + compatibility
# =============================================================================

# Simplified ERROR_CODES mapping
ERROR_CODES: dict[str, str] = {
    "GENERIC_ERROR": FlextCoreConstants.Errors.GENERIC_ERROR,
    "VALIDATION_ERROR": "FLEXT_VALIDATION_ERROR",
    "CONNECTION_ERROR": FlextCoreConstants.Errors.CONNECTION_ERROR,
    "TIMEOUT_ERROR": FlextCoreConstants.Errors.TIMEOUT_ERROR,
    "OPERATION_ERROR": "FLEXT_OPERATION_ERROR",
    "TYPE_ERROR": "FLEXT_TYPE_ERROR",
    "CONFIG_ERROR": "FLEXT_CONFIG_ERROR",
    "CONFIGURATION_ERROR": "FLEXT_CONFIG_ERROR",  # Alias for consistency
    "AUTH_ERROR": "FLEXT_AUTH_ERROR",
    "PERMISSION_ERROR": "FLEXT_PERMISSION_ERROR",
    "BIND_ERROR": "FLEXT_BIND_ERROR",
    "CHAIN_ERROR": "FLEXT_CHAIN_ERROR",
    "MAP_ERROR": "MAP_ERROR",
}

# Direct message access
MESSAGES = FlextCoreConstants.Messages
SERVICE_NAME_EMPTY: Final[str] = "Service name cannot be empty"

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Legacy alias for FlextCoreConstants - maintain backward compatibility
FlextConstants = FlextCoreConstants

__all__: Final[list[str]] = [
    # Legacy constants
    "ERROR_CODES",
    "MESSAGES",
    "SERVICE_NAME_EMPTY",
    # Enumerations
    "FlextConnectionType",
    # Backward compatibility
    "FlextConstants",
    # Main constants class with nested structure
    "FlextCoreConstants",
    "FlextDataFormat",
    "FlextEntityStatus",
    "FlextEnvironment",
    "FlextFieldType",
    "FlextLogLevel",
    "FlextOperationStatus",
    # Use FlextConstants hierarchical structure
    # Use FlextConstants for hierarchical access
]
