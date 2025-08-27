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

        # Example: FlextConstants.Core.VERSION

        timeout = FlextConstants.Defaults.TIMEOUT
        error_code = FlextConstants.Errors.VALIDATION_ERROR
        log_level = FlextConstants.Observability.DEFAULT_LOG_LEVEL

    Using enumerations for type safety::

        # Example: FlextLogLevel.INFO, FlextEnvironment.PRODUCTION

        current_level = FlextLogLevel.INFO
        env = FlextEnvironment.PRODUCTION
        numeric_level = current_level.get_numeric_value()  # Returns 20

    Error handling with structured codes::

        # Example usage:

        # Modern structured approach
        error = FlextConstants.Errors.CONNECTION_ERROR
        message = FlextConstants.Messages.DATABASE_CONNECTION_FAILED

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


class FlextConstants:
    """Hierarchical constants system organizing FLEXT constants by domain.

    This is the single consolidated class for all FLEXT Core constants, following the
    Flext[Area][Module] pattern where this represents FlextConstants.
    All other FLEXT libraries should inherit from this class to maintain hierarchy.

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
            app_name = FlextConstants.Core.NAME
            version = FlextConstants.Core.VERSION

            # Network configuration
            timeout = FlextConstants.Network.DEFAULT_TIMEOUT
            port = FlextConstants.Platform.FLEXT_SERVICE_PORT

            # Error handling
            error_code = FlextConstants.Errors.VALIDATION_ERROR
            error_message = FlextConstants.Messages.VALIDATION_FAILED

            # Performance tuning
            batch_size = FlextConstants.Performance.DEFAULT_BATCH_SIZE
            threshold = FlextConstants.Performance.SLOW_QUERY_THRESHOLD

        Type-safe constant access::

            # All constants are typed and validated
            max_retries: int = FlextConstants.Defaults.MAX_RETRIES
            error_codes: dict[str, str] = FlextConstants.Errors.MESSAGES
            log_levels: list[str] = FlextConstants.Observability.LOG_LEVELS

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

        # Python version requirements
        MIN_PYTHON_VERSION: Final[tuple[int, int, int]] = (3, 13, 0)
        MAX_PYTHON_VERSION: Final[tuple[int, int, int]] = (3, 14, 0)

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

        # Business domain validation (from shared_domain examples)
        MIN_AGE: Final[int] = 18
        MAX_AGE: Final[int] = 120
        MAX_EMAIL_LENGTH: Final[int] = 254
        MIN_NAME_LENGTH: Final[int] = 2
        MAX_NAME_LENGTH: Final[int] = 100
        MIN_STREET_LENGTH: Final[int] = 5
        MIN_CITY_LENGTH: Final[int] = 2
        MIN_POSTAL_CODE_LENGTH: Final[int] = 3
        MIN_COUNTRY_LENGTH: Final[int] = 2
        CURRENCY_CODE_LENGTH: Final[int] = 3
        MIN_PHONE_LENGTH: Final[int] = 10
        MAX_PHONE_LENGTH: Final[int] = 15
        MIN_PRODUCT_NAME_LENGTH: Final[int] = 3
        MAX_PRODUCT_PRICE: Final[int] = 100000

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
        CREATION: Final[str] = "FLEXT_3002"

        # Command and Query Processing Errors
        COMMAND_PROCESSING_FAILED: Final[str] = "FLEXT_2006"
        COMMAND_HANDLER_NOT_FOUND: Final[str] = "FLEXT_2007"
        COMMAND_BUS_ERROR: Final[str] = "FLEXT_2008"
        QUERY_PROCESSING_FAILED: Final[str] = "FLEXT_2009"

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
        NOT_FOUND_ERROR: Final[str] = "NOT_FOUND_ERROR"
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
            COMMAND_PROCESSING_FAILED: "Command processing failed",
            COMMAND_HANDLER_NOT_FOUND: "Command handler not found",
            COMMAND_BUS_ERROR: "Command bus error",
            QUERY_PROCESSING_FAILED: "Query processing failed",
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

        # LDAP and directory patterns (found in workspace)
        LDAP_DN_PATTERN: Final[str] = r"^[a-zA-Z]+=.+$"
        LDAP_CN_PATTERN: Final[str] = r"cn=.*"
        LDAP_OU_PATTERN: Final[str] = r"ou=.*"
        LDAP_DC_PATTERN: Final[str] = r"dc=.*"
        LDAP_BASE_DN_VALIDATION: Final[str] = r".*=.*"

        # Auth and security patterns (found in flext-auth)
        ALPHANUMERIC_UNDERSCORE_DASH: Final[str] = r"^[a-zA-Z0-9_-]+$"
        ENTITY_ID_PATTERN: Final[str] = r"^[a-zA-Z0-9_-]+$"

        # OID patterns (found in schema processors)
        OID_PATTERN: Final[str] = r"\(\s*([\d.]+)"
        OID_NUMERIC_PATTERN: Final[str] = r"\(\s*([0-9]+(?:\.[0-9]+)*)"

        # LDIF schema patterns
        LDIF_NAME_PATTERN: Final[str] = r"NAME\s+'([^']+)'"

        # URL credential redaction pattern
        URL_CREDENTIAL_PATTERN: Final[str] = r"://([^:]+):([^@]+)@"
        URL_CREDENTIAL_REPLACEMENT: Final[str] = r"://[REDACTED]@"

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

        # Common operation timeouts
        COMMAND_TIMEOUT: Final[int] = 300  # 5 minutes
        PROCESS_COMMUNICATION_TIMEOUT: Final[int] = 300  # 5 minutes

        # Processing limits
        MAX_ENTRIES_PER_OPERATION: Final[int] = 50000
        HASH_MOD_LIMIT: Final[int] = 100000
        FALLBACK_OID: Final[int] = 999999

        # Monitoring intervals
        METRICS_INTERVAL: Final[int] = 60  # seconds
        HEALTH_CHECK_INTERVAL: Final[int] = 30  # seconds

        # Time and size calculation constants (consolidated from utilities.py)
        SECONDS_PER_MINUTE: Final[int] = 60
        SECONDS_PER_HOUR: Final[int] = 3600
        BYTES_PER_KB: Final[int] = 1024

        # Payload processing constants
        PAYLOAD_MAX_SIZE: Final[int] = 65536  # 64KB maximum uncompressed size
        PAYLOAD_COMPRESSION_LEVEL: Final[int] = 6  # zlib compression level

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
        DOTENV_FILES: Final[list[str]] = [".env", ".env.local", ".env.production"]
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
    # URLS AND ENDPOINTS - Centralized URL and API endpoint constants
    # =========================================================================

    class Urls:
        """URL and endpoint constants for the FLEXT ecosystem.

        This class centralizes all URL patterns, API endpoints, and service URLs
        found throughout the workspace to eliminate hardcoded URLs and provide
        single source of truth for all network endpoints.

        Architecture Principles Applied:
            - Single Source of Truth: All URLs centralized here
            - DRY Principle: Eliminates URL duplication across workspace
            - Configuration Management: Supports different environments
        """

        # Base service URLs - Default development configuration
        FLEXCORE_BASE_URL: Final[str] = "http://localhost:8080"
        FLEXT_SERVICE_BASE_URL: Final[str] = "http://localhost:8081"
        MELTANO_UI_BASE_URL: Final[str] = "http://localhost:5000"

        # Observability and monitoring URLs
        GRAFANA_BASE_URL: Final[str] = "http://localhost:3000"
        PROMETHEUS_BASE_URL: Final[str] = "http://localhost:9090"
        JAEGER_BASE_URL: Final[str] = "http://localhost:16686"
        PROMETHEUS_PUSH_GATEWAY: Final[str] = "http://localhost:9091"
        OTEL_EXPORTER_OTLP_ENDPOINT: Final[str] = "http://localhost:4317"
        JAEGER_COLLECTOR_ENDPOINT: Final[str] = "http://localhost:14268/api/traces"
        ELASTICSEARCH_URL: Final[str] = "http://localhost:9200"
        ALERT_MANAGER_URL: Final[str] = "http://localhost:9093"

        # Database connection URLs
        DEFAULT_POSTGRES_URL: Final[str] = (
            "postgresql://flext:password@localhost:5432/flext"
        )
        DEFAULT_REDIS_URL: Final[str] = "redis://localhost:6379/0"

        # External service URLs
        PYPI_API_BASE: Final[str] = "https://pypi.org/pypi"
        GITHUB_FLEXT_REPO: Final[str] = "https://github.com/flext-sh/flext"
        FLEXT_DOCS_URL: Final[str] = "https://docs.flext.sh"
        MELTANO_HUB_URL: Final[str] = "https://hub.meltano.com"

        # CDN and external resources
        BOOTSTRAP_CDN: Final[str] = (
            "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
        )
        BOOTSTRAP_ICONS_CDN: Final[str] = (
            "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css"
        )
        HTMX_CDN: Final[str] = "https://unpkg.com/htmx.org@1.9.10"
        ALPINE_JS_CDN: Final[str] = "https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"
        CHART_JS_CDN: Final[str] = "https://cdn.jsdelivr.net/npm/chart.js"
        FONT_AWESOME_CDN: Final[str] = (
            "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
        )
        DOMPURIFY_CDN: Final[str] = (
            "https://cdnjs.cloudflare.com/ajax/libs/dompurify/2.4.0/purify.min.js"
        )

        # Oracle and database specific URLs
        ORACLE_INSTANT_CLIENT_URL: Final[str] = (
            "https://download.oracle.com/otn_software/linux/instantclient/2340000/instantclient-basic-linux.x64-23.4.0.24.05.zip"
        )

    class Endpoints:
        """API endpoint patterns for the FLEXT ecosystem.

        This class centralizes all API endpoint paths used throughout the workspace
        to provide consistent API routing and eliminate hardcoded endpoint paths.

        Architecture Principles Applied:
            - RESTful Design: Follows REST conventions
            - Versioning: Supports API versioning patterns
            - Consistency: Unified endpoint naming across services
        """

        # Core API versioning
        API_V1_BASE: Final[str] = "/api/v1"

        # Health and status endpoints
        HEALTH: Final[str] = "/health"
        STATUS: Final[str] = "/status"
        METRICS: Final[str] = "/metrics"
        LOGS: Final[str] = "/logs"
        TRACES: Final[str] = "/traces"

        # FlexCore API endpoints
        FLEXCORE_HEALTH: Final[str] = "/api/v1/health"
        FLEXCORE_STATUS: Final[str] = "/api/v1/status"
        FLEXCORE_PLUGINS: Final[str] = "/api/v1/plugins"
        FLEXCORE_RUNTIMES: Final[str] = "/api/v1/runtimes"
        FLEXCORE_WORKFLOWS: Final[str] = "/api/v1/workflows"
        FLEXCORE_CONFIG: Final[str] = "/api/v1/config"
        FLEXCORE_COORDINATION: Final[str] = "/api/v1/coordination/command"

        # Meltano API endpoints
        MELTANO_PROJECTS: Final[str] = "/api/v1/runtimes/meltano/projects"
        MELTANO_EXECUTE: Final[str] = "/api/v1/runtimes/meltano/execute"
        MELTANO_INSTALL: Final[str] = "/api/v1/runtimes/meltano/install"
        MELTANO_CONFIG: Final[str] = "/api/v1/runtimes/meltano/config"
        MELTANO_SCHEDULES: Final[str] = "/api/v1/runtimes/meltano/schedules"

        # Singer API endpoints
        SINGER_TAPS: Final[str] = "/api/v1/runtimes/meltano/singer/taps"
        SINGER_TARGETS: Final[str] = "/api/v1/runtimes/meltano/singer/targets"
        SINGER_EXTRACT: Final[str] = "/api/v1/runtimes/meltano/singer/extract"
        SINGER_LOAD: Final[str] = "/api/v1/runtimes/meltano/singer/load"
        SINGER_SCHEMAS: Final[str] = "/api/v1/runtimes/meltano/singer/schemas"
        SINGER_TEST_TAP: Final[str] = "/api/v1/runtimes/meltano/singer/test/tap"
        SINGER_TEST_TARGET: Final[str] = "/api/v1/runtimes/meltano/singer/test/target"

        # DBT API endpoints
        DBT_MODELS: Final[str] = "/api/v1/runtimes/meltano/dbt/models"
        DBT_RUN: Final[str] = "/api/v1/runtimes/meltano/dbt/run"
        DBT_TEST: Final[str] = "/api/v1/runtimes/meltano/dbt/test"
        DBT_COMPILE: Final[str] = "/api/v1/runtimes/meltano/dbt/compile"
        DBT_DOCS: Final[str] = "/api/v1/runtimes/meltano/dbt/docs"
        DBT_LINEAGE: Final[str] = "/api/v1/runtimes/meltano/dbt/lineage"

        # Application management endpoints (flext-web)
        APPS_BASE: Final[str] = "/api/v1/apps"
        APP_DETAIL: Final[str] = "/api/v1/apps/{app_id}"
        APP_START: Final[str] = "/api/v1/apps/{app_id}/start"
        APP_STOP: Final[str] = "/api/v1/apps/{app_id}/stop"

        # Pipeline management endpoints
        PIPELINES: Final[str] = "/api/v1/pipelines"
        PIPELINE_DETAIL: Final[str] = "/api/v1/pipelines/{id}"
        PIPELINE_EXECUTE: Final[str] = "/api/v1/pipelines/{id}/execute"

        # Data source management endpoints
        SOURCES: Final[str] = "/api/v1/sources"
        SOURCE_SCHEMA: Final[str] = "/api/v1/sources/{id}/schema"
        SOURCE_TEST: Final[str] = "/api/v1/sources/{id}/test"

    # =========================================================================
    # DOMAIN-SPECIFIC CONSTANTS - Extension points for subclasses
    # =========================================================================

    class Web:
        """Web interface specific constants.

        These constants are used by flext-web module and provide defaults
        for web application configuration, ports, and HTTP settings.
        """

        # Port configurations
        DEFAULT_HTTP_PORT: Final[int] = 80
        DEFAULT_HTTPS_PORT: Final[int] = 443
        DEFAULT_DEVELOPMENT_PORT: Final[int] = 8080
        MIN_PORT: Final[int] = 1
        MAX_PORT: Final[int] = 65535

        # Application limits
        MAX_APP_NAME_LENGTH: Final[int] = 255
        MIN_APP_NAME_LENGTH: Final[int] = 1
        MAX_CONCURRENT_APPS: Final[int] = 100
        MAX_APP_TIMEOUT: Final[int] = 3600

        # HTTP status codes (commonly used)
        HTTP_OK: Final[int] = 200
        HTTP_CREATED: Final[int] = 201
        HTTP_BAD_REQUEST: Final[int] = 400
        HTTP_UNAUTHORIZED: Final[int] = 401
        HTTP_FORBIDDEN: Final[int] = 403
        HTTP_NOT_FOUND: Final[int] = 404
        HTTP_INTERNAL_ERROR: Final[int] = 500
        HTTP_GATEWAY_TIMEOUT: Final[int] = 504
        MAX_HTTP_STATUS: Final[int] = 599

        # MIME types (centralized from target clients and API modules)
        JSON_MIME: Final[str] = "application/json"
        XML_MIME: Final[str] = "application/xml"
        TEXT_MIME: Final[str] = "text/plain"
        HTML_MIME: Final[str] = "text/html"

    class CLI:
        """Command line interface specific constants.

        These constants are used by flext-cli module and provide defaults
        for terminal display, table formatting, and command line operations.
        """

        # Terminal dimensions
        DEFAULT_TERMINAL_WIDTH: Final[int] = 80
        DEFAULT_TERMINAL_HEIGHT: Final[int] = 24
        MIN_TERMINAL_WIDTH: Final[int] = 40

        # Table display
        DEFAULT_TABLE_WIDTH: Final[int] = 120
        MAX_COLUMN_WIDTH: Final[int] = 50
        PROGRESS_BAR_WIDTH: Final[int] = 40

        # Display colors (Rich color names)
        SUCCESS_COLOR: Final[str] = "green"
        ERROR_COLOR: Final[str] = "red"
        WARNING_COLOR: Final[str] = "yellow"
        INFO_COLOR: Final[str] = "blue"

        # Display formatting
        SPINNER_STYLE: Final[str] = "dots"
        MAX_OUTPUT_WIDTH: Final[int] = 120

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

        # Tracing configuration
        DEFAULT_TRACE_TIMEOUT: Final[float] = 30.0
        DEFAULT_HEALTH_CHECK_INTERVAL: Final[float] = 60.0
        DEFAULT_METRIC_COLLECTION_INTERVAL: Final[float] = 15.0

        # Metric limits
        MAX_METRIC_NAME_LENGTH: Final[int] = 255
        MAX_METRIC_LABELS: Final[int] = 20
        MAX_TRACE_SPAN_COUNT: Final[int] = 1000

        # Alert thresholds
        DEFAULT_ERROR_THRESHOLD: Final[float] = 0.01  # 1%
        DEFAULT_LATENCY_THRESHOLD: Final[float] = 1000.0  # 1 second
        MAX_ALERT_HISTORY: Final[int] = 1000

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
        SERIALIZATION_FORMAT_BINARY: Final[str] = "binary"

    class Quality:
        """Code quality specific constants.

        These constants are used by flext-quality module and provide
        defaults for quality analysis, complexity metrics, and compliance.
        """

        # Quality thresholds
        MIN_COVERAGE_THRESHOLD: Final[float] = 80.0
        MAX_COMPLEXITY_THRESHOLD: Final[int] = 10
        MAX_DUPLICATION_THRESHOLD: Final[float] = 5.0

        # Analysis limits
        MAX_FILE_SIZE_BYTES: Final[int] = 1048576  # 1MB
        MAX_FUNCTION_LINES: Final[int] = 50
        MAX_CLASS_METHODS: Final[int] = 20

        # PEP8 compliance
        MAX_LINE_LENGTH: Final[int] = 88  # Black default
        MAX_FUNCTION_ARGUMENTS: Final[int] = 5
        MAX_NESTING_DEPTH: Final[int] = 4

    class Oracle:
        """Oracle database specific constants.

        These constants are used by flext-db-oracle module and provide
        defaults for Oracle database connections, queries, and operations.
        """

        # Oracle-specific ports and connection settings
        DEFAULT_ORACLE_PORT: Final[int] = 1521
        DEFAULT_SERVICE_NAME: Final[str] = "XEPDB1"
        DEFAULT_CHARSET: Final[str] = "UTF8"
        DEFAULT_NLS_LANG: Final[str] = "AMERICAN_AMERICA.UTF8"

        # Connection pool settings
        DEFAULT_POOL_MIN: Final[int] = 1
        DEFAULT_POOL_MAX: Final[int] = 10
        DEFAULT_POOL_INCREMENT: Final[int] = 1

        # Query limits and timeouts
        MAX_FETCH_SIZE: Final[int] = 1000
        DEFAULT_ARRAY_SIZE: Final[int] = 100
        MAX_IDENTIFIER_LENGTH: Final[int] = 128  # Oracle 12.2+ limit

        # Oracle data types and limits
        MAX_VARCHAR2_LENGTH: Final[int] = 4000
        MAX_CLOB_SIZE: Final[int] = 4294967295  # 4GB - 1
        MAX_NUMBER_PRECISION: Final[int] = 38

        # Connection string patterns
        TNS_CONNECTION_PATTERN: Final[str] = r"^[A-Za-z0-9_]+$"
        SID_PATTERN: Final[str] = r"^[A-Za-z][A-Za-z0-9_]*$"
        SERVICE_NAME_PATTERN: Final[str] = r"^[A-Za-z][A-Za-z0-9_.]*$"

    class LDAP:
        """LDAP specific constants.

        These constants are used by flext-ldap module and provide defaults
        for LDAP connections, searches, and directory operations.
        """

        # Standard LDAP ports
        DEFAULT_LDAP_PORT: Final[int] = 389
        DEFAULT_LDAPS_PORT: Final[int] = 636
        DEFAULT_GLOBAL_CATALOG_PORT: Final[int] = 3268
        DEFAULT_GLOBAL_CATALOG_SSL_PORT: Final[int] = 3269

        # Search scope constants
        SCOPE_BASE: Final[int] = 0
        SCOPE_ONELEVEL: Final[int] = 1
        SCOPE_SUBTREE: Final[int] = 2

        # Operation limits
        DEFAULT_PAGE_SIZE: Final[int] = 1000
        MAX_SEARCH_RESULTS: Final[int] = 10000
        DEFAULT_TIMEOUT: Final[int] = 30
        MAX_RETRIES: Final[int] = 3

        # DN and attribute patterns
        DN_PATTERN: Final[str] = r"^[a-zA-Z]+=.+$"
        CN_PATTERN: Final[str] = r"cn=.*"
        OU_PATTERN: Final[str] = r"ou=.*"
        DC_PATTERN: Final[str] = r"dc=.*"

        # Schema and object class limits
        MAX_ATTRIBUTE_LENGTH: Final[int] = 255
        MAX_OBJECT_CLASS_COUNT: Final[int] = 100
        MAX_ENTRY_SIZE: Final[int] = 1048576  # 1MB

    class GRPC:
        """gRPC specific constants.

        These constants are used by flext-grpc module and provide defaults
        for gRPC services, connections, and messaging.
        """

        # gRPC default port and connection settings
        DEFAULT_GRPC_PORT: Final[int] = 50051
        MIN_PORT: Final[int] = 1024  # Avoid privileged ports
        MAX_PORT: Final[int] = 65535

        # Service configuration
        DEFAULT_MAX_WORKERS: Final[int] = 10
        DEFAULT_TIMEOUT: Final[float] = 30.0
        MAX_MESSAGE_SIZE: Final[int] = 4194304  # 4MB

        # Connection limits
        MAX_CONNECTION_IDLE: Final[int] = 300  # 5 minutes
        MAX_CONNECTION_AGE: Final[int] = 1800  # 30 minutes
        KEEPALIVE_TIME: Final[int] = 60  # 1 minute

        # Service name validation
        MAX_SERVICE_NAME_LENGTH: Final[int] = 255
        SERVICE_NAME_PATTERN: Final[str] = r"^[a-zA-Z][a-zA-Z0-9_.]*$"
        METHOD_NAME_PATTERN: Final[str] = r"^[A-Z][a-zA-Z0-9]*$"

        # Performance tuning
        DEFAULT_CHANNEL_OPTIONS: Final[dict[str, int]] = {
            "grpc.keepalive_time_ms": 60000,
            "grpc.keepalive_timeout_ms": 5000,
            "grpc.http2.max_pings_without_data": 0,
            "grpc.http2.min_time_between_pings_ms": 10000,
        }

        # Streaming configuration
        SERVER_STREAMING_BATCH_SIZE: Final[int] = 50
        CLIENT_STREAMING_BUFFER_THRESHOLD: Final[int] = 3
        BIDIRECTIONAL_STREAMING_QUEUE_SIZE: Final[int] = 1000
        STREAM_CLEANUP_MAX_AGE_SECONDS: Final[float] = 300.0
        STREAM_PROCESSING_TIMEOUT_SECONDS: Final[float] = 30.0
        MAX_CONCURRENT_STREAMS_PER_CLIENT: Final[int] = 100
        STREAM_METRICS_COLLECTION_INTERVAL: Final[float] = 60.0
        STREAM_HEALTH_DEGRADED_THRESHOLD: Final[float] = 60.0

        # Memory management and buffer configuration
        MAX_BUFFER_SIZE_BYTES: Final[int] = 50 * 1024 * 1024  # 50MB per stream buffer
        MEMORY_PRESSURE_THRESHOLD: Final[float] = (
            0.85  # 85% memory usage triggers cleanup
        )
        BUFFER_CLEANUP_BATCH_SIZE: Final[int] = 100  # Clean up buffers in batches
        LOW_MEMORY_THRESHOLD: Final[int] = (
            100 * 1024 * 1024
        )  # 100MB system memory threshold
        ADAPTIVE_BUFFER_SCALING_FACTOR: Final[float] = (
            0.8  # Scale buffers by 80% under pressure
        )
        MEMORY_CLEANUP_INTERVAL_SHORT: Final[int] = (
            30  # 30 seconds for frequent cleanup
        )
        MEMORY_CLEANUP_INTERVAL_LONG: Final[int] = (
            60  # 60 seconds for infrequent cleanup
        )
        HIGH_PRESSURE_RATIO_THRESHOLD: Final[float] = (
            0.5  # 50% streams under pressure triggers cleanup
        )

    # =========================================================================
    # DBT ORACLE CONSTANTS - DBT Oracle-specific constants
    # =========================================================================

    class DBT:
        """DBT-specific constants for data transformation.

        These constants are used by flext-dbt-oracle and provide defaults
        for DBT model configurations, testing, and performance.
        """

        # Model types
        MODEL_TYPE_TABLE: Final[str] = "table"
        MODEL_TYPE_VIEW: Final[str] = "view"
        MODEL_TYPE_INCREMENTAL: Final[str] = "incremental"
        MODEL_TYPE_EPHEMERAL: Final[str] = "ephemeral"

        # Materialization strategies
        MATERIALIZATION_TABLE: Final[str] = "table"
        MATERIALIZATION_VIEW: Final[str] = "view"
        MATERIALIZATION_INCREMENTAL: Final[str] = "incremental"

        # Test severity levels
        TEST_SEVERITY_WARN: Final[str] = "warn"
        TEST_SEVERITY_ERROR: Final[str] = "error"

        # Freshness check intervals (in hours)
        FRESHNESS_WARN_AFTER: Final[int] = 24
        FRESHNESS_ERROR_AFTER: Final[int] = 48

        # Batch processing
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        LARGE_BATCH_SIZE: Final[int] = 5000
        MAX_BATCH_SIZE: Final[int] = 10000

        # Performance thresholds
        SLOW_QUERY_THRESHOLD: Final[float] = 5.0  # seconds
        VERY_SLOW_QUERY_THRESHOLD: Final[float] = 30.0  # seconds
        LARGE_OBJECT_COUNT_THRESHOLD: Final[int] = 100
        VERY_LARGE_OBJECT_COUNT_THRESHOLD: Final[int] = 1000

    class WebExtended:
        """Extended web constants for comprehensive web operations.

        Additional web constants beyond the basic Web class for advanced
        web application features and validations.
        """

        # Session and security
        SESSION_TIMEOUT: Final[int] = 3600  # 1 hour
        MIN_SECRET_KEY_LENGTH: Final[int] = 32

        # Application limits (extended from basic Web)
        MAX_CONCURRENT_APPS: Final[int] = 100
        MIN_APP_TIMEOUT: Final[int] = 1
        MAX_APP_TIMEOUT: Final[int] = 3600

        # Content types (additional)
        HTML_CONTENT_TYPE: Final[str] = "text/html"
        CSS_CONTENT_TYPE: Final[str] = "text/css"
        JS_CONTENT_TYPE: Final[str] = "application/javascript"

        # HTTP advanced
        MIN_HTTP_STATUS_CODE: Final[int] = 100
        MAX_HTTP_STATUS_CODE: Final[int] = 599

    class QualityExtended:
        """Extended quality constants for comprehensive quality assessment.

        Quality thresholds and assessment levels used by flext-quality
        for comprehensive code quality evaluation.
        """

        # Quality score thresholds
        OUTSTANDING_THRESHOLD: Final[float] = 95.0
        EXCELLENT_THRESHOLD: Final[float] = 90.0
        GOOD_THRESHOLD: Final[float] = 80.0
        ACCEPTABLE_THRESHOLD: Final[float] = 70.0
        BELOW_AVERAGE_THRESHOLD: Final[float] = 60.0

        # Coverage requirements (specific targets)
        TARGET_COVERAGE: Final[float] = 95.0
        EXCELLENT_COVERAGE: Final[float] = 98.0

        # Complexity limits (specific)
        WARNING_COMPLEXITY: Final[int] = 7
        IDEAL_COMPLEXITY: Final[int] = 5

        # Security and maintainability
        MINIMUM_SECURITY_SCORE: Final[float] = 90.0
        TARGET_SECURITY_SCORE: Final[float] = 95.0
        MINIMUM_MAINTAINABILITY: Final[float] = 80.0
        TARGET_MAINTAINABILITY: Final[float] = 90.0

    class Metrics:
        """Metrics and observability constants for comprehensive monitoring.

        Metric types, alert levels, trace statuses, and health check
        configurations used across the observability ecosystem.
        """

        # Metric types
        METRIC_TYPE_COUNTER: Final[str] = "counter"
        METRIC_TYPE_GAUGE: Final[str] = "gauge"
        METRIC_TYPE_HISTOGRAM: Final[str] = "histogram"
        METRIC_TYPE_SUMMARY: Final[str] = "summary"

        # Alert levels
        ALERT_LEVEL_INFO: Final[str] = "info"
        ALERT_LEVEL_WARNING: Final[str] = "warning"
        ALERT_LEVEL_ERROR: Final[str] = "error"
        ALERT_LEVEL_CRITICAL: Final[str] = "critical"

        # Trace statuses
        TRACE_STATUS_STARTED: Final[str] = "started"
        TRACE_STATUS_RUNNING: Final[str] = "running"
        TRACE_STATUS_COMPLETED: Final[str] = "completed"
        TRACE_STATUS_FAILED: Final[str] = "failed"

        # Health check statuses
        HEALTH_STATUS_HEALTHY: Final[str] = "healthy"
        HEALTH_STATUS_DEGRADED: Final[str] = "degraded"
        HEALTH_STATUS_UNHEALTHY: Final[str] = "unhealthy"

        # Service names for observability
        SERVICE_METRICS: Final[str] = "metrics"
        SERVICE_TRACING: Final[str] = "tracing"
        SERVICE_ALERTS: Final[str] = "alerts"
        SERVICE_HEALTH: Final[str] = "health"
        SERVICE_LOGGING: Final[str] = "logging"

    class Cache:
        """Cache configuration constants across the ecosystem.

        TTL values, cache sizes, and cleanup intervals used by various
        caching systems throughout the FLEXT platform.
        """

        # TTL values in seconds
        METADATA_CACHE_TTL: Final[int] = 3600  # 1 hour
        QUERY_CACHE_TTL: Final[int] = 300  # 5 minutes
        SHORT_CACHE_TTL: Final[int] = 60  # 1 minute
        LONG_CACHE_TTL: Final[int] = 86400  # 24 hours

        # Cache sizes
        MAX_CACHE_ENTRIES: Final[int] = 1000
        DEFAULT_CACHE_SIZE: Final[int] = 500
        LARGE_CACHE_SIZE: Final[int] = 5000

        # Cache maintenance
        CACHE_CLEANUP_INTERVAL: Final[int] = 600  # 10 minutes
        CACHE_EVICTION_THRESHOLD: Final[float] = 0.85  # 85% full

    # =========================================================================
    # SINGER ECOSYSTEM CONSTANTS - Tap and Target patterns
    # =========================================================================

    class Singer:
        """Singer ecosystem constants for taps and targets.

        Common constants used across Singer taps, targets, and transformations
        in the FLEXT Singer ecosystem.
        """

        # Stream processing
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        MAX_BATCH_SIZE: Final[int] = 10000
        DEFAULT_MAX_PARALLEL_STREAMS: Final[int] = 4

        # Record limits
        MAX_RECORD_SIZE: Final[int] = 10485760  # 10MB
        DEFAULT_BUFFER_SIZE: Final[int] = 65536  # 64KB

        # Connection timeouts
        DEFAULT_CONNECTION_TIMEOUT: Final[int] = 30
        DISCOVERY_TIMEOUT: Final[int] = 60
        DEFAULT_REQUEST_TIMEOUT: Final[int] = 300  # 5 minutes

    class OracleOIC:
        """Oracle OIC (Oracle Integration Cloud) constants.

        API paths, endpoints, and configurations for Oracle Integration Cloud
        integrations used by flext-tap-oracle-oic.
        """

        # Oracle OIC API base paths
        OIC_API_BASE_PATH: Final[str] = "/ic/api/integration/v1"
        OIC_MONITORING_API_PATH: Final[str] = "/ic/api/monitoring/v1"
        OIC_DESIGNTIME_API_PATH: Final[str] = "/ic/api/designtime/v1"
        OIC_PROCESS_API_PATH: Final[str] = "/ic/api/process/v1"
        OIC_B2B_API_PATH: Final[str] = "/ic/api/b2b/v1"
        OIC_ENVIRONMENT_API_PATH: Final[str] = "/ic/api/environment/v1"

        # Core API endpoints
        INTEGRATIONS_ENDPOINT: Final[str] = "/integrations"
        CONNECTIONS_ENDPOINT: Final[str] = "/connections"
        PACKAGES_ENDPOINT: Final[str] = "/packages"
        MONITORING_INSTANCES: Final[str] = "/monitoring/instances"
        MONITORING_MESSAGES: Final[str] = "/monitoring/messages"
        MONITORING_ERRORS: Final[str] = "/monitoring/errors"

        # System endpoints
        HEALTH_ENDPOINT: Final[str] = "/health"
        METADATA_ENDPOINT: Final[str] = "/metadata"

    class Targets:
        """Target constants for Singer targets ecosystem.

        Common constants used across all Singer targets in the FLEXT
        ecosystem for data loading and processing.
        """

        # Default processing settings
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        LARGE_BATCH_SIZE: Final[int] = 5000
        MAX_BATCH_SIZE: Final[int] = 10000

        # Connection settings
        DEFAULT_MAX_PARALLEL_STREAMS: Final[int] = 4
        MAX_PARALLEL_STREAMS: Final[int] = 8

        # Error handling
        MAX_RETRIES: Final[int] = 3
        RETRY_DELAY: Final[float] = 1.0  # seconds

        # Data validation
        MAX_FIELD_NAME_LENGTH: Final[int] = 128
        MAX_TABLE_NAME_LENGTH: Final[int] = 64

    class Taps:
        """Tap constants for Singer taps ecosystem.

        Common constants used across all Singer taps in the FLEXT
        ecosystem for data extraction and discovery.
        """

        # Discovery settings
        DEFAULT_DISCOVERY_TIMEOUT: Final[int] = 60
        MAX_DISCOVERY_TIMEOUT: Final[int] = 300

        # Stream settings
        DEFAULT_STREAM_BUFFER_SIZE: Final[int] = 1000
        MAX_STREAM_BUFFER_SIZE: Final[int] = 10000

        # Replication settings
        FULL_REPLICATION: Final[str] = "FULL_TABLE"
        INCREMENTAL_REPLICATION: Final[str] = "INCREMENTAL"
        LOG_BASED_REPLICATION: Final[str] = "LOG_BASED"

        # State management
        STATE_BOOKMARK_PROPERTIES: Final[list[str]] = [
            "modified_at",
            "updated_at",
            "created_at",
        ]

    class Meltano:
        """Meltano constants for pipeline orchestration.

        Constants used in Meltano configurations and pipeline management
        across the FLEXT Meltano ecosystem.
        """

        # Default database ports
        DEFAULT_POSTGRES_PORT: Final[int] = 5432
        DEFAULT_MYSQL_PORT: Final[int] = 3306
        DEFAULT_ORACLE_PORT: Final[int] = 1521

        # Pipeline timeouts
        DEFAULT_TIMEOUT: Final[int] = 300  # 5 minutes
        DISCOVERY_TIMEOUT: Final[int] = 60
        EXTRACT_TIMEOUT: Final[int] = 3600  # 1 hour
        LOAD_TIMEOUT: Final[int] = 1800  # 30 minutes

        # Environment types
        ENVIRONMENT_DEV: Final[str] = "dev"
        ENVIRONMENT_STAGING: Final[str] = "staging"
        ENVIRONMENT_PROD: Final[str] = "prod"

    class LDIF:
        """LDIF processing constants for RFC 2849 compliance.

        Comprehensive constants for LDIF (LDAP Data Interchange Format) processing,
        validation, and analytics following RFC 2849 specifications.
        """

        # RFC 2849 Line Processing Constants
        DEFAULT_LINE_WRAP_LENGTH: Final[int] = 76
        MAX_LINE_LENGTH: Final[int] = 998
        CONTINUATION_CHAR: Final[str] = " "
        ATTRIBUTE_SEPARATOR: Final[str] = ": "
        BASE64_SEPARATOR: Final[str] = ":: "
        URL_SEPARATOR: Final[str] = ":< "
        COMMENT_PREFIX: Final[str] = "# "
        VERSION_PREFIX: Final[str] = "version: "
        CONTROL_PREFIX: Final[str] = "control: "
        DN_ATTRIBUTE: Final[str] = "dn"
        CHANGETYPE_ATTRIBUTE: Final[str] = "changetype"

        # LDIF Entry Processing Limits
        DEFAULT_MAX_ENTRIES: Final[int] = 20000
        LARGE_BATCH_MAX_ENTRIES: Final[int] = 100000
        MEMORY_EFFICIENT_BATCH_SIZE: Final[int] = 1000
        MAX_ATTRIBUTE_VALUE_LENGTH: Final[int] = 1000000
        DEFAULT_BUFFER_SIZE: Final[int] = 8192
        MAX_BUFFER_SIZE: Final[int] = 65536

        # LDAP Object Classes (RFC Compliant)
        LDAP_PERSON_CLASSES: Final[frozenset[str]] = frozenset({
            "person",
            "organizationalPerson",
            "inetOrgPerson",
            "posixAccount",
            "shadowAccount",
            "sambaSamAccount",
            "mailRecipient",
            "uidObject",
        })

        LDAP_ORGANIZATIONAL_CLASSES: Final[frozenset[str]] = frozenset({
            "organization",
            "organizationalUnit",
            "organizationalRole",
            "organizationalPerson",
            "dcObject",
            "domainComponent",
        })

        LDAP_GROUP_CLASSES: Final[frozenset[str]] = frozenset({
            "groupOfNames",
            "groupOfUniqueNames",
            "posixGroup",
            "sambaGroupMapping",
            "mailGroup",
            "distributionList",
            "organizationalRole",
        })

        # LDIF Change Types (RFC 2849)
        CHANGETYPE_ADD: Final[str] = "add"
        CHANGETYPE_DELETE: Final[str] = "delete"
        CHANGETYPE_MODIFY: Final[str] = "modify"
        CHANGETYPE_MODDN: Final[str] = "moddn"
        CHANGETYPE_MODRDN: Final[str] = "modrdn"

        # Modification Operations
        MOD_ADD: Final[str] = "add"
        MOD_DELETE: Final[str] = "delete"
        MOD_REPLACE: Final[str] = "replace"
        MOD_INCREMENT: Final[str] = "increment"

        # LDIF Encoding Constants
        UTF8_ENCODING: Final[str] = "utf-8"
        ASCII_ENCODING: Final[str] = "ascii"
        LATIN1_ENCODING: Final[str] = "latin1"
        DEFAULT_ENCODING: Final[str] = UTF8_ENCODING

        # Validation Messages
        VALIDATION_MESSAGES: Final[dict[str, str]] = {
            "INVALID_DN": "Invalid Distinguished Name format",
            "MISSING_DN": "Entry missing required DN attribute",
            "INVALID_CHANGETYPE": "Invalid changetype value",
            "INVALID_ATTRIBUTE_NAME": "Invalid LDAP attribute name",
            "INVALID_BASE64_VALUE": "Invalid base64 encoded value",
            "INVALID_URL_VALUE": "Invalid URL reference value",
            "DUPLICATE_DN": "Duplicate DN found in LDIF",
            "INVALID_MODIFICATION": "Invalid modification operation",
            "MISSING_OBJECTCLASS": "Entry missing required objectClass",
            "SCHEMA_VIOLATION": "Entry violates LDAP schema rules",
            "INVALID_SYNTAX": "Invalid LDIF syntax",
            "LINE_TOO_LONG": "Line exceeds maximum allowed length",
            "INVALID_CONTINUATION": "Invalid line continuation",
            "UNSUPPORTED_VERSION": "Unsupported LDIF version",
            "INVALID_CONTROL": "Invalid LDIF control",
        }

        # Processing Analytics Constants
        ANALYTICS_BATCH_SIZE: Final[int] = 5000
        METRICS_COLLECTION_INTERVAL: Final[int] = 1000
        PROGRESS_REPORT_INTERVAL: Final[int] = 10000
        MEMORY_CHECK_INTERVAL: Final[int] = 50000

        # Memory Management
        LOW_MEMORY_THRESHOLD_MB: Final[int] = 100
        CRITICAL_MEMORY_THRESHOLD_MB: Final[int] = 50
        GC_COLLECTION_THRESHOLD: Final[int] = 10000

        # File Processing
        DEFAULT_CHUNK_SIZE: Final[int] = 1024 * 1024  # 1MB
        MAX_FILE_SIZE_MB: Final[int] = 2048  # 2GB
        BACKUP_FILE_EXTENSION: Final[str] = ".bak"
        TEMP_FILE_PREFIX: Final[str] = "ldif_temp_"

        # Network and Connection Constants
        LDAP_DEFAULT_PORT: Final[int] = 389
        LDAPS_DEFAULT_PORT: Final[int] = 636
        CONNECTION_TIMEOUT: Final[int] = 30
        SEARCH_TIMEOUT: Final[int] = 60
        BIND_TIMEOUT: Final[int] = 10

        # Schema Validation Constants
        REQUIRED_PERSON_ATTRIBUTES: Final[frozenset[str]] = frozenset({
            "cn",
            "sn",
            "objectClass",
        })

        REQUIRED_ORG_ATTRIBUTES: Final[frozenset[str]] = frozenset({"o", "objectClass"})

        REQUIRED_ORGUNIT_ATTRIBUTES: Final[frozenset[str]] = frozenset({
            "ou",
            "objectClass",
        })

        # Performance Tuning Constants
        OPTIMAL_THREAD_COUNT: Final[int] = 4
        MAX_THREAD_COUNT: Final[int] = 16
        THREAD_POOL_TIMEOUT: Final[int] = 300
        QUEUE_MAX_SIZE: Final[int] = 10000

        # Error Recovery Constants
        MAX_RETRY_ATTEMPTS: Final[int] = 3
        RETRY_DELAY_SECONDS: Final[int] = 1
        EXPONENTIAL_BACKOFF_MULTIPLIER: Final[float] = 2.0
        MAX_RETRY_DELAY_SECONDS: Final[int] = 60

        # Export/Import Configuration
        EXPORT_BATCH_SIZE: Final[int] = 2000
        IMPORT_BATCH_SIZE: Final[int] = 1000
        PARALLEL_EXPORT_THRESHOLD: Final[int] = 50000
        PARALLEL_IMPORT_THRESHOLD: Final[int] = 20000

        # Quality Assurance Constants
        DUPLICATE_CHECK_ENABLED: Final[bool] = True
        SCHEMA_VALIDATION_ENABLED: Final[bool] = True
        SYNTAX_VALIDATION_ENABLED: Final[bool] = True
        REFERENCE_VALIDATION_ENABLED: Final[bool] = False

    class OracleWMS:
        """Oracle WMS (Warehouse Management System) constants.

        Comprehensive constants for Oracle WMS API integration, entity management,
        data processing, and system operations.
        """

        # Core system constants
        NAME: Final[str] = "flext-oracle-wms"
        VERSION: Final[str] = "0.9.0"
        DEFAULT_ENVIRONMENT: Final[str] = "default"

        # API configuration constants
        API_VERSIONS: Final[list[str]] = ["v10", "v9", "v8"]
        DEFAULT_API_VERSION: Final[str] = "v10"
        LGF_API_BASE: Final[str] = "/wms/lgfapi"

        # API timeouts and limits
        DEFAULT_TIMEOUT: Final[float] = 30.0
        DEFAULT_MAX_RETRIES: Final[int] = 3
        DEFAULT_RETRY_DELAY: Final[float] = 1.0

        # HTTP status codes
        HTTP_OK: Final[int] = 200
        HTTP_BAD_REQUEST: Final[int] = 400
        HTTP_UNAUTHORIZED: Final[int] = 401
        HTTP_FORBIDDEN: Final[int] = 403
        MIN_HTTP_STATUS_CODE: Final[int] = 100
        MAX_HTTP_STATUS_CODE: Final[int] = 599
        AUTH_ERROR_CODES: Final[tuple[int, ...]] = (401, 403)

        # Authentication configuration
        AUTH_METHODS: Final[list[str]] = ["basic", "bearer", "api_key"]
        MIN_TOKEN_LENGTH: Final[int] = 10
        MIN_API_KEY_LENGTH: Final[int] = 10

        # Entity types - Core entities
        CORE_ENTITIES: Final[list[str]] = ["company", "facility", "location", "item"]
        ORDER_ENTITIES: Final[list[str]] = ["order_hdr", "order_dtl"]
        INVENTORY_ENTITIES: Final[list[str]] = ["inventory", "allocation"]
        MOVEMENT_ENTITIES: Final[list[str]] = ["pick_hdr", "pick_dtl"]
        SHIPMENT_ENTITIES: Final[list[str]] = ["shipment", "oblpn"]

        # Entity validation
        MAX_ENTITY_NAME_LENGTH: Final[int] = 100
        ENTITY_NAME_PATTERN: Final[str] = r"^[a-z0-9_]+$"

        # Filtering constants
        FILTER_OPERATORS: Final[list[str]] = [
            "eq",
            "ne",
            "gt",
            "ge",
            "lt",
            "le",
            "in",
            "not_in",
            "like",
            "not_like",
        ]
        MAX_FILTER_CONDITIONS: Final[int] = 50

        # Pagination configuration
        PAGINATION_MODES: Final[list[str]] = ["offset", "cursor", "token"]
        DEFAULT_PAGE_SIZE: Final[int] = 100
        MAX_PAGE_SIZE: Final[int] = 1000
        MIN_PAGE_SIZE: Final[int] = 1

        # Processing configuration
        WRITE_MODES: Final[list[str]] = ["insert", "update", "upsert", "delete"]
        DEFAULT_BATCH_SIZE: Final[int] = 50
        MAX_BATCH_SIZE: Final[int] = 500

        # Rate limiting
        DEFAULT_RATE_LIMIT: Final[int] = 60  # requests per minute
        MIN_REQUEST_DELAY: Final[float] = 0.1  # seconds

        # Schema discovery
        DEFAULT_SAMPLE_SIZE: Final[int] = 100
        MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.7
        MAX_SCHEMA_DEPTH: Final[int] = 10

        # Connection management
        DEFAULT_POOL_SIZE: Final[int] = 5
        MAX_POOL_SIZE: Final[int] = 20
        DEFAULT_CONNECT_TIMEOUT: Final[int] = 10
        DEFAULT_READ_TIMEOUT: Final[int] = 30

        # Caching configuration
        DEFAULT_CACHE_TTL: Final[int] = 300  # 5 minutes
        MAX_CACHE_SIZE: Final[int] = 1000

        # API paths
        ENTITY_DISCOVERY_PATH: Final[str] = "/entity/"
        ENTITY_DATA_PATH: Final[str] = "/entity/{entity_name}/"
        ENTITY_BY_ID_PATH: Final[str] = "/entity/{entity_name}/{entity_id}/"
        METADATA_BASE_PATH: Final[str] = "/metadata"
        SCHEMA_BASE_PATH: Final[str] = "/schema"
        INIT_STAGE_PATH: Final[str] = "/init_stage_interface/"
        RUN_STAGE_PATH: Final[str] = "/run_stage_interface/"
        STATUS_CHECK_PATH: Final[str] = "/status/"
        HEALTH_CHECK_PATH: Final[str] = "/health/"

        # Response field names
        RESULT_COUNT_FIELD: Final[str] = "result_count"
        PAGE_COUNT_FIELD: Final[str] = "page_count"
        PAGE_NUMBER_FIELD: Final[str] = "page_nbr"
        NEXT_PAGE_FIELD: Final[str] = "next_page"
        PREVIOUS_PAGE_FIELD: Final[str] = "previous_page"
        RESULTS_FIELD: Final[str] = "results"
        DATA_FIELD: Final[str] = "data"
        ID_FIELD: Final[str] = "id"
        URL_FIELD: Final[str] = "url"
        CREATE_USER_FIELD: Final[str] = "create_user"
        CREATE_TS_FIELD: Final[str] = "create_ts"
        MOD_USER_FIELD: Final[str] = "mod_user"
        MOD_TS_FIELD: Final[str] = "mod_ts"
        STATUS_FIELD: Final[str] = "status"
        MESSAGE_FIELD: Final[str] = "message"
        ERROR_FIELD: Final[str] = "error"

        # Error messages
        CONNECTION_FAILED_MSG: Final[str] = "Failed to connect to Oracle WMS"
        AUTHENTICATION_FAILED_MSG: Final[str] = "Authentication failed"
        TIMEOUT_ERROR_MSG: Final[str] = "Request timeout"
        API_ERROR_MSG: Final[str] = "Oracle WMS API error"
        INVALID_ENDPOINT_MSG: Final[str] = "Invalid API endpoint"
        INVALID_RESPONSE_MSG: Final[str] = "Invalid API response format"
        ENTITY_NOT_FOUND_MSG: Final[str] = "Entity not found"
        INVALID_ENTITY_TYPE_MSG: Final[str] = "Invalid entity type"
        ENTITY_VALIDATION_FAILED_MSG: Final[str] = "Entity validation failed"
        INVALID_DATA_FORMAT_MSG: Final[str] = "Invalid data format"
        DATA_VALIDATION_FAILED_MSG: Final[str] = "Data validation failed"
        SCHEMA_GENERATION_FAILED_MSG: Final[str] = "Schema generation failed"
        FLATTENING_FAILED_MSG: Final[str] = "Data flattening failed"
        DISCOVERY_FAILED_MSG: Final[str] = "Entity discovery failed"
        PROCESSING_FAILED_MSG: Final[str] = "Data processing failed"

    class AlgarMigration:
        """ALGAR Oracle to OUD migration constants.

        Comprehensive constants for ALGAR Telecom's Oracle Unified Directory
        migration system including phases, processing limits, and configurations.
        """

        # Core project constants
        PROJECT_NAME: Final[str] = "algar-oud-mig"
        DESCRIPTION: Final[str] = "ALGAR Oracle to OUD Migration Tool"
        AUTHOR: Final[str] = "Algar Telecom"
        LICENSE: Final[str] = "Proprietary"

        # Supported formats
        SUPPORTED_INPUT_FORMATS: Final[tuple[str, ...]] = ("ldif",)
        SUPPORTED_OUTPUT_FORMATS: Final[tuple[str, ...]] = ("ldif",)

        # Migration phases
        MIGRATION_PHASES: Final[list[str]] = [
            "validation",
            "parsing",
            "schema",
            "hierarchy",
            "users",
            "groups",
            "acls",
            "categorization",
            "output_generation",
            "synchronization",
            "completed",
            "failed",
            "data",
        ]

        # Migration statuses
        MIGRATION_STATUSES: Final[list[str]] = [
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        ]

        # OUD (Oracle Unified Directory) connection constants
        OUD_DEFAULT_HOST: Final[str] = "localhost"
        OUD_DEFAULT_PORT: Final[int] = 3389
        OUD_DOCKER_MAPPED_PORT: Final[int] = 11389
        OUD_DEFAULT_BIND_DN: Final[str] = "cn=orcladmin"
        OUD_DEFAULT_BASE_DN: Final[str] = "dc=ctbc,dc=com"
        OUD_BASE_DN_SHORT: Final[str] = "dc=ctbc"
        OUD_DEFAULT_USE_SSL: Final[bool] = False
        OUD_DEFAULT_TIMEOUT: Final[int] = 30
        OUD_DEFAULT_POOL_SIZE: Final[int] = 10
        OUD_DEFAULT_MAX_RETRIES: Final[int] = 3
        OUD_MAX_PORT_VALUE: Final[int] = 65535
        OUD_BASE_DN_PATTERN: Final[str] = r"dc=ctbc$"

        # Processing configuration
        DEFAULT_BATCH_SIZE: Final[int] = 100
        MAX_BATCH_SIZE: Final[int] = 10000
        DEFAULT_MAX_WORKERS: Final[int] = 4
        MAX_LINE_LENGTH: Final[int] = 2000
        MAX_ENTRIES_PER_FILE: Final[int] = 50000
        CHUNK_SIZE: Final[int] = 8192
        ENABLE_TRANSFORMATIONS: Final[bool] = True

        # Schema filtering
        SCHEMA_OID_PATTERNS: Final[tuple[str, ...]] = ("99.*",)
        SCHEMA_EXACT_OIDS: Final[tuple[str, ...]] = ("1.3.6.1.1.1.2.0",)

        # Entry categories for migration
        ENTRY_CATEGORIES: Final[tuple[str, ...]] = (
            "hierarchy",
            "groups",
            "users",
            "roles",
            "resources",
            "policies",
            "other",
        )

        # Main categories for business rules validation
        MAIN_ENTRY_CATEGORIES: Final[tuple[str, ...]] = ("hierarchy", "users", "groups")

        # Category processing order
        CATEGORY_PROCESSING_ORDER: Final[tuple[str, ...]] = (
            "hierarchy",
            "policies",
            "groups",
            "users",
            "roles",
            "resources",
            "other",
        )

        # File extensions and encoding
        LDIF_EXTENSION: Final[str] = ".ldif"
        UTF8_ENCODING: Final[str] = "utf-8"

        # Processing limits
        MAX_ENTRIES_DISCOVERY: Final[int] = 1000
        MAX_LOG_STRING_LENGTH: Final[int] = 100
        PERCENTAGE_MULTIPLIER: Final[float] = 100.0
        ERROR_MESSAGE_TRUNCATE_LENGTH: Final[int] = 100

        # Error ranges
        MIGRATION_ERROR_RANGE_START: Final[int] = 5000
        MIGRATION_ERROR_RANGE_END: Final[int] = 5999

        # Default configuration
        DEFAULT_TARGET_SCHEMA: Final[str] = "00_custom_schema_oud.ldif"
        DEFAULT_SKIP_VALIDATION: Final[bool] = False

        # Timeouts
        COMMAND_TIMEOUT: Final[int] = 300
        OPERATION_TIMEOUT: Final[int] = 300
        CONNECTION_TIMEOUT: Final[int] = 30

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

    class Reliability:
        """Reliability constants for the FLEXT ecosystem.

        This class contains reliability-related constants including retry policies,
        timeout values, circuit breaker thresholds, and error handling parameters
        following reliability engineering best practices.

        Architecture Principles Applied:
            - Single Responsibility: Only reliability configuration
            - Fault Tolerance: Constants support resilience patterns
            - Circuit Breaking: Thresholds for failure isolation
        """

        # Retry configuration
        MAX_RETRY_ATTEMPTS: Final[int] = 3
        DEFAULT_RETRY_DELAY: Final[float] = 1.0
        EXPONENTIAL_BACKOFF_MULTIPLIER: Final[float] = 2.0

        # Circuit breaker thresholds
        CIRCUIT_BREAKER_FAILURE_THRESHOLD: Final[int] = 5
        CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[int] = 60
        CIRCUIT_BREAKER_SUCCESS_THRESHOLD: Final[int] = 3

        # Timeout configuration (seconds)
        DEFAULT_OPERATION_TIMEOUT: Final[float] = 30.0
        SHORT_TIMEOUT: Final[float] = 5.0
        LONG_TIMEOUT: Final[float] = 300.0

        # Health check configuration
        HEALTH_CHECK_TIMEOUT: Final[float] = 10.0
        HEALTH_CHECK_RETRIES: Final[int] = 2

    # NEW: Handler system constants (ADDED from string mapping analysis)
    class Handlers:
        """Handler system constants for command/query processing.

        Consolidated from handlers.py and commands.py to centralize all
        handler-related constants in one place following SOLID principles.
        """

        # Handler execution limits
        MAX_CHAIN_HANDLERS: Final[int] = 50
        MAX_PIPELINE_STAGES: Final[int] = 20
        MAX_COMMAND_HANDLERS: Final[int] = 10000  # From commands.py

        # Handler execution parameters
        RETRY_DELAY: Final[float] = 1.0
        MEMORY_THRESHOLD_MB: Final[int] = 100
        METRICS_COLLECTION_INTERVAL: Final[int] = 60
        SLOW_HANDLER_THRESHOLD: Final[float] = (
            5.0  # From Performance.SLOW_QUERY_THRESHOLD
        )

        # Handler states
        HANDLER_STATE_IDLE: Final[str] = "idle"
        HANDLER_STATE_PROCESSING: Final[str] = "processing"
        HANDLER_STATE_COMPLETED: Final[str] = "completed"
        HANDLER_STATE_FAILED: Final[str] = "failed"
        HANDLER_STATE_TIMEOUT: Final[str] = "timeout"
        HANDLER_STATE_PAUSED: Final[str] = "paused"

        # Handler types
        HANDLER_TYPE_BASIC: Final[str] = "basic"
        HANDLER_TYPE_VALIDATING: Final[str] = "validating"
        HANDLER_TYPE_AUTHORIZING: Final[str] = "authorizing"
        HANDLER_TYPE_METRICS: Final[str] = "metrics"
        HANDLER_TYPE_COMMAND: Final[str] = "command"
        HANDLER_TYPE_QUERY: Final[str] = "query"
        HANDLER_TYPE_EVENT: Final[str] = "event"
        HANDLER_TYPE_PIPELINE: Final[str] = "pipeline"
        HANDLER_TYPE_CHAIN: Final[str] = "chain"

        # Command constants (from commands.py)
        MIN_COMMAND_ID_LENGTH: Final[int] = 2  # Same as MIN_SERVICE_NAME_LENGTH
        MAX_COMMAND_TYPE_LENGTH: Final[int] = 64  # Same as MAX_SERVICE_NAME_LENGTH
        MAX_CORRELATION_ID_LENGTH: Final[int] = 64
        MAX_QUERY_RESULTS: Final[int] = 10000  # Same as MAX_BATCH_SIZE
        SLOW_COMMAND_THRESHOLD: Final[int] = 30  # Same as Performance.TIMEOUT
        COMMAND_TIMEOUT_ERROR: Final[str] = "Command execution timed out"

        # Handler registration errors
        HANDLER_NOT_FOUND: Final[str] = "Handler not found"
        HANDLER_ALREADY_REGISTERED: Final[str] = "Handler already registered"
        HANDLER_NOT_CALLABLE: Final[str] = "Handler is not callable"
        EVENT_PROCESSING_FAILED: Final[str] = "Event processing failed"

    # =========================================================================
    # COMMAND CONSTANTS - CQRS command patterns
    # =========================================================================

    class Commands:
        """Command system constants for CQRS patterns.

        Consolidated from commands.py to centralize command-related constants.
        """

        # Command states
        STATE_CREATED: Final[str] = "created"
        STATE_VALIDATING: Final[str] = "validating"
        STATE_PROCESSING: Final[str] = "processing"
        STATE_COMPLETED: Final[str] = "completed"
        STATE_FAILED: Final[str] = "failed"
        STATE_TIMEOUT: Final[str] = "timeout"
        STATE_CANCELLED: Final[str] = "cancelled"

        # Command types
        TYPE_CREATE: Final[str] = "create"
        TYPE_UPDATE: Final[str] = "update"
        TYPE_DELETE: Final[str] = "delete"
        TYPE_QUERY: Final[str] = "query"
        TYPE_BATCH: Final[str] = "batch"
        TYPE_ASYNC: Final[str] = "async"

    # =========================================================================
    # UTILITIES CONSTANTS - Helper utilities
    # =========================================================================

    class Utilities:
        """Utilities constants for helper functions.

        Consolidated from utilities.py to centralize utility constants.
        """

        # Time conversion constants
        SECONDS_PER_MINUTE: Final[int] = 60
        SECONDS_PER_HOUR: Final[int] = 3600

        # Size conversion constants
        BYTES_PER_KB: Final[int] = 1024
        BYTES_PER_MB: Final[int] = 1024 * 1024
        BYTES_PER_GB: Final[int] = 1024 * 1024 * 1024

    # =========================================================================
    # API CONSTANTS - API and service communication
    # =========================================================================

    class Api:
        """API constants for service communication across FLEXT ecosystem.

        Consolidated API-related constants from flext-api, flext-auth, etc.
        """

        # API timeouts
        DEFAULT_API_TIMEOUT: Final[float] = 30.0
        GRPC_DEFAULT_TIMEOUT: Final[float] = 30.0
        WMS_DEFAULT_TIMEOUT: Final[float] = 30.0

        # Batch processing
        DEFAULT_BATCH_SIZE: Final[int] = 1000
        MAX_BATCH_SIZE: Final[int] = 10000

        # Rate limiting
        DEFAULT_RATE_LIMIT: Final[int] = 100
        MAX_RATE_LIMIT: Final[int] = 1000

    # =========================================================================
    # AUTH CONSTANTS - Authentication and authorization
    # =========================================================================

    class Auth:
        """Authentication constants across FLEXT ecosystem.

        Consolidated from flext-auth and other auth-related modules.
        """

        # Token expiration
        DEFAULT_TOKEN_EXPIRY: Final[int] = 3600  # 1 hour
        MAX_TOKEN_EXPIRY: Final[int] = 86400  # 24 hours

        # Password requirements
        MIN_PASSWORD_LENGTH: Final[int] = 8
        MAX_PASSWORD_LENGTH: Final[int] = 128

        # Username validation (centralized from flext-auth validation.py)
        MAX_USERNAME_LENGTH: Final[int] = 50
        MIN_USERNAME_LENGTH: Final[int] = 3
        MAX_COMMON_LENGTH: Final[int] = 12

        # Session management
        DEFAULT_SESSION_TIMEOUT: Final[int] = 1800  # 30 minutes
        MAX_SESSIONS_PER_USER: Final[int] = 5

    # =========================================================================
    # DATABASE CONSTANTS - Database connection and operations
    # =========================================================================

    class Database:
        """Database constants for Oracle, LDAP, and other databases.

        Consolidated from flext-db-oracle, flext-ldap, etc.
        """

        # Connection timeouts
        DEFAULT_DB_TIMEOUT: Final[int] = 30
        MAX_CONNECTION_RETRIES: Final[int] = 3
        CONNECTION_RETRY_DELAY: Final[float] = 1.0

        # Query limits
        DEFAULT_QUERY_LIMIT: Final[int] = 1000
        MAX_QUERY_LIMIT: Final[int] = 10000

        # Oracle specific
        ORACLE_DEFAULT_PORT: Final[int] = 1521
        ORACLE_MAX_CONNECTIONS: Final[int] = 100

        # LDAP specific
        LDAP_DEFAULT_PORT: Final[int] = 389
        LDAP_SSL_PORT: Final[int] = 636

    # =========================================================================
    # LEGACY CONSTANTS - Backward compatibility
    # =========================================================================

    class Legacy:
        """Legacy constants for backward compatibility.

        Consolidated from legacy.py to maintain compatibility.
        """

        # Field types (from legacy.py)
        FIELD_TYPE_STRING: Final[str] = "string"
        FIELD_TYPE_INTEGER: Final[str] = "integer"
        FIELD_TYPE_BOOLEAN: Final[str] = "boolean"
        FIELD_TYPE_FLOAT: Final[str] = "float"
        FIELD_TYPE_EMAIL: Final[str] = "email"
        FIELD_TYPE_UUID: Final[str] = "uuid"
        FIELD_TYPE_DATETIME: Final[str] = "datetime"
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


FlextOperationStatus = FlextConstants.Enums.OperationStatus
FlextLogLevel = FlextConstants.Enums.LogLevel
FlextFieldType = FlextConstants.Enums.FieldType
FlextEnvironment = FlextConstants.Enums.Environment
FlextEntityStatus = FlextConstants.Enums.EntityStatus
FlextDataFormat = FlextConstants.Enums.DataFormat
FlextConnectionType = FlextConstants.Enums.ConnectionType


# =============================================================================
# LEGACY CONSTANTS MOVED TO LEGACY.PY
# =============================================================================
# Legacy flat constants ERROR_CODES, MESSAGES, STATUS_CODES, VALIDATION_RULES,
# DEFAULT_TIMEOUT, DEFAULT_RETRIES, DEFAULT_PAGE_SIZE, VERSION, NAME, and
# pattern aliases have been moved to legacy.py with deprecation warnings.
#
# NEW USAGE: Use proper FlextConstants nested structure
#   FlextConstants.Errors.VALIDATION_ERROR
#   FlextConstants.Defaults.TIMEOUT
#   FlextConstants.Patterns.EMAIL_PATTERN
#
# Use FlextConstants hierarchical structure for new code

# =============================================================================
# EXPORTS - Semantic constants + compatibility
# =============================================================================

# Simplified ERROR_CODES mapping
ERROR_CODES: dict[str, str] = {
    "GENERIC_ERROR": FlextConstants.Errors.GENERIC_ERROR,
    "VALIDATION_ERROR": "FLEXT_VALIDATION_ERROR",
    "CONNECTION_ERROR": FlextConstants.Errors.CONNECTION_ERROR,
    "TIMEOUT_ERROR": FlextConstants.Errors.TIMEOUT_ERROR,
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
MESSAGES = FlextConstants.Messages
SERVICE_NAME_EMPTY: Final[str] = "Service name cannot be empty"

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Legacy alias for FlextConstants - maintain backward compatibility
FlextCoreConstants = FlextConstants


# =============================================================================
# DOMAIN-SPECIFIC CONSTANT FACADES - Domain inheritance patterns
# =============================================================================


class FlextWebConstants(FlextConstants):
    """Web domain-specific constants facade extending FlextConstants."""


class FlextCliConstants(FlextConstants):
    """CLI domain-specific constants facade extending FlextConstants."""


class FlextObservabilityConstants(FlextConstants):
    """Observability domain-specific constants facade extending FlextConstants."""


class FlextQualityConstants(FlextConstants):
    """Quality domain-specific constants facade extending FlextConstants."""


class FlextTargetConstants(FlextConstants):
    """Target domain-specific constants facade extending FlextConstants."""


class FlextTapConstants(FlextConstants):
    """Tap domain-specific constants facade extending FlextConstants."""


class FlextMeltanoConstants(FlextConstants):
    """Meltano domain-specific constants facade extending FlextConstants."""


class FlextLdifConstants(FlextConstants):
    """LDIF domain-specific constants facade extending FlextConstants."""


class FlextOracleWmsConstants(FlextConstants):
    """Oracle WMS domain-specific constants facade extending FlextConstants."""


class FlextAlgarConstants(FlextConstants):
    """ALGAR Migration domain-specific constants facade extending FlextConstants."""


__all__: Final[list[str]] = [
    "FlextAlgarConstants",
    "FlextCliConstants",
    "FlextConstants",  # Main class with all constants
    "FlextEntityStatus",  # Entity status enum alias
    "FlextLdifConstants",
    "FlextMeltanoConstants",
    "FlextObservabilityConstants",
    "FlextOracleWmsConstants",
    "FlextQualityConstants",
    "FlextTapConstants",
    "FlextTargetConstants",
    # Domain-specific facades
    "FlextWebConstants",
]
