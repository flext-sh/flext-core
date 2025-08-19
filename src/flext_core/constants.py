"""Core constants and enums."""

from __future__ import annotations

from enum import Enum
from typing import ClassVar


class FlextConstants:
    """Central container for all ecosystem constants.

    Organizes constants into logical categories with type safety.
    """

    # Class-level placeholder kept for backward compatibility;
    # the authoritative flat ERROR_CODES mapping is built at module level
    # after this class is fully defined.
    ERROR_CODES: ClassVar[dict[str, str]] = {}

    # Legacy compatibility - VERSION as direct attribute
    VERSION: ClassVar[str] = "0.9.0"

    class Core:
        """Core system constants."""

        NAME = "FLEXT"
        VERSION = "0.9.0"
        ECOSYSTEM_SIZE = 33
        PYTHON_VERSION = "3.13+"
        ARCHITECTURE = "clean_architecture"

        # Magic number replacements
        CONFIGURATION_ARGUMENT_INDEX_THRESHOLD = 2
        MAX_BRANCHES_ALLOWED = 12
        MAX_RETURN_STATEMENTS_ALLOWED = 6

    class Network:
        """Network and port constants."""

        # Port ranges
        MIN_PORT = 1
        MAX_PORT = 65535

        # Common service ports
        HTTP_PORT = 80
        HTTPS_PORT = 443
        LDAP_PORT = 389
        LDAPS_PORT = 636

    class Validation:
        """Validation constants."""

        # String length limits
        MIN_SERVICE_NAME_LENGTH = 2
        MAX_SERVICE_NAME_LENGTH = 64
        MIN_SECRET_KEY_LENGTH = 32

        # Percentage limits
        MIN_PERCENTAGE = 0.0
        MAX_PERCENTAGE = 100.0

    class Errors:
        """Error codes and messages."""

        # Error Code Ranges
        BUSINESS_ERROR_RANGE = (1000, 1999)
        TECHNICAL_ERROR_RANGE = (2000, 2999)
        VALIDATION_ERROR_RANGE = (3000, 3999)
        SECURITY_ERROR_RANGE = (4000, 4999)

        # Common Error Codes (following pattern structure)
        GENERIC_ERROR = "FLEXT_0001"
        VALIDATION_ERROR = "FLEXT_3001"
        BUSINESS_RULE_VIOLATION = "FLEXT_1001"
        AUTHORIZATION_DENIED = "FLEXT_4001"
        AUTHENTICATION_FAILED = "FLEXT_4002"
        RESOURCE_NOT_FOUND = "FLEXT_1004"
        DUPLICATE_RESOURCE = "FLEXT_1005"
        CONNECTION_ERROR = "FLEXT_2001"
        TIMEOUT_ERROR = "FLEXT_2002"
        CONFIGURATION_ERROR = "FLEXT_2003"
        SERIALIZATION_ERROR = "FLEXT_2004"
        EXTERNAL_SERVICE_ERROR = "FLEXT_2005"

        # Legacy compatibility (existing ecosystem)
        BUSINESS_RULE_ERROR = "BUSINESS_RULE_ERROR"  # Referenced by semantic.py
        AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"  # Referenced by semantic.py
        TYPE_ERROR = "TYPE_ERROR"
        UNWRAP_ERROR = "UNWRAP_ERROR"
        CLI_ERROR = "CLI_ERROR"
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
        RETRY_ERROR = "RETRY_ERROR"
        CIRCUIT_BREAKER_ERROR = "CIRCUIT_BREAKER_ERROR"
        CONCURRENCY_ERROR = "CONCURRENCY_ERROR"
        RESOURCE_ERROR = "RESOURCE_ERROR"
        DATABASE_ERROR = "DATABASE_ERROR"
        API_ERROR = "API_ERROR"
        EVENT_ERROR = "EVENT_ERROR"
        SECURITY_ERROR = "SECURITY_ERROR"
        AUTH_ERROR = "AUTH_ERROR"
        CRITICAL_ERROR = "CRITICAL_ERROR"
        EXTERNAL_ERROR = "EXTERNAL_ERROR"
        MIGRATION_ERROR = "MIGRATION_ERROR"
        PERMISSION_ERROR = "PERMISSION_ERROR"
        PROCESSING_ERROR = "PROCESSING_ERROR"
        CHAINED_ERROR = "CHAINED_ERROR"
        RETRYABLE_ERROR = "RETRYABLE_ERROR"
        UNKNOWN_ERROR = "UNKNOWN_ERROR"
        FALLBACK_FAILURE = "FALLBACK_FAILURE"
        OPERATION_AND_FALLBACK_FAILURE = "OPERATION_AND_FALLBACK_FAILURE"
        OPERATION_FAILURE = "OPERATION_FAILURE"
        NOT_FOUND = "NOT_FOUND"
        ALREADY_EXISTS = "ALREADY_EXISTS"
        CONFIG_ERROR = "CONFIG_ERROR"

        # Error Messages
        MESSAGES: ClassVar[dict[str, str]] = {
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
        }

    class Messages:
        """System messages."""

        SUCCESS = "Operation completed successfully"
        STARTED = "Operation started"
        COMPLETED = "Operation completed"
        FAILED = "Operation failed"
        VALIDATING = "Validating input"
        CONFIGURING = "Configuring system"
        UNKNOWN_ERROR = "Unknown error occurred"
        FILTER_FAILED = "Filter condition not met"
        VALIDATION_FAILED = "Validation failed"
        OPERATION_FAILED = "Operation failed"
        UNWRAP_FAILED = "Cannot unwrap failed result"
        NULL_DATA = "Result data is None"
        INVALID_INPUT = "Invalid input provided"
        TYPE_MISMATCH = "Type mismatch error"
        ENTITY_ID_EMPTY = "Entity ID cannot be empty"
        SERVICE_NAME_EMPTY = "Service name cannot be empty"
        NAME_EMPTY = "Name cannot be empty"
        MESSAGE_EMPTY = "Message cannot be empty"
        EVENT_TYPE_EMPTY = "Event type cannot be empty"
        VALUE_EMPTY = "Value cannot be empty"
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
        """Status values."""

        ACTIVE = "active"
        INACTIVE = "inactive"
        PENDING = "pending"
        COMPLETED = "completed"
        FAILED = "failed"
        RUNNING = "running"
        CANCELLED = "cancelled"
        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"
        PROCESSING = "PROCESSING"

    class Patterns:
        """Validation patterns."""

        # Identifiers
        UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
        IDENTIFIER_PATTERN = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        SERVICE_NAME_PATTERN = r"^[a-zA-Z][a-zA-Z0-9_-]*$"

        # Authentication
        EMAIL_PATTERN = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        USERNAME_PATTERN = r"^[a-zA-Z0-9_-]{3,32}$"
        # This is a regex pattern for credential validation, not a hardcoded password
        CREDENTIAL_STRENGTH_PATTERN = (
            r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
        )

        # Network
        IPV4_PATTERN = (
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
            r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        )
        HOSTNAME_PATTERN = (
            r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
            r"(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
        )
        URL_PATTERN = r"^https?://.+"

        # Versioning
        SEMVER_PATTERN = (
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
            r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
            r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
            r"?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        )
        SEMANTIC_VERSION_PATTERN = r"^\d+\.\d+\.\d+(-\w+(\.\d+)?)?$"

    class Defaults:
        """Default values."""

        # Network and timeouts (CONSOLIDATED - was duplicated across patterns)
        TIMEOUT = 30
        MAX_RETRIES = 3
        CONNECTION_TIMEOUT = 10

        # Pagination and limits
        PAGE_SIZE = 100
        MAX_PAGE_SIZE = 1000

        # Configuration defaults (MOVED FROM Config pattern - ELIMINATES DUPLICATION)
        CONFIG_TIMEOUT = 30
        CONFIG_RETRIES = 3

        # CLI defaults (MOVED FROM CLI pattern - ELIMINATES DUPLICATION)
        CLI_HELP_WIDTH = 80
        CLI_TIMEOUT = 30

        # Model defaults (MOVED FROM Pydantic pattern - ELIMINATES DUPLICATION)
        MODEL_TIMEOUT = 30
        VALIDATION_TIMEOUT = 5

        # Database defaults
        DB_TIMEOUT = 30
        DB_POOL_SIZE = 10

        # Cache defaults
        CACHE_TTL = 300  # 5 minutes
        CACHE_MAX_SIZE = 1000

    class Limits:
        """System limits."""

        MAX_STRING_LENGTH = 1000
        MAX_LIST_SIZE = 10000
        MIN_PASSWORD_LENGTH = 8
        MAX_PASSWORD_LENGTH = 128
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        MAX_BATCH_SIZE = 10000
        MIN_PORT = 1
        MAX_PORT = 65535
        MAX_THREADS = 100

    class Performance:
        """Performance constants."""

        DEFAULT_BATCH_SIZE = 1000
        MAX_BATCH_SIZE = 10000
        CACHE_TTL = 300  # 5 minutes
        CACHE_MAX_SIZE = 1000
        POOL_SIZE = 10
        MAX_CONNECTIONS = 100
        KEEP_ALIVE_TIMEOUT = 60

        # Performance Thresholds (from pattern documentation)
        SLOW_QUERY_THRESHOLD = 1.0  # seconds
        SLOW_REQUEST_THRESHOLD = 2.0  # seconds
        HIGH_MEMORY_THRESHOLD = 0.8  # 80%
        HIGH_CPU_THRESHOLD = 0.8  # 80%

        # Monitoring
        METRICS_INTERVAL = 60  # seconds
        HEALTH_CHECK_INTERVAL = 30  # seconds

        # Time calculation constants (consolidated from utilities.py - SOLID SRP)
        SECONDS_PER_MINUTE = 60
        SECONDS_PER_HOUR = 3600
        BYTES_PER_KB = 1024

    # NEW: Configuration system constants (MOVED FROM Config/CLI patterns)
    class Configuration:
        """Configuration system constants."""

        # Provider priorities (MOVED FROM Config pattern)
        CLI_PRIORITY = 1
        ENV_PRIORITY = 2
        DOTENV_PRIORITY = 3
        CONFIG_FILE_PRIORITY = 4
        CONSTANTS_PRIORITY = 5

        # Configuration files (MOVED FROM Config pattern)
        DOTENV_FILES: ClassVar[list[str]] = [".env", ".env.local", ".env.production"]
        CONFIG_FILES: ClassVar[list[str]] = [
            "config.json",
            "config.yaml",
            "flext.config.json",
        ]

        # Environment types
        ENVIRONMENTS: ClassVar[list[str]] = [
            "development",
            "staging",
            "production",
            "test",
        ]
        DEFAULT_ENVIRONMENT = "development"

        # Configuration validation
        REQUIRED_FIELDS: ClassVar[list[str]] = ["REQUIRED"]
        OPTIONAL_FIELDS: ClassVar[list[str]] = ["OPTIONAL"]

    # NEW: CLI system constants (MOVED FROM CLI pattern)
    class Cli:
        """CLI system constants."""

        # Standard arguments
        HELP_ARGS: ClassVar[list[str]] = ["--help", "-h"]
        VERSION_ARGS: ClassVar[list[str]] = ["--version", "-v"]
        CONFIG_ARGS: ClassVar[list[str]] = ["--config", "-c"]
        VERBOSE_ARGS: ClassVar[list[str]] = ["--verbose", "-vv"]

        # Prefixes
        LONG_PREFIX = "--"
        SHORT_PREFIX = "-"

        # Output formats
        OUTPUT_FORMATS: ClassVar[list[str]] = ["json", "yaml", "table", "csv"]
        DEFAULT_OUTPUT_FORMAT = "table"

        # Exit codes
        SUCCESS_EXIT_CODE = 0
        ERROR_EXIT_CODE = 1
        INVALID_USAGE_EXIT_CODE = 2

    # NEW: Infrastructure constants (Database, Cache, etc.)
    class Infrastructure:
        """Infrastructure constants."""

        # Database ports
        DEFAULT_DB_PORT = 5432
        DEFAULT_ORACLE_PORT = 1521
        DEFAULT_REDIS_PORT = 6379
        DEFAULT_MYSQL_PORT = 3306
        DEFAULT_MONGODB_PORT = 27017

        # Connection pools
        DEFAULT_POOL_SIZE = 10
        MAX_POOL_SIZE = 50
        MIN_POOL_SIZE = 1

        # Network
        DEFAULT_HOST = "localhost"
        LOCALHOST_ALIASES: ClassVar[list[str]] = ["localhost", "127.0.0.1", "::1"]
        DEFAULT_PROTOCOL = "http"
        SECURE_PROTOCOLS: ClassVar[list[str]] = ["https", "wss", "ssl"]

    # NEW: Model system constants (MOVED FROM a Pydantic pattern)
    class Models:
        """Model system constants."""

        # Validation settings (MOVED FROM a Pydantic pattern)
        VALIDATE_ASSIGNMENT = True
        USE_ENUM_VALUES = True
        STR_STRIP_WHITESPACE = True

        # Serialization settings (MOVED FROM a Pydantic pattern)
        ARBITRARY_TYPES_ALLOWED = True
        VALIDATE_DEFAULT = True

        # Model behavior (MOVED FROM a Pydantic pattern)
        EXTRA_FORBID = "forbid"
        EXTRA_ALLOW = "allow"
        EXTRA_IGNORE = "ignore"

        # Field types
        FIELD_TYPES: ClassVar[list[str]] = [
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
        ]

        # Model states
        MODEL_STATES: ClassVar[list[str]] = ["draft", "valid", "invalid", "frozen"]

    # NEW: Observability constants (MOVED from observability patterns)
    class Observability:
        """Observability constants."""

        LOG_LEVELS: ClassVar[list[str]] = [
            "TRACE",
            "DEBUG",
            "INFO",
            "WARN",
            "ERROR",
            "FATAL",
        ]
        DEFAULT_LOG_LEVEL = "INFO"

        # Trace level constant (consolidated from loggings.py - SOLID SRP)
        TRACE_LEVEL = 5

        SPAN_TYPES: ClassVar[list[str]] = ["business", "technical", "error"]
        METRIC_TYPES: ClassVar[list[str]] = ["counter", "histogram", "gauge"]
        ALERT_LEVELS: ClassVar[list[str]] = ["info", "warning", "error", "critical"]

        CORRELATION_ID_HEADER = "X-Correlation-ID"
        TRACE_ID_HEADER = "X-Trace-ID"

        # Serialization constants (consolidated from payload.py - SOLID SRP)
        FLEXT_SERIALIZATION_VERSION = "1.0.0"
        SERIALIZATION_FORMAT_JSON = "json"
        SERIALIZATION_FORMAT_JSON_COMPRESSED = "json_compressed"

    # NEW: Handler system constants (ADDED from string mapping analysis)
    class Handlers:
        """Handler system constants for command/query processing."""

        # Handler registration errors
        HANDLER_NOT_FOUND: ClassVar[str] = "Handler not found"
        HANDLER_ALREADY_REGISTERED: ClassVar[str] = "Handler already registered"
        HANDLER_NOT_CALLABLE: ClassVar[str] = "Handler is not callable"
        NO_HANDLER_REGISTERED: ClassVar[str] = "No handler registered for type"
        REGISTRY_NOT_FOUND: ClassVar[str] = "No handlers registry found"

        # Permission and authorization
        MISSING_PERMISSION: ClassVar[str] = "Missing permission"
        PERMISSION_DENIED: ClassVar[str] = "Permission denied"
        AUTH_REQUIRED: ClassVar[str] = "Authentication required"

        # Event processing
        EVENT_PROCESSING_FAILED: ClassVar[str] = "Event processing failed"
        EVENT_HANDLER_FAILED: ClassVar[str] = "Event handler failed"
        CHAIN_HANDLER_FAILED: ClassVar[str] = "Chain handler failed"
        METRICS_COLLECTION_FAILED: ClassVar[str] = "Metrics collection failed"

        # Handler validation errors
        REQUEST_CANNOT_BE_NONE: ClassVar[str] = "Request cannot be None"
        MESSAGE_CANNOT_BE_NONE: ClassVar[str] = "Message cannot be None"
        VALIDATION_FAILED: ClassVar[str] = "Validation failed"
        EVENT_MISSING_TYPE: ClassVar[str] = "Event missing event_type"
        NO_USER_IN_CONTEXT: ClassVar[str] = "No user in context"
        INVALID_PERMISSIONS_FORMAT: ClassVar[str] = (
            "Invalid permissions format in context"
        )
        HANDLER_NAME_EMPTY: ClassVar[str] = "Handler name must be a non-empty string"
        INVALID_HANDLER_PROVIDED: ClassVar[str] = "Invalid handler provided"
        HANDLER_NAME_MUST_BE_STRING: ClassVar[str] = "Handler name must be a string"
        NO_HANDLER_COULD_PROCESS: ClassVar[str] = "No handler could process the request"
        CHAIN_PROCESSING_FAILED: ClassVar[str] = "Chain processing failed"
        NOT_IMPLEMENTED: ClassVar[str] = "Not implemented"
        AUTHORIZATION_FAILED: ClassVar[str] = "Authorization failed"
        QUERY_HANDLER_NOT_IMPLEMENTED: ClassVar[str] = "Query handler not implemented"

        # Handler templates (for f-strings)
        HANDLER_NOT_FOUND_TEMPLATE: ClassVar[str] = (
            "Handler '{name}' not found. Available: {available}"
        )
        MISSING_PERMISSION_TEMPLATE: ClassVar[str] = "Missing permission: {permission}"
        EVENT_PROCESSING_FAILED_TEMPLATE: ClassVar[str] = (
            "Event processing failed: {error}"
        )
        HANDLER_FAILED_TEMPLATE: ClassVar[str] = "Handler failed: {error}"
        NO_HANDLER_FOR_TYPE_TEMPLATE: ClassVar[str] = (
            "No handler registered for {type_name}"
        )
        CHAIN_HANDLER_FAILED_TEMPLATE: ClassVar[str] = "Chain handler failed: {error}"
        METRICS_FAILED_TEMPLATE: ClassVar[str] = "Metrics collection failed: {error}"
        NO_HANDLER_FOR_COMMAND_TEMPLATE: ClassVar[str] = (
            "No handler registered for {command_type}"
        )
        NO_HANDLER_FOR_QUERY_TEMPLATE: ClassVar[str] = (
            "No handler registered for {query_type}"
        )

    # NEW: Entity system constants (ADDED from string mapping analysis)
    class Entities:
        """Entity system constants for domain modeling."""

        # Entity validation
        ENTITY_ID_INVALID: ClassVar[str] = "Invalid entity ID"
        ENTITY_ID_EMPTY: ClassVar[str] = "Entity ID cannot be empty"
        ENTITY_NAME_EMPTY: ClassVar[str] = "Entity name cannot be empty"
        ENTITY_VALIDATION_FAILED: ClassVar[str] = "Entity validation failed"

        # Cache operations
        CACHE_KEY_TEMPLATE: ClassVar[str] = "{class_name}:{id}"
        CACHE_KEY_HASH_TEMPLATE: ClassVar[str] = "{class_name}:{hash}"

        # Entity operations
        OPERATION_LOG_TEMPLATE: ClassVar[str] = "Operation: {operation}"
        ENTITY_CREATED: ClassVar[str] = "Entity created"
        ENTITY_UPDATED: ClassVar[str] = "Entity updated"
        ENTITY_DELETED: ClassVar[str] = "Entity deleted"

        # Entity state
        ENTITY_ACTIVE: ClassVar[str] = "Entity is active"
        ENTITY_INACTIVE: ClassVar[str] = "Entity is inactive"
        ENTITY_STATE_INVALID: ClassVar[str] = "Invalid entity state"

        # Entity templates
        INVALID_ENTITY_ID_TEMPLATE: ClassVar[str] = "Invalid entity ID: {entity_id}"
        ENTITY_NOT_FOUND_TEMPLATE: ClassVar[str] = "Entity not found: {entity_id}"
        ENTITY_OPERATION_TEMPLATE: ClassVar[str] = "Entity {operation}: {entity_id}"

    # NEW: Validation system constants (EXPANDED from string mapping analysis)
    class ValidationSystem:
        """Extended validation constants beyond basic patterns."""

        # Error categories
        VALIDATION_ERROR_CATEGORY: ClassVar[str] = "VALIDATION"
        BUSINESS_ERROR_CATEGORY: ClassVar[str] = "BUSINESS"
        INFRASTRUCTURE_ERROR_CATEGORY: ClassVar[str] = "INFRASTRUCTURE"
        CONFIGURATION_ERROR_CATEGORY: ClassVar[str] = "CONFIGURATION"
        GENERAL_ERROR_CATEGORY: ClassVar[str] = "GENERAL"

        # Value object validation
        VALUE_OBJECT_INVALID: ClassVar[str] = "Value object validation failed"
        VALUE_OBJECT_EMPTY: ClassVar[str] = "Value object cannot be empty"
        VALUE_OBJECT_FORMAT_INVALID: ClassVar[str] = "Invalid value object format"

        # Type validation
        TYPE_MISMATCH: ClassVar[str] = "Type mismatch error"
        TYPE_CONVERSION_FAILED: ClassVar[str] = "Type conversion failed"
        TYPE_VALIDATION_FAILED: ClassVar[str] = "Type validation failed"

        # Business rule validation
        BUSINESS_RULE_VIOLATED: ClassVar[str] = "Business rule violated"
        DOMAIN_RULE_VIOLATION: ClassVar[str] = "Domain rule violation"
        INVARIANT_VIOLATION: ClassVar[str] = "Invariant violation"

        # Validation templates
        VALIDATION_ERROR_TEMPLATE: ClassVar[str] = "Validation failed: {details}"
        BUSINESS_RULE_TEMPLATE: ClassVar[str] = "Business rule violation: {rule}"
        TYPE_ERROR_TEMPLATE: ClassVar[str] = (
            "Type error: expected {expected}, got {actual}"
        )

    # NEW: Infrastructure messaging constants (EXPANDED from string mapping analysis)
    class InfrastructureMessages:
        """Infrastructure layer messaging constants."""

        # Serialization
        SERIALIZATION_FAILED: ClassVar[str] = "Serialization failed"
        DESERIALIZATION_FAILED: ClassVar[str] = "Deserialization failed"
        SERIALIZATION_WARNING: ClassVar[str] = "Serialization warning"

        # Delegation system
        DELEGATION_SUCCESS: ClassVar[str] = "SUCCESS"
        DELEGATION_FAILED: ClassVar[str] = "FAILED"
        DELEGATION_STATUS_TEMPLATE: ClassVar[str] = "status: {status}"

        # Configuration
        CONFIG_LOADING: ClassVar[str] = "Loading configuration"
        CONFIG_LOADED: ClassVar[str] = "Configuration loaded"
        CONFIG_FAILED: ClassVar[str] = "Configuration loading failed"
        CONFIG_FORMAT_JSON: ClassVar[str] = "json"

        # Logging and monitoring
        OPERATION_STARTED: ClassVar[str] = "Operation started"
        OPERATION_COMPLETED: ClassVar[str] = "Operation completed"
        OPERATION_FAILED: ClassVar[str] = "Operation failed"

        # Infrastructure templates
        SERIALIZATION_ERROR_TEMPLATE: ClassVar[str] = "Serialization error: {error}"
        CONFIG_ERROR_TEMPLATE: ClassVar[str] = "Configuration error: {details}"
        OPERATION_LOG_TEMPLATE: ClassVar[str] = "Operation {operation}: {status}"

    # NEW: Platform constants (FLEXT specific infrastructure)
    class Platform:
        """Platform-wide constants."""

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

    # Backward compatibility: Add commonly used constants as class attributes
    DEFAULT_TIMEOUT = Defaults.TIMEOUT


# =============================================================================
# ENUMS - Field types and other enums that modules import
# =============================================================================


class FlextFieldType(Enum):
    """Field type enumeration for validation and schema definition."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    UUID = "uuid"
    EMAIL = "email"


class FlextLogLevel(Enum):
    """Log level enumeration with numeric value support."""

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


class FlextEnvironment(Enum):
    """Environment type enumeration."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    STAGING = "staging"
    TESTING = "testing"


class FlextConnectionType(Enum):
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


class FlextDataFormat(Enum):
    """Data format enumeration."""

    JSON = "json"
    XML = "xml"
    CSV = "csv"
    LDIF = "ldif"
    YAML = "yaml"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class FlextOperationStatus(Enum):
    """Operation status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class FlextEntityStatus(Enum):
    """Entity status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DELETED = "deleted"
    SUSPENDED = "suspended"


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
# OLD USAGE (deprecated): Import from legacy.py
#   from flext_core.legacy import ERROR_CODES, DEFAULT_TIMEOUT, EMAIL_PATTERN

# =============================================================================
# EXPORTS - Semantic constants + backward compatibility
# =============================================================================

# Build legacy flat ERROR_CODES mapping with mixed strategy:
# numeric codes for key infrastructure errors, textual identifiers otherwise.
_special_numeric = {
    "GENERIC_ERROR": FlextConstants.Errors.GENERIC_ERROR,
    "VALIDATION_ERROR": "FLEXT_VALIDATION_ERROR",
    "CONNECTION_ERROR": FlextConstants.Errors.CONNECTION_ERROR,
    "TIMEOUT_ERROR": FlextConstants.Errors.TIMEOUT_ERROR,
    "OPERATION_ERROR": "FLEXT_OPERATION_ERROR",
    "TYPE_ERROR": "FLEXT_TYPE_ERROR",
    "CONFIG_ERROR": "FLEXT_CONFIG_ERROR",
    "AUTH_ERROR": "FLEXT_AUTH_ERROR",
    "PERMISSION_ERROR": "FLEXT_PERMISSION_ERROR",
}
ERROR_CODES: dict[str, str] = {}
for name in [n for n in dir(FlextConstants.Errors) if not n.startswith("_")]:
    ERROR_CODES[name] = _special_numeric.get(name, name)

# Keep a reference on the class for callers that used FlextConstants.ERROR_CODES
FlextConstants.ERROR_CODES = ERROR_CODES

# Messages namespace re-export for compatibility
MESSAGES = FlextConstants.Messages
SERVICE_NAME_EMPTY = "Service name cannot be empty"

__all__: list[str] = [
    # Legacy constants
    "ERROR_CODES",
    "MESSAGES",
    "SERVICE_NAME_EMPTY",
    # Enumerations
    "FlextConnectionType",
    # Main constants class with nested structure
    "FlextConstants",
    "FlextDataFormat",
    "FlextEntityStatus",
    "FlextEnvironment",
    "FlextFieldType",
    "FlextLogLevel",
    "FlextOperationStatus",
    # Note: Legacy constants moved to legacy.py
    # Import from flext_core.legacy if needed for backward compatibility
]
