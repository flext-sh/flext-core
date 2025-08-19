from enum import Enum
from typing import ClassVar

__all__ = [
    "ERROR_CODES",
    "MESSAGES",
    "SERVICE_NAME_EMPTY",
    "FlextConnectionType",
    "FlextConstants",
    "FlextDataFormat",
    "FlextEntityStatus",
    "FlextEnvironment",
    "FlextFieldType",
    "FlextLogLevel",
    "FlextOperationStatus",
]

class FlextConstants:
    ERROR_CODES: ClassVar[dict[str, str]]
    VERSION: ClassVar[str]
    class Core:
        NAME: str
        VERSION: str
        ECOSYSTEM_SIZE: int
        PYTHON_VERSION: str
        ARCHITECTURE: str
        CONFIGURATION_ARGUMENT_INDEX_THRESHOLD: int
        MAX_BRANCHES_ALLOWED: int
        MAX_RETURN_STATEMENTS_ALLOWED: int

    class Network:
        MIN_PORT: int
        MAX_PORT: int
        HTTP_PORT: int
        HTTPS_PORT: int
        LDAP_PORT: int
        LDAPS_PORT: int

    class Validation:
        MIN_SERVICE_NAME_LENGTH: int
        MAX_SERVICE_NAME_LENGTH: int
        MIN_SECRET_KEY_LENGTH: int
        MIN_PERCENTAGE: float
        MAX_PERCENTAGE: float

    class Errors:
        BUSINESS_ERROR_RANGE: tuple[int, int]
        TECHNICAL_ERROR_RANGE: tuple[int, int]
        VALIDATION_ERROR_RANGE: tuple[int, int]
        SECURITY_ERROR_RANGE: tuple[int, int]
        GENERIC_ERROR: str
        VALIDATION_ERROR: str
        BUSINESS_RULE_VIOLATION: str
        AUTHORIZATION_DENIED: str
        AUTHENTICATION_FAILED: str
        RESOURCE_NOT_FOUND: str
        DUPLICATE_RESOURCE: str
        CONNECTION_ERROR: str
        TIMEOUT_ERROR: str
        CONFIGURATION_ERROR: str
        SERIALIZATION_ERROR: str
        EXTERNAL_SERVICE_ERROR: str
        BUSINESS_RULE_ERROR: str
        AUTHENTICATION_ERROR: str
        TYPE_ERROR: str
        UNWRAP_ERROR: str
        CLI_ERROR: str
        NULL_DATA_ERROR: str
        EXPECTATION_ERROR: str
        INVALID_ARGUMENT: str
        OPERATION_ERROR: str
        FILTER_ERROR: str
        EXCEPTION_ERROR: str
        MULTIPLE_ERRORS: str
        CONTEXT_ERROR: str
        CONDITIONAL_ERROR: str
        SIDE_EFFECT_ERROR: str
        CHAIN_ERROR: str
        MAP_ERROR: str
        BIND_ERROR: str
        RETRY_ERROR: str
        CIRCUIT_BREAKER_ERROR: str
        CONCURRENCY_ERROR: str
        RESOURCE_ERROR: str
        DATABASE_ERROR: str
        API_ERROR: str
        EVENT_ERROR: str
        SECURITY_ERROR: str
        AUTH_ERROR: str
        CRITICAL_ERROR: str
        EXTERNAL_ERROR: str
        MIGRATION_ERROR: str
        PERMISSION_ERROR: str
        PROCESSING_ERROR: str
        CHAINED_ERROR: str
        RETRYABLE_ERROR: str
        UNKNOWN_ERROR: str
        FALLBACK_FAILURE: str
        OPERATION_AND_FALLBACK_FAILURE: str
        OPERATION_FAILURE: str
        NOT_FOUND: str
        ALREADY_EXISTS: str
        CONFIG_ERROR: str
        MESSAGES: ClassVar[dict[str, str]]

    class Messages:
        SUCCESS: str
        STARTED: str
        COMPLETED: str
        FAILED: str
        VALIDATING: str
        CONFIGURING: str
        UNKNOWN_ERROR: str
        FILTER_FAILED: str
        VALIDATION_FAILED: str
        OPERATION_FAILED: str
        UNWRAP_FAILED: str
        NULL_DATA: str
        INVALID_INPUT: str
        TYPE_MISMATCH: str
        ENTITY_ID_EMPTY: str
        SERVICE_NAME_EMPTY: str
        NAME_EMPTY: str
        MESSAGE_EMPTY: str
        EVENT_TYPE_EMPTY: str
        VALUE_EMPTY: str
        BUSINESS_RULE_VIOLATED: str
        RETRY_EXHAUSTED: str
        CIRCUIT_BREAKER_OPEN: str
        CONCURRENT_MODIFICATION: str
        RESOURCE_UNAVAILABLE: str
        SERIALIZATION_FAILED: str
        DATABASE_CONNECTION_FAILED: str
        API_CALL_FAILED: str
        EVENT_PROCESSING_FAILED: str
        OPERATION_TIMEOUT: str
        SECURITY_VIOLATION: str
        CONFIGURATION_INVALID: str

    class Status:
        ACTIVE: str
        INACTIVE: str
        PENDING: str
        COMPLETED: str
        FAILED: str
        RUNNING: str
        CANCELLED: str
        SUCCESS: str
        FAILURE: str
        PROCESSING: str

    class Patterns:
        UUID_PATTERN: str
        SLUG_PATTERN: str
        IDENTIFIER_PATTERN: str
        SERVICE_NAME_PATTERN: str
        EMAIL_PATTERN: str
        USERNAME_PATTERN: str
        CREDENTIAL_STRENGTH_PATTERN: str
        IPV4_PATTERN: str
        HOSTNAME_PATTERN: str
        URL_PATTERN: str
        SEMVER_PATTERN: str
        SEMANTIC_VERSION_PATTERN: str

    class Defaults:
        TIMEOUT: int
        MAX_RETRIES: int
        CONNECTION_TIMEOUT: int
        PAGE_SIZE: int
        MAX_PAGE_SIZE: int
        CONFIG_TIMEOUT: int
        CONFIG_RETRIES: int
        CLI_HELP_WIDTH: int
        CLI_TIMEOUT: int
        MODEL_TIMEOUT: int
        VALIDATION_TIMEOUT: int
        DB_TIMEOUT: int
        DB_POOL_SIZE: int
        CACHE_TTL: int
        CACHE_MAX_SIZE: int

    class Limits:
        MAX_STRING_LENGTH: int
        MAX_LIST_SIZE: int
        MIN_PASSWORD_LENGTH: int
        MAX_PASSWORD_LENGTH: int
        MAX_FILE_SIZE: int
        MAX_BATCH_SIZE: int
        MIN_PORT: int
        MAX_PORT: int
        MAX_THREADS: int

    class Performance:
        DEFAULT_BATCH_SIZE: int
        MAX_BATCH_SIZE: int
        CACHE_TTL: int
        CACHE_MAX_SIZE: int
        POOL_SIZE: int
        MAX_CONNECTIONS: int
        KEEP_ALIVE_TIMEOUT: int
        SLOW_QUERY_THRESHOLD: float
        SLOW_REQUEST_THRESHOLD: float
        HIGH_MEMORY_THRESHOLD: float
        HIGH_CPU_THRESHOLD: float
        METRICS_INTERVAL: int
        HEALTH_CHECK_INTERVAL: int
        SECONDS_PER_MINUTE: int
        SECONDS_PER_HOUR: int
        BYTES_PER_KB: int

    class Configuration:
        CLI_PRIORITY: int
        ENV_PRIORITY: int
        DOTENV_PRIORITY: int
        CONFIG_FILE_PRIORITY: int
        CONSTANTS_PRIORITY: int
        DOTENV_FILES: ClassVar[list[str]]
        CONFIG_FILES: ClassVar[list[str]]
        ENVIRONMENTS: ClassVar[list[str]]
        DEFAULT_ENVIRONMENT: str
        REQUIRED_FIELDS: ClassVar[list[str]]
        OPTIONAL_FIELDS: ClassVar[list[str]]

    class Cli:
        HELP_ARGS: ClassVar[list[str]]
        VERSION_ARGS: ClassVar[list[str]]
        CONFIG_ARGS: ClassVar[list[str]]
        VERBOSE_ARGS: ClassVar[list[str]]
        LONG_PREFIX: str
        SHORT_PREFIX: str
        OUTPUT_FORMATS: ClassVar[list[str]]
        DEFAULT_OUTPUT_FORMAT: str
        SUCCESS_EXIT_CODE: int
        ERROR_EXIT_CODE: int
        INVALID_USAGE_EXIT_CODE: int

    class Infrastructure:
        DEFAULT_DB_PORT: int
        DEFAULT_ORACLE_PORT: int
        DEFAULT_REDIS_PORT: int
        DEFAULT_MYSQL_PORT: int
        DEFAULT_MONGODB_PORT: int
        DEFAULT_POOL_SIZE: int
        MAX_POOL_SIZE: int
        MIN_POOL_SIZE: int
        DEFAULT_HOST: str
        LOCALHOST_ALIASES: ClassVar[list[str]]
        DEFAULT_PROTOCOL: str
        SECURE_PROTOCOLS: ClassVar[list[str]]

    class Models:
        VALIDATE_ASSIGNMENT: bool
        USE_ENUM_VALUES: bool
        STR_STRIP_WHITESPACE: bool
        ARBITRARY_TYPES_ALLOWED: bool
        VALIDATE_DEFAULT: bool
        EXTRA_FORBID: str
        EXTRA_ALLOW: str
        EXTRA_IGNORE: str
        FIELD_TYPES: ClassVar[list[str]]
        MODEL_STATES: ClassVar[list[str]]

    class Observability:
        LOG_LEVELS: ClassVar[list[str]]
        DEFAULT_LOG_LEVEL: str
        TRACE_LEVEL: int
        SPAN_TYPES: ClassVar[list[str]]
        METRIC_TYPES: ClassVar[list[str]]
        ALERT_LEVELS: ClassVar[list[str]]
        CORRELATION_ID_HEADER: str
        TRACE_ID_HEADER: str
        FLEXT_SERIALIZATION_VERSION: str
        SERIALIZATION_FORMAT_JSON: str
        SERIALIZATION_FORMAT_JSON_COMPRESSED: str

    class Handlers:
        HANDLER_NOT_FOUND: ClassVar[str]
        HANDLER_ALREADY_REGISTERED: ClassVar[str]
        HANDLER_NOT_CALLABLE: ClassVar[str]
        NO_HANDLER_REGISTERED: ClassVar[str]
        REGISTRY_NOT_FOUND: ClassVar[str]
        MISSING_PERMISSION: ClassVar[str]
        PERMISSION_DENIED: ClassVar[str]
        AUTH_REQUIRED: ClassVar[str]
        EVENT_PROCESSING_FAILED: ClassVar[str]
        EVENT_HANDLER_FAILED: ClassVar[str]
        CHAIN_HANDLER_FAILED: ClassVar[str]
        METRICS_COLLECTION_FAILED: ClassVar[str]
        REQUEST_CANNOT_BE_NONE: ClassVar[str]
        MESSAGE_CANNOT_BE_NONE: ClassVar[str]
        VALIDATION_FAILED: ClassVar[str]
        EVENT_MISSING_TYPE: ClassVar[str]
        NO_USER_IN_CONTEXT: ClassVar[str]
        INVALID_PERMISSIONS_FORMAT: ClassVar[str]
        HANDLER_NAME_EMPTY: ClassVar[str]
        INVALID_HANDLER_PROVIDED: ClassVar[str]
        HANDLER_NAME_MUST_BE_STRING: ClassVar[str]
        NO_HANDLER_COULD_PROCESS: ClassVar[str]
        CHAIN_PROCESSING_FAILED: ClassVar[str]
        NOT_IMPLEMENTED: ClassVar[str]
        AUTHORIZATION_FAILED: ClassVar[str]
        QUERY_HANDLER_NOT_IMPLEMENTED: ClassVar[str]
        HANDLER_NOT_FOUND_TEMPLATE: ClassVar[str]
        MISSING_PERMISSION_TEMPLATE: ClassVar[str]
        EVENT_PROCESSING_FAILED_TEMPLATE: ClassVar[str]
        HANDLER_FAILED_TEMPLATE: ClassVar[str]
        NO_HANDLER_FOR_TYPE_TEMPLATE: ClassVar[str]
        CHAIN_HANDLER_FAILED_TEMPLATE: ClassVar[str]
        METRICS_FAILED_TEMPLATE: ClassVar[str]
        NO_HANDLER_FOR_COMMAND_TEMPLATE: ClassVar[str]
        NO_HANDLER_FOR_QUERY_TEMPLATE: ClassVar[str]

    class Entities:
        ENTITY_ID_INVALID: ClassVar[str]
        ENTITY_ID_EMPTY: ClassVar[str]
        ENTITY_NAME_EMPTY: ClassVar[str]
        ENTITY_VALIDATION_FAILED: ClassVar[str]
        CACHE_KEY_TEMPLATE: ClassVar[str]
        CACHE_KEY_HASH_TEMPLATE: ClassVar[str]
        OPERATION_LOG_TEMPLATE: ClassVar[str]
        ENTITY_CREATED: ClassVar[str]
        ENTITY_UPDATED: ClassVar[str]
        ENTITY_DELETED: ClassVar[str]
        ENTITY_ACTIVE: ClassVar[str]
        ENTITY_INACTIVE: ClassVar[str]
        ENTITY_STATE_INVALID: ClassVar[str]
        INVALID_ENTITY_ID_TEMPLATE: ClassVar[str]
        ENTITY_NOT_FOUND_TEMPLATE: ClassVar[str]
        ENTITY_OPERATION_TEMPLATE: ClassVar[str]

    class ValidationSystem:
        VALIDATION_ERROR_CATEGORY: ClassVar[str]
        BUSINESS_ERROR_CATEGORY: ClassVar[str]
        INFRASTRUCTURE_ERROR_CATEGORY: ClassVar[str]
        CONFIGURATION_ERROR_CATEGORY: ClassVar[str]
        GENERAL_ERROR_CATEGORY: ClassVar[str]
        VALUE_OBJECT_INVALID: ClassVar[str]
        VALUE_OBJECT_EMPTY: ClassVar[str]
        VALUE_OBJECT_FORMAT_INVALID: ClassVar[str]
        TYPE_MISMATCH: ClassVar[str]
        TYPE_CONVERSION_FAILED: ClassVar[str]
        TYPE_VALIDATION_FAILED: ClassVar[str]
        BUSINESS_RULE_VIOLATED: ClassVar[str]
        DOMAIN_RULE_VIOLATION: ClassVar[str]
        INVARIANT_VIOLATION: ClassVar[str]
        VALIDATION_ERROR_TEMPLATE: ClassVar[str]
        BUSINESS_RULE_TEMPLATE: ClassVar[str]
        TYPE_ERROR_TEMPLATE: ClassVar[str]

    class InfrastructureMessages:
        SERIALIZATION_FAILED: ClassVar[str]
        DESERIALIZATION_FAILED: ClassVar[str]
        SERIALIZATION_WARNING: ClassVar[str]
        DELEGATION_SUCCESS: ClassVar[str]
        DELEGATION_FAILED: ClassVar[str]
        DELEGATION_STATUS_TEMPLATE: ClassVar[str]
        CONFIG_LOADING: ClassVar[str]
        CONFIG_LOADED: ClassVar[str]
        CONFIG_FAILED: ClassVar[str]
        CONFIG_FORMAT_JSON: ClassVar[str]
        OPERATION_STARTED: ClassVar[str]
        OPERATION_COMPLETED: ClassVar[str]
        OPERATION_FAILED: ClassVar[str]
        SERIALIZATION_ERROR_TEMPLATE: ClassVar[str]
        CONFIG_ERROR_TEMPLATE: ClassVar[str]
        OPERATION_LOG_TEMPLATE: ClassVar[str]

    class Platform:
        FLEXCORE_PORT: int
        FLEXT_SERVICE_PORT: int
        FLEXT_API_PORT: int
        FLEXT_WEB_PORT: int
        FLEXT_GRPC_PORT: int
        POSTGRESQL_PORT: int
        REDIS_PORT: int
        MONITORING_PORT: int
        METRICS_PORT: int
        DEV_DB_PORT: int
        DEV_REDIS_PORT: int
        DEV_WEBHOOK_PORT: int
        DEFAULT_HOST: str
        PRODUCTION_HOST: str
        LOOPBACK_HOST: str
        DEFAULT_BASE_URL: str
        PRODUCTION_BASE_URL: str
        DB_MIN_CONNECTIONS: int
        DB_MAX_CONNECTIONS: int
        DB_CONNECTION_TIMEOUT: int
        DB_QUERY_TIMEOUT: int
        DEFAULT_POSTGRES_URL: str
        DEFAULT_SQLITE_URL: str
        REDIS_URL: str
        REDIS_TIMEOUT: int
        CACHE_TTL_SHORT: int
        CACHE_TTL_MEDIUM: int
        CACHE_TTL_LONG: int
        CACHE_TTL_EXTENDED: int
        ACCESS_TOKEN_LIFETIME: int
        REFRESH_TOKEN_LIFETIME: int
        RATE_LIMIT_REQUESTS: int
        RATE_LIMIT_WINDOW: int
        MAX_LOGIN_ATTEMPTS: int
        LOCKOUT_DURATION: int
        HTTP_CONNECT_TIMEOUT: int
        HTTP_READ_TIMEOUT: int
        HTTP_TOTAL_TIMEOUT: int
        SERVICE_STARTUP_TIMEOUT: int
        SERVICE_SHUTDOWN_TIMEOUT: int
        MAX_NAME_LENGTH: int
        MAX_DESCRIPTION_LENGTH: int
        MAX_FILE_SIZE: int
        MIN_PORT_NUMBER: int
        MAX_PORT_NUMBER: int

    DEFAULT_TIMEOUT: int

class FlextFieldType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    UUID = "uuid"
    EMAIL = "email"

class FlextLogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRACE = "TRACE"
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    @classmethod
    def get_numeric_levels(cls) -> dict[str, int]: ...
    def get_numeric_value(self) -> int: ...

class FlextEnvironment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    STAGING = "staging"
    TESTING = "testing"

class FlextConnectionType(Enum):
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
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    LDIF = "ldif"
    YAML = "yaml"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"

class FlextOperationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class FlextEntityStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DELETED = "deleted"
    SUSPENDED = "suspended"

ERROR_CODES: dict[str, str]
MESSAGES = FlextConstants.Messages
SERVICE_NAME_EMPTY: str
