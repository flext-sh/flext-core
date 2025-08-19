from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TypeVar

__all__ = [  # noqa: RUF022
    "Cacheable",
    "Comparable",
    "Configurable",
    # Core type variables (only for truly generic types)
    "E",
    "F",
    "FlextFieldId",
    "FlextFieldName",
    "FlextFieldTypeStr",
    "FlextSerializable",
    "FlextTypes",
    "FlextValidatable",
    "FlextValidator",
    "P",
    "R",
    "Serializable",
    "T",
    "U",
    "V",
    # Type aliases for domain types (not TypeVars)
    "TAggregateId",
    "TAnyDict",
    "TAnyList",
    "TCallable",
    "TCommandBusId",
    "TCommandId",
    "TCommandMetadata",
    "TCommandPayload",
    "TCommandPriority",
    "TCommandResult",
    "TCommandType",
    "TConfig",
    "TConfigDefaults",
    "TConfigDict",
    "TConfigEnv",
    "TConfigKey",
    "TConfigMergeStrategy",
    "TConfigOverrides",
    "TConfigPath",
    "TConfigSettings",
    "TConfigValidationRule",
    "TConfigValue",
    "TConfigVersion",
    "TConnection",
    "TConnectionString",
    "TContextDict",
    "TCorrelationId",
    "TCredentials",
    "TCustomValidator",
    "TData",
    "TDeploymentStage",
    "TDict",
    "TDomainEventData",
    "TDomainEventType",
    "TDomainEvents",
    "TEntityChanges",
    "TEntityDefaults",
    "TEntityId",
    "TEntityMetadata",
    "TEntityRule",
    "TEntityState",
    "TEntityTimestamp",
    "TEntityVersion",
    "TEnvironmentName",
    "TErrorCode",
    "TErrorHandler",
    "TErrorMessage",
    "TEvent",
    "TEventStream",
    "TEventVersion",
    "TFactory",
    "TFactoryResult",
    "TFieldError",
    "TFieldInfo",
    "TFieldMetadata",
    "TFieldName",
    "TFieldRule",
    "TFieldValue",
    "THandlerName",
    "TList",
    "TLogConfiguration",
    "TLogFilter",
    "TLogFormat",
    "TLogHandler",
    "TLogLevel",
    "TLogMessage",
    "TLogMetrics",
    "TLogRecord",
    "TLoggerName",
    "TMessage",
    "TMiddlewareName",
    "TOperationName",
    "TOptional",
    "TPaginationToken",
    "TPredicate",
    "TQueryCriteria",
    "TQueryId",
    "TQueryProjection",
]

# Core type variables (only for truly generic types that need to be variable)
T = TypeVar("T")  # noqa: PYI001
U = TypeVar("U")  # noqa: PYI001
V = TypeVar("V")  # noqa: PYI001
R = TypeVar("R")  # noqa: PYI001
E = TypeVar("E")  # noqa: PYI001
F = TypeVar("F")  # noqa: PYI001
P = TypeVar("P")  # noqa: PYI001

# Type aliases for domain types (not TypeVars - these are concrete types)
type TComparable = object
type TSerializable = object
type TValidatable = object
type TEntity = object
type TAnyObject = object
type TCommand = object
type TQuery = object
type TRequest = object
type TResponse = object
type TResult = object
type TService = object
type TOptional = object
type EntryT = object

# Backward compatibility - export TypeVars directly without aliases

class FlextTypes:
    class Core:
        type Value = str | int | float | bool | None
        type Data = dict[str, object]
        type Config = dict[str, str | int | float | bool | None]
        type EntityId = str
        type Id = str
        type Key = str
        type AnyDict = dict[str, object]
        type AnyList = list[object]
        type StringDict = dict[str, str]
        type JsonDict = dict[str, object]
        type Dict = FlextTypes.Core.AnyDict
        type List = FlextTypes.Core.AnyList
        type ConnectionString = str
        type LogMessage = str
        type ErrorCode = str
        type ErrorMessage = str
        type AnyCallable = Callable[[object], object]
        type Factory[T] = Callable[[], T] | Callable[[object], T]
        type Transformer[T, U] = Callable[[T], U]
        type Predicate[T] = Callable[[T], bool]
        type Validator[T] = Callable[[T], bool]
        type ErrorHandler = Callable[[Exception], str]

    class Data:
        type Dict = dict[str, object]
        type StringDict = dict[str, str]
        type JsonDict = dict[str, object]
        type List = list[object]
        type StringList = list[str]

    class Service:
        type ServiceName = str
        type FlextServiceKey = str | type[object]
        type Container = Mapping[str, object]
        type ServiceLocator = Callable[[str], object]
        type ServiceFactory[T] = Callable[[], T]
        type EventHandler[TEvent] = Callable[[TEvent], None]
        type EventBus = Callable[[object], None]
        type FlextMessageHandler[T] = Callable[[T], object]
        type JsonDict = dict[str, object]
        type Metadata = dict[str, object]
        type Settings = dict[str, object]
        type Configuration = Mapping[str, object]
        type Transform[T, U] = Callable[[T], U]
        type Handler[T, R] = Callable[[T], R]
        _ServiceInstance = TypeVar("_ServiceInstance")
        ServiceInstance = _ServiceInstance  # Backward compatibility
        type CorrelationId = str
        type RequestId = str
        type TraceId = str

    class Domain:
        type EntityVersion = int
        type EntityTimestamp = datetime
        type DomainEventType = str
        type DomainEventData = dict[str, object]
        type AggregateId = str
        type EntityRule = str
        type EntityState = str
        type EntityMetadata = dict[str, object]
        type EntityDefaults = dict[str, object]
        type EntityChanges = dict[str, object]
        type FactoryResult[T] = object
        type DomainEvents = list[object]
        type EventStream = list[object]
        type EventVersion = int
        type ValueData = dict[str, object]
        type ValueValidation = Callable[[object], bool]

    class CQRS:
        type CommandId = str
        type CommandType = str
        type HandlerName = str
        type CommandPayload = dict[str, object]
        type CommandResult = object
        type CommandMetadata = dict[str, object]
        type MiddlewareName = str
        type ValidationRule = str
        type CommandBusId = str
        type CommandPriority = int
        type QueryId = str
        type QueryType = str
        type QueryResult[T] = object
        type QueryCriteria = dict[str, object]
        type QueryProjection = list[object]
        type PaginationToken = str
        type Event = dict[str, object]
        type Message = dict[str, object]

    class Validation:
        type ValidationRule = str
        type ValidationError = str
        type ValidationResult = object
        type ValidationContext = dict[str, object]
        type ValidatorName = str
        type ValidationConfig = dict[str, object]
        type ValidationConstraint = object
        type ValidationSchema = dict[str, object]
        type FieldName = str
        type FieldValue = object
        type FieldRule = str
        type FieldError = str
        type FieldInfo = dict[str, object]
        type FieldMetadata = dict[str, object]
        type FieldId = str
        type FieldTypeStr = str
        type CustomValidator = Callable[[object], object]
        type ValidationPipeline = list[FlextTypes.Validation.CustomValidator]

    class Config:
        type ConfigKey = str
        type ConfigValue = object
        type ConfigPath = str
        type ConfigEnv = str
        type ConfigValidationRule = str
        type ConfigMergeStrategy = str
        type ConfigSettings = dict[str, object]
        type ConfigDefaults = dict[str, object]
        type ConfigOverrides = dict[str, object]
        type ConfigDict = dict[str, str | int | float | bool | None]
        type EnvironmentName = str
        type DeploymentStage = str
        type ConfigVersion = str
        type DirectoryPath = str
        type FilePath = str
        type EnvVar = str
        type ConfigSection = str

    class Logging:
        type LoggerName = str
        type LogLevel = str
        type LogFormat = str
        type LogHandler = str
        type LogFilter = str
        type CorrelationId = str
        type SessionId = str
        type TransactionId = str
        type OperationName = str
        type LogRecord = dict[str, object]
        type LogMetrics = dict[str, object]
        type LogConfiguration = dict[str, object]

    class Auth:
        type Token = str
        type UserData = dict[str, object]
        type Credentials = dict[str, object]
        type UserId = str
        type Role = str
        type Permission = str
        type Provider = object
        type Connection = str
        type ContextDict = dict[str, object]

    class Singer:
        type StreamName = str
        type SchemaName = str
        type TableName = str
        type Record = dict[str, object]
        type RecordId = str
        type TapConfig = dict[str, object]
        type TargetConfig = dict[str, object]
        type State = dict[str, object]
        type Bookmark = dict[str, object]
        type Catalog = dict[str, object]
        type Stream = dict[str, object]
        type Schema = dict[str, object]

    class Protocols:
        type Comparable = object
        type Serializable = object
        type Validatable = object
        type Timestamped = object
        type Cacheable = object
        type Configurable = object
        type Validator[T] = Callable[[T], bool | str]

    class TypeGuards:
        @staticmethod
        def has_attribute(obj: object, attr: str) -> bool: ...
        @staticmethod
        def is_instance_of(obj: object, target_type: type) -> bool: ...
        @staticmethod
        def is_dict_like(obj: object) -> bool: ...
        @staticmethod
        def is_list_like(obj: object) -> bool: ...
        @staticmethod
        def is_callable(obj: object) -> bool: ...

# =============================================================================
# TYPE VARIABLES - All TypeVars properly defined
# =============================================================================

# Core type variables (already defined above)
# T, U, V, R, E, F, P, TComparable, TSerializable, TValidatable, TEntity, TAnyObject, TCommand, TQuery, TRequest, TResponse, TResult, TService, TOptional, EntryT

# Domain and data type aliases
type TEntityId = str
type TValue = str | int | float | bool | None
type TData = dict[str, object]
type TConfig = dict[str, str | int | float | bool | None]

# CQRS type aliases
type TEvent = dict[str, object]
type TMessage = dict[str, object]
type TCommandId = str
type TCommandType = str
type THandlerName = str
type TCommandPayload = dict[str, object]
type TCommandResult = object
type TCommandMetadata = dict[str, object]
type TMiddlewareName = str
type TValidationRule = str
type TCommandBusId = str
type TCommandPriority = int
type TQueryId = str
type TQueryType = str
type TQueryResult = object
type TQueryCriteria = dict[str, object]
type TQueryProjection = list[object]
type TPaginationToken = str

# Service type aliases
type TServiceName = str
type TServiceKey = str | type[object]

# Callable type aliases
type TFactory = Callable[[], object] | Callable[[object], object]
type TTransformer = Callable[[object], object]
type TPredicate = Callable[[object], bool]
type TValidator = Callable[[object], bool]
type TCallable = Callable[[object], object]
type TErrorHandler = Callable[[Exception], str]

# Infrastructure type aliases
type TConnectionString = str
type TLogMessage = str
type TErrorCode = str
type TErrorMessage = str

# Collection type aliases
type TAnyDict = dict[str, object]
type TAnyList = list[object]
type TDict = dict[str, object]
type TList = list[object]
type TStringDict = dict[str, str]

# Authentication and token types
type TUserData = dict[str, object]
type TToken = str
type TCredentials = dict[str, object]
type TConnection = str
type TUserId = str

# Context and correlation types
type TContextDict = dict[str, object]
type TCorrelationId = str
type TRequestId = str
type TConfigDict = dict[str, str | int | float | bool | None]

# Field and metadata types
type TFieldInfo = dict[str, object]
type TFieldMetadata = dict[str, object]

# Config types
type TConfigKey = str
type TConfigValue = object
type TConfigPath = str
type TConfigEnv = str
type TConfigValidationRule = str
type TConfigMergeStrategy = str
type TConfigSettings = dict[str, object]
type TConfigDefaults = dict[str, object]
type TConfigOverrides = dict[str, object]
type TEnvironmentName = str
type TDeploymentStage = str
type TConfigVersion = str

# Validation types
type TValidationError = str
type TValidationResult = object
type TValidationContext = dict[str, object]
type TValidatorName = str
type TValidationConfig = dict[str, object]
type TValidationConstraint = object
type TValidationSchema = dict[str, object]
type TFieldName = str
type TFieldValue = object
type TFieldRule = str
type TFieldError = str
type TCustomValidator = Callable[[object], object]
type TValidationPipeline = list[Callable[[object], object]]

# Type guard types
type TTypeGuard = Callable[[object], bool]
type TGuardFunction = Callable[[object], bool]
type TGuardResult = bool
type TGuardContext = dict[str, object]

# Logging types
type TLoggerName = str

# Business types (testing convenience)
type TBusinessId = str
type TBusinessName = str
type TBusinessCode = str
type TBusinessStatus = str
type TBusinessType = str

# Cache types (testing convenience)
type TCacheKey = str
type TCacheValue = str | int | float | bool | None
type TCacheTTL = int

# Filesystem aliases
type TDirectoryPath = str
type TFilePath = str
type TLogLevel = str
type TLogFormat = str
type TLogHandler = str
type TLogFilter = str
type TSessionId = str
type TTransactionId = str
type TOperationName = str
type TLogRecord = dict[str, object]
type TLogMetrics = dict[str, object]
type TLogConfiguration = dict[str, object]

# Entity types
type TEntityVersion = int
type TEntityTimestamp = datetime
type TDomainEventType = str
type TDomainEventData = dict[str, object]
type TAggregateId = str
type TEntityRule = str
type TEntityState = str
type TEntityMetadata = dict[str, object]
type TEntityDefaults = dict[str, object]
type TEntityChanges = dict[str, object]
type TFactoryResult = object
type TDomainEvents = list[object]
type TEventStream = list[object]
type TEventVersion = int

# Protocol convenience aliases
type Comparable = object
type Serializable = object
type Timestamped = object
type Validatable = object
type Cacheable = object
type Configurable = object

# Convenience protocol aliases with transition support
type FlextSerializable = object
type FlextValidatable = object
type FlextValidator = Callable[[object], bool | str]

# Field type definitions
type FlextFieldId = str
type FlextFieldName = str
type FlextFieldTypeStr = str

# Test type definitions
type TTestData = str
type TTestConfig = str

def get_centralized_types_usage_info() -> str: ...
