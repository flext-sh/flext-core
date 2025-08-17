from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TypeVar

from _typeshed import Incomplete

__all__ = [
    "Cacheable",
    "Comparable",
    "Configurable",
    "E",
    "EntryT",
    "F",
    "FlextSerializable",
    "FlextTypes",
    "FlextValidatable",
    "FlextValidator",
    "P",
    "R",
    "Serializable",
    "T",
    "TAggregateId",
    "TAnyDict",
    "TAnyList",
    "TAnyObject",
    "TCallable",
    "TCommand",
    "TCommandBusId",
    "TCommandId",
    "TCommandMetadata",
    "TCommandPayload",
    "TCommandPriority",
    "TCommandResult",
    "TCommandType",
    "TComparable",
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
    "TEntity",
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
    "TQuery",
    "TQueryCriteria",
    "TQueryId",
    "TQueryProjection",
    "TQueryResult",
    "TQueryType",
    "TRequest",
    "TRequestId",
    "TResponse",
    "TResult",
    "TSerializable",
    "TService",
    "TServiceKey",
    "TServiceName",
    "TSessionId",
    "TStringDict",
    "TToken",
    "TTransactionId",
    "TTransformer",
    "TUserData",
    "TUserId",
    "TValidatable",
    "TValidationConfig",
    "TValidationConstraint",
    "TValidationContext",
    "TValidationError",
    "TValidationPipeline",
    "TValidationResult",
    "TValidationRule",
    "TValidationSchema",
    "TValidator",
    "TValidatorName",
    "TValue",
    "Timestamped",
    "U",
    "V",
    "Validatable",
    "get_centralized_types_usage_info",
]

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
R = TypeVar("R")
E = TypeVar("E")
F = TypeVar("F")
P = TypeVar("P")
TComparable = TypeVar("TComparable")
TSerializable = TypeVar("TSerializable")
TValidatable = TypeVar("TValidatable")
TEntity = TypeVar("TEntity")
TAnyObject = TypeVar("TAnyObject")
TCommand = TypeVar("TCommand")
TQuery = TypeVar("TQuery")
TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")
TResult = TypeVar("TResult")
TService = TypeVar("TService")
TOptional = TypeVar("TOptional")
EntryT = TypeVar("EntryT")

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
        ServiceInstance = TypeVar("ServiceInstance")
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

TEntityId: Incomplete
TValue: Incomplete
TData: Incomplete
TConfig: Incomplete
TEvent: Incomplete
TMessage: Incomplete
TCommandId: Incomplete
TCommandType: Incomplete
THandlerName: Incomplete
TCommandPayload: Incomplete
TCommandResult: Incomplete
TCommandMetadata: Incomplete
TMiddlewareName: Incomplete
TValidationRule: Incomplete
TCommandBusId: Incomplete
TCommandPriority: Incomplete
TQueryId: Incomplete
TQueryType: Incomplete
TQueryResult: Incomplete
TQueryCriteria: Incomplete
TQueryProjection: Incomplete
TPaginationToken: Incomplete
TServiceName: Incomplete
TServiceKey: Incomplete
TFactory: Incomplete
TTransformer: Incomplete
TPredicate: Incomplete
TValidator: Incomplete
TCallable: Incomplete
TErrorHandler: Incomplete
TConnectionString: Incomplete
TLogMessage: Incomplete
TErrorCode: Incomplete
TErrorMessage: Incomplete
TAnyDict: Incomplete
TAnyList: Incomplete
TDict: Incomplete
TList: Incomplete
TStringDict: Incomplete
TUserData: Incomplete
TToken: Incomplete
TCredentials: Incomplete
TConnection: Incomplete
TUserId: Incomplete
TContextDict: Incomplete
TCorrelationId: Incomplete
TRequestId: Incomplete
TConfigDict: Incomplete
TFieldInfo: Incomplete
TFieldMetadata: Incomplete
TConfigKey: Incomplete
TConfigValue: Incomplete
TConfigPath: Incomplete
TConfigEnv: Incomplete
TConfigValidationRule: Incomplete
TConfigMergeStrategy: Incomplete
TConfigSettings: Incomplete
TConfigDefaults: Incomplete
TConfigOverrides: Incomplete
TEnvironmentName: Incomplete
TDeploymentStage: Incomplete
TConfigVersion: Incomplete
TValidationError: Incomplete
TValidationResult: Incomplete
TValidationContext: Incomplete
TValidatorName: Incomplete
TValidationConfig: Incomplete
TValidationConstraint: Incomplete
TValidationSchema: Incomplete
TFieldName: Incomplete
TFieldValue: Incomplete
TFieldRule: Incomplete
TFieldError: Incomplete
TCustomValidator: Incomplete
TValidationPipeline: Incomplete
TLoggerName: Incomplete
TBusinessId = str
TBusinessName = str
TBusinessCode = str
TBusinessStatus = str
TBusinessType = str
TCacheKey = str
type TCacheValue = str | int | float | bool | None
TCacheTTL = int
TLogLevel: Incomplete
TLogFormat: Incomplete
TLogHandler: Incomplete
TLogFilter: Incomplete
TSessionId: Incomplete
TTransactionId: Incomplete
TOperationName: Incomplete
TLogRecord: Incomplete
TLogMetrics: Incomplete
TLogConfiguration: Incomplete
TEntityVersion: Incomplete
TEntityTimestamp: Incomplete
TDomainEventType: Incomplete
TDomainEventData: Incomplete
TAggregateId: Incomplete
TEntityRule: Incomplete
TEntityState: Incomplete
TEntityMetadata: Incomplete
TEntityDefaults: Incomplete
TEntityChanges: Incomplete
TFactoryResult: Incomplete
TDomainEvents: Incomplete
TEventStream: Incomplete
TEventVersion: Incomplete
Comparable: Incomplete
Serializable: Incomplete
Timestamped: Incomplete
Validatable: Incomplete
Cacheable: Incomplete
Configurable: Incomplete
FlextSerializable = Serializable
FlextValidatable = Validatable
FlextValidator: Incomplete

def get_centralized_types_usage_info() -> str: ...
