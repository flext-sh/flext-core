"""FLEXT Core Semantic Types - Hierarchical Type System Foundation.

This module implements the FLEXT Unified Semantic Patterns as specified in
/home/marlonsc/flext/flext-core/docs/FLEXT_UNIFIED_SEMANTIC_PATTERNS.md

UNIFIED SEMANTIC PATTERNS - This module is part of the harmonized FLEXT
semantic pattern system that eliminates duplication across the ecosystem.

Unified Pattern: Flext[Domain][Type][Context]
Example Usage: FlextTypes.Data.Connection, FlextTypes.Auth.Token, FlextTypes.Core.Result

Harmonized Architecture (4 Layers):
    Layer 0: Foundation (flext-core) - Core types, models, protocols
    Layer 1: Domain Protocols (flext-*/protocols.py) - Structural contracts
    Layer 2: Domain Extensions (flext-* subprojects) - Specialized implementations
    Layer 3: Composite Applications (services/apps) - Multi-domain compositions

Type Organization Principles:
    - Hierarchical Namespace: Organized by domain for semantic clarity
    - Composition over Inheritance: Protocol-based contracts over deep inheritance
    - Minimal Core Foundation: Maximum 50 type aliases in core foundation
    - Project Extension Points: Clear extension patterns for subprojects
    - Backward Compatibility: Legacy aliases during migration period

Quality Standards:
    - Python 3.13+ modern type syntax with `type` statements
    - MyPy strict mode compatibility with zero errors
    - Protocol-based structural typing for flexibility
    - Generic type parameterization for type safety
    - Cross-language serialization support (Go bridge)

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Literal, Protocol, TypeVar

# =============================================================================
# CORE TYPE VARIABLES - Foundation for generic types
# =============================================================================

# Core generic type variables
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
E = TypeVar("E", bound=Exception)

# Domain-specific type variables
TEntity = TypeVar("TEntity")
TValue = TypeVar("TValue")
TConfig = TypeVar("TConfig")
TResult = TypeVar("TResult")
TEvent = TypeVar("TEvent")
TCommand = TypeVar("TCommand")
TQuery = TypeVar("TQuery")

# Core type alias for JSON-compatible dictionary
JsonDict = dict[str, object]

# =============================================================================
# FLEXT SEMANTIC TYPE SYSTEM - Hierarchical namespace organization
# =============================================================================


class FlextTypes:
    """Semantic type system with hierarchical domain organization.

    Provides a structured approach to type organization across the FLEXT ecosystem
    following the semantic pattern: Flext[Domain][Type][Context]

    Usage:
        # Core functional types
        predicate: FlextTypes.Core.Predicate[User] = lambda u: u.is_active
        factory: FlextTypes.Core.Factory[Connection] = ConnectionFactory()

        # Data domain types
        connection: FlextTypes.Data.Connection = oracle_connection
        credentials: FlextTypes.Data.Credentials = {"user": "admin"}

        # Authentication domain types
        token: FlextTypes.Auth.Token = jwt_token
        provider: FlextTypes.Auth.Provider = OAuthProvider()

    Extension Pattern for Subprojects:
        # In flext-target-oracle/types.py
        class FlextOracleTypes(FlextTypes):
            class Data(FlextTypes.Data):
                type OracleConnection = OracleConnectionConfig
                type OracleCredentials = OracleCredentialsConfig
    """

    class Core:
        """Core functional and architectural types."""

        # Functional types
        type Predicate[T] = Callable[[T], bool]
        type Factory[T] = Callable[[], T] | Callable[[object], T]
        type Transformer[T, R] = Callable[[T], R]
        type AsyncTransformer[T, R] = Callable[[T], Awaitable[R]]
        type Validator[T] = Callable[[T], bool | str]
        type Serializer[T] = Callable[[T], str | bytes | JsonDict]
        type Deserializer[T] = Callable[[str | bytes | JsonDict], T]

        # Container and dependency types
        type Container = Mapping[str, object]
        type ServiceLocator = Callable[[str], object]
        type ServiceFactory[T] = Callable[[], T]
        type ServiceRegistry = JsonDict

        # Result and error handling types
        type Result[T, E] = T | E
        type OptionalResult[T] = T | None
        type ErrorHandler[E] = Callable[[E], None]
        type Fallback[T] = Callable[[], T]

        # Event and messaging types
        type EventHandler[TEvent] = (
            Callable[[TEvent], None] | Callable[[TEvent], Awaitable[None]]
        )
        type EventBus = Callable[[object], None]
        type MessageHandler[T] = Callable[[T], object]
        type MessageQueue[T] = Sequence[T]

        # Metadata and configuration types
        type JsonDict = dict[str, object]  # Standard JSON-compatible dictionary
        type Metadata = JsonDict
        type Settings = JsonDict
        type Configuration = Mapping[str, object]
        type Environment = dict[str, str]

    class Data:
        """Data integration and storage domain types."""

        # Connection and database types
        type Connection = object  # Protocol-based, defined by each project
        type ConnectionString = str
        type ConnectionPool = Sequence[object]
        type DatabaseConnection = object
        type Credentials = dict[str, str]
        type ConnectionConfig = JsonDict

        # Data processing types
        type Record = JsonDict
        type RecordBatch = Sequence[JsonDict]
        type Schema = JsonDict
        type DataStream = Iterable[JsonDict] | Generator[JsonDict]
        type DataReader[T] = Callable[[], Iterable[T]]
        type DataWriter[T] = Callable[[Iterable[T]], None]

        # Query and operation types
        type Query = str | JsonDict
        type QueryParams = JsonDict
        type QueryResult = Sequence[JsonDict]
        type Operation = Callable[[], object]
        type Transaction = Callable[[], object]
        type Cursor = object

        # Serialization and format types
        type Serializable = JsonDict | list[object] | str | int | float | bool | None
        type JsonData = JsonDict | list[object]
        type CsvData = Sequence[Sequence[str]]
        type XmlData = str
        type BinaryData = bytes

        # Pipeline and ETL types
        type Pipeline = Sequence[Callable[[object], object]]
        type Extractor[T] = Callable[[], Iterable[T]]
        type Transformer[T, R] = Callable[[T], R]
        type Loader[T] = Callable[[Iterable[T]], None]

    class Auth:
        """Authentication and authorization domain types."""

        # Token and credential types
        type Token = str
        type RefreshToken = str
        type AccessToken = str
        type ApiKey = str
        type Secret = str
        type Password = str
        type Hash = str

        # Authentication types
        type AuthProvider = object  # Protocol-based
        type AuthenticatedUser = JsonDict
        type LoginCredentials = dict[str, str]
        type AuthContext = JsonDict
        type AuthSession = JsonDict

        # Authorization types
        type Permission = str
        type Role = str
        type Scope = str | Sequence[str]
        type Policy = JsonDict
        type AuthorizationContext = JsonDict

        # Security types
        type SecurityContext = JsonDict
        type SecurityPolicy = JsonDict
        type CryptographicKey = bytes | str
        type Certificate = str | bytes
        type Signature = str | bytes

    class Observability:
        """Monitoring and observability domain types."""

        # Logging types
        type Logger = object  # Protocol-based
        type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        type LogMessage = str
        type LogContext = JsonDict
        type LogEntry = JsonDict
        type LogHandler = Callable[[JsonDict], None]

        # Metrics types
        type Metric = JsonDict
        type MetricValue = int | float
        type MetricTags = dict[str, str]
        type Counter = int
        type Gauge = float
        type Histogram = Sequence[float]
        type MetricCollector = Callable[[], JsonDict]

        # Tracing types
        type Tracer = object  # Protocol-based
        type Span = object  # Protocol-based
        type SpanContext = object  # Protocol-based
        type TraceId = str
        type SpanId = str
        type CorrelationId = str
        type TraceContext = JsonDict

        # Alerting types
        type Alert = JsonDict
        type AlertHandler = Callable[[JsonDict], None]
        type AlertLevel = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
        type AlertCondition = Callable[[object], bool]
        type NotificationChannel = object

    class Singer:
        """Singer protocol and data integration domain types."""

        # Singer specification types
        type SingerStream = JsonDict
        type SingerRecord = JsonDict
        type SingerSchema = JsonDict
        type SingerCatalog = JsonDict
        type SingerConfig = JsonDict
        type SingerState = JsonDict

        # Tap and target types
        type Tap = object  # Protocol-based
        type Target = object  # Protocol-based
        type TapConfig = JsonDict
        type TargetConfig = JsonDict
        type StreamName = str
        type TableName = str

        # Replication types
        type ReplicationKey = str
        type ReplicationMethod = Literal["FULL_TABLE", "INCREMENTAL", "LOG_BASED"]
        type BookmarkProperties = JsonDict
        type SelectionCriteria = JsonDict

        # DBT integration types
        type DbtModel = JsonDict
        type DbtProject = JsonDict
        type DbtProfile = JsonDict
        type DbtManifest = JsonDict
        type DbtRunResult = JsonDict

    class Bridge:
        """Go-Python bridge and cross-language integration types."""

        # Message Types (following pattern documentation)
        type MessageId = str
        type MessageType = str
        type MessagePayload = JsonDict

        # Protocol Types
        type RequestId = str
        type ResponseId = str
        type ErrorCode = str

        # Serialization Types
        type SerializedData = bytes
        type EncodingType = Literal["json", "msgpack", "protobuf"]

        # Contract Types
        type ServiceName = str
        type MethodName = str
        type ServiceContract = dict[MethodName, JsonDict]

        # Bridge Message Structure (enhanced from pattern docs)
        type BridgeMessage = dict[
            str,
            object,
        ]  # Complete structure defined in protocols
        type BridgeRequest = JsonDict
        type BridgeResponse = JsonDict
        type BridgeError = dict[str, str]
        type BridgeContext = JsonDict

        # Cross-language compatibility types
        type SerializableType = (
            str | int | float | bool | None | JsonDict | list[object]
        )
        type GoCompatibleType = SerializableType
        type PythonToGoType = SerializableType
        type GoToPythonType = SerializableType

        # Service integration types
        type ServiceProxy = object  # Protocol-based
        type ServiceAdapter = object  # Protocol-based
        type ApiContract = JsonDict
        type MessageContract = JsonDict

    class Web:
        """Web and HTTP domain types."""

        # HTTP types
        type HttpMethod = Literal[
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
            "HEAD",
            "OPTIONS",
        ]
        type HttpStatus = int
        type HttpHeaders = dict[str, str]
        type HttpRequest = JsonDict
        type HttpResponse = JsonDict
        type HttpHandler = Callable[[JsonDict], JsonDict]

        # API types
        type ApiEndpoint = str
        type ApiResponse = JsonDict
        type ApiError = JsonDict
        type RestClient = object  # Protocol-based
        type GraphQLQuery = str
        type GraphQLVariables = JsonDict

        # Web application types
        type Route = str
        type RouteHandler = Callable[[JsonDict], JsonDict]
        type Middleware = Callable[[JsonDict], JsonDict]
        type WebContext = JsonDict
        type Session = JsonDict

    class CLI:
        """Command-line interface domain types."""

        # Command types
        type Command = str
        type CommandArgs = Sequence[str]
        type CommandOptions = JsonDict
        type CommandHandler = Callable[[Sequence[str]], int]
        type CliContext = JsonDict

        # Input/Output types
        type CliInput = str
        type CliOutput = str
        type CliError = str
        type ExitCode = int
        type CliFormatter = Callable[[object], str]

        # Configuration types
        type CliConfig = JsonDict
        type CliCommand = JsonDict
        type CliOption = JsonDict
        type CliArgument = JsonDict


# =============================================================================
# DOMAIN PROTOCOLS - Structural typing for cross-project compatibility
# =============================================================================


class ConnectionProtocol(Protocol):
    """Protocol for database and service connections."""

    def connect(self) -> bool:
        """Establish connection."""
        ...

    def disconnect(self) -> None:
        """Close connection."""
        ...

    def is_connected(self) -> bool:
        """Check connection status."""
        ...


class AuthProtocol(Protocol):
    """Protocol for authentication providers."""

    def authenticate(
        self,
        credentials: FlextTypes.Auth.LoginCredentials,
    ) -> FlextTypes.Auth.AuthenticatedUser | None:
        """Authenticate user with credentials."""
        ...

    def is_authenticated(self, context: FlextTypes.Auth.AuthContext) -> bool:
        """Check if context is authenticated."""
        ...


class ObservabilityProtocol(Protocol):
    """Protocol for observability components."""

    def log(
        self,
        level: FlextTypes.Observability.LogLevel,
        message: str,
        **context: object,
    ) -> None:
        """Log message with context."""
        ...

    def record_metric(
        self,
        name: str,
        value: FlextTypes.Observability.MetricValue,
        tags: FlextTypes.Observability.MetricTags | None = None,
    ) -> None:
        """Record metric value."""
        ...


class SingerProtocol(Protocol):
    """Protocol for Singer taps and targets."""

    def discover(
        self,
        config: FlextTypes.Singer.SingerConfig,
    ) -> FlextTypes.Singer.SingerCatalog:
        """Discover available streams."""
        ...

    def sync(
        self,
        config: FlextTypes.Singer.SingerConfig,
        catalog: FlextTypes.Singer.SingerCatalog,
        state: FlextTypes.Singer.SingerState | None = None,
    ) -> None:
        """Synchronize data."""
        ...


# =============================================================================
# UTILITY TYPE FACTORIES - Helper functions for common type patterns
# =============================================================================


class FlextTypeFactory:
    """Factory for creating common type patterns."""

    @staticmethod
    def predicate[T](func: Callable[[T], bool]) -> FlextTypes.Core.Predicate[T]:
        """Create a predicate function."""
        return func

    @staticmethod
    def factory[T](func: Callable[[], T]) -> FlextTypes.Core.Factory[T]:
        """Create a factory function."""
        return func

    @staticmethod
    def transformer[T, R](func: Callable[[T], R]) -> FlextTypes.Core.Transformer[T, R]:
        """Create a transformer function."""
        return func

    @staticmethod
    def validator[T](func: Callable[[T], bool | str]) -> FlextTypes.Core.Validator[T]:
        """Create a validator function."""
        return func

    @staticmethod
    def event_handler[TEvent](
        func: Callable[[TEvent], None],
    ) -> FlextTypes.Core.EventHandler[TEvent]:
        """Create an event handler."""
        return func


# =============================================================================
# EXTENSION PATTERNS - Templates for subproject type extensions
# =============================================================================


class FlextTypeExtension:
    """Base class for project-specific type extensions.

    Usage in subprojects:
        # In flext-target-oracle/types.py
        class FlextOracleTypes(FlextTypeExtension):
            class Data(FlextTypes.Data):
                type OracleConnection = OracleConnectionConfig
                type OracleCredentials = OracleCredentialsConfig
                type OracleQuery = OracleQueryBuilder
    """

    @classmethod
    def extend_core(cls) -> type[FlextTypes.Core]:
        """Extend core types with project-specific additions."""
        return FlextTypes.Core

    @classmethod
    def extend_data(cls) -> type[FlextTypes.Data]:
        """Extend data types with project-specific additions."""
        return FlextTypes.Data

    @classmethod
    def extend_auth(cls) -> type[FlextTypes.Auth]:
        """Extend auth types with project-specific additions."""
        return FlextTypes.Auth


# =============================================================================
# SEMANTIC ENUMS - Common enumeration types
# =============================================================================


class FlextConnectionType(StrEnum):
    """Semantic connection types across FLEXT ecosystem."""

    DATABASE = "database"
    REDIS = "redis"
    LDAP = "ldap"
    ORACLE = "oracle"
    POSTGRES = "postgres"
    REST_API = "rest_api"
    GRPC = "grpc"
    FILE = "file"
    STREAM = "stream"


class FlextDataFormat(StrEnum):
    """Semantic data formats."""

    JSON = "json"
    XML = "xml"
    CSV = "csv"
    LDIF = "ldif"
    YAML = "yaml"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class FlextOperationStatus(StrEnum):
    """Semantic operation status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class FlextLogLevel(StrEnum):
    """Semantic log levels."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# LEGACY COMPATIBILITY - Backward compatibility during migration
# =============================================================================

# Legacy type aliases for backward compatibility (will be deprecated)
TDict = dict[str, object]
TList = list[object]
TOptional = object | None
TCallable = Callable[[object], object]
TAnyDict = dict[str, object]
TStringDict = dict[str, str]

# Legacy function type aliases
TFactory = FlextTypes.Core.Factory[object]
TPredicate = FlextTypes.Core.Predicate[object]
TTransformer = FlextTypes.Core.Transformer[object, object]
TValidator = FlextTypes.Core.Validator[object]

# Legacy domain type aliases
TConnection = FlextTypes.Data.Connection
TCredentials = FlextTypes.Data.Credentials
TToken = FlextTypes.Auth.Token
TLogger = FlextTypes.Observability.Logger

# =============================================================================
# EXPORTS - Semantic foundation + protocols + legacy compatibility
# =============================================================================

__all__ = [
    "AuthProtocol",
    # Domain protocols
    "ConnectionProtocol",
    "E",
    # Semantic enums
    "FlextConnectionType",
    "FlextDataFormat",
    "FlextLogLevel",
    "FlextOperationStatus",
    "FlextTypeExtension",
    "FlextTypeFactory",
    # Core semantic type system
    "FlextTypes",
    "K",
    "ObservabilityProtocol",
    "SingerProtocol",
    # Type variables
    "T",
    "TAnyDict",
    "TCallable",
    "TCommand",
    "TConfig",
    "TConnection",
    "TCredentials",
    # Legacy compatibility (temporary during migration)
    "TDict",
    "TEntity",
    "TEvent",
    "TFactory",
    "TList",
    "TLogger",
    "TOptional",
    "TPredicate",
    "TQuery",
    "TResult",
    "TStringDict",
    "TToken",
    "TTransformer",
    "TValidator",
    "TValue",
    "V",
]

# Total exports: 29 items (13 semantic + 8 protocols/enums + 8 legacy)
# Meets requirement of ≤50 exports for core foundation
