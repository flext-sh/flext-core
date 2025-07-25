"""Advanced Type System for FLEXT Core - Reduce Application Boilerplate.

This module provides advanced type aliases, generic types, and type utilities
that dramatically reduce boilerplate code in applications.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import TypeVar

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Awaitable

# =============================================================================
# GENERIC TYPE VARIABLES
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")
E = TypeVar("E", bound=Exception)

# Specific constraint types
EntityT = TypeVar("EntityT")
ValueObjectT = TypeVar("ValueObjectT")
AggregateT = TypeVar("AggregateT")
EventT = TypeVar("EventT")
CommandT = TypeVar("CommandT")
QueryT = TypeVar("QueryT")

# =============================================================================
# RESULT AND ERROR HANDLING TYPES
# =============================================================================

# Result pattern shortcuts
ResultOf = "FlextResult[T] | T"
MaybeResult = "FlextResult[T] | None"
AsyncResult = "Awaitable[FlextResult[T]]"

# Error handling shortcuts
ErrorOr = T | Exception
ErrorHandler = Callable[[Exception], "FlextResult[T]"]
RetryStrategy = Callable[[int, Exception], bool]

# =============================================================================
# FUNCTION SIGNATURE SHORTCUTS
# =============================================================================

# Common function patterns
Validator = Callable[[T], "FlextResult[T]"]
Transformer = Callable[[T], R]
AsyncTransformer = Callable[[T], "Awaitable[R]"]
Predicate = Callable[[T], bool]
AsyncPredicate = Callable[[T], "Awaitable[bool]"]

# Service patterns
ServiceFactory = Callable[..., T]
ServiceInitializer = Callable[[T], "FlextResult[None]"]
ServiceDestroyer = Callable[[T], "FlextResult[None]"]

# Event handling
EventHandler = Callable[[EventT], "FlextResult[None]"]
AsyncEventHandler = Callable[[EventT], "Awaitable[FlextResult[None]]"]

# Command/Query patterns
CommandHandler = Callable[[CommandT], "FlextResult[R]"]
QueryHandler = Callable[[QueryT], "FlextResult[R]"]
AsyncCommandHandler = Callable[[CommandT], "Awaitable[FlextResult[R]]"]
AsyncQueryHandler = Callable[[QueryT], "Awaitable[FlextResult[R]]"]

# =============================================================================
# DATA STRUCTURE SHORTCUTS
# =============================================================================

# Common data structures
FlextDict = dict[str, Any]
FlextMapping = Mapping[str, Any]
FlextSequence = Sequence[T]
FlextList = list[T]

# Metadata and configuration
MetadataDict = dict[str, Any]
ConfigDict = dict[str, Any]
HeadersDict = dict[str, str]
QueryParams = dict[str, str | int | bool]

# Timestamps and IDs
Timestamp = datetime
EntityId = str
UserId = str
SessionId = str
RequestId = str
CorrelationId = str

# =============================================================================
# PROTOCOL DEFINITIONS (STRUCTURAL TYPING)
# =============================================================================

class Identifiable(Protocol):
    """Protocol for objects that have an ID."""

    @property
    def id(self) -> EntityId:
        """Return the object's unique identifier."""
        ...


class Timestamped(Protocol):
    """Protocol for objects with timestamps."""

    @property
    def created_at(self) -> Timestamp:
        """Return creation timestamp."""
        ...

    @property
    def updated_at(self) -> Timestamp:
        """Return last update timestamp."""
        ...


class Serializable(Protocol):
    """Protocol for objects that can be serialized."""

    def to_dict(self) -> FlextDict:
        """Convert to dictionary representation."""
        ...

    @classmethod
    def from_dict(cls, data: FlextDict) -> FlextResult[Self]:
        """Create from dictionary representation."""
        ...


class Validatable(Protocol):
    """Protocol for objects that can be validated."""

    def validate(self) -> FlextResult[None]:
        """Validate the object."""
        ...


class Cacheable(Protocol):
    """Protocol for objects that can be cached."""

    @property
    def cache_key(self) -> str:
        """Return cache key for this object."""
        ...

    @property
    def cache_ttl(self) -> int:
        """Return cache TTL in seconds."""
        ...


# =============================================================================
# GENERIC CLASSES FOR COMMON PATTERNS
# =============================================================================

class Repository[T](Protocol):
    """Generic repository protocol."""

    def find_by_id(self, entity_id: EntityId) -> FlextResult[T]:
        """Find entity by ID."""
        ...

    def save(self, entity: T) -> FlextResult[None]:
        """Save entity."""
        ...

    def delete(self, entity_id: EntityId) -> FlextResult[None]:
        """Delete entity by ID."""
        ...

    def find_all(self) -> FlextResult[list[T]]:
        """Find all entities."""
        ...


class Service[T](Protocol):
    """Generic service protocol."""

    def execute(self, request: T) -> FlextResult[R]:
        """Execute service operation."""
        ...


class Factory[T](Protocol):
    """Generic factory protocol."""

    def create(self, **kwargs: Any) -> FlextResult[T]:
        """Create new instance."""
        ...


class Builder[T](Protocol):
    """Generic builder protocol."""

    def build(self) -> FlextResult[T]:
        """Build the final object."""
        ...

    def reset(self) -> Builder[T]:
        """Reset builder to initial state."""
        ...


# =============================================================================
# UNION TYPES FOR FLEXIBILITY
# =============================================================================

# ID types
AnyId = EntityId | UserId | SessionId | RequestId | str

# Data input types
DataInput = FlextDict | str | bytes | Any
DataOutput = FlextDict | str | bytes | Any

# Configuration types
ConfigValue = str | int | float | bool | list[Any] | dict[str, Any]
ConfigSource = FlextDict | str  # dict or file path

# Error types
AnyError = Exception | str
ValidationError = ValueError | TypeError | Exception

# Time types
TimeInput = datetime | str | int | float  # Various time formats
Duration = int | float  # seconds

# =============================================================================
# TYPE ALIASES FOR SPECIFIC DOMAINS
# =============================================================================

# Web/API types
HttpMethod = str  # GET, POST, PUT, DELETE, etc.
HttpStatus = int  # 200, 404, 500, etc.
UrlPath = str
QueryString = str

# Database types
DatabaseUrl = str
TableName = str
ColumnName = str
SqlQuery = str

# Message/Event types
MessageType = str
EventName = str
QueueName = str
TopicName = str

# File/Storage types
FilePath = str
FileName = str
FileSize = int
MimeType = str
StorageKey = str

# Security types
Token = str
ApiKey = str
Secret = str
Hash = str
Salt = str

# Business domain types
Email = str
PhoneNumber = str
Currency = str
Amount = float
Percentage = float

# =============================================================================
# CALLABLE TYPE SHORTCUTS
# =============================================================================

# Lifecycle hooks
InitHook = Callable[[], "FlextResult[None]"]
CleanupHook = Callable[[], "FlextResult[None]"]
HealthCheck = Callable[[], "FlextResult[bool]"]

# Middleware patterns
Middleware = Callable[[T], "FlextResult[T]"]
AsyncMiddleware = Callable[[T], "Awaitable[FlextResult[T]]"]

# Serialization
Serializer = Callable[[T], "FlextResult[str]"]
Deserializer = Callable[[str], "FlextResult[T]"]
JsonSerializer = Callable[[T], "FlextResult[FlextDict]"]
JsonDeserializer = Callable[[FlextDict], "FlextResult[T]"]

# Authentication/Authorization
Authenticator = Callable[[str], "FlextResult[UserId]"]
Authorizer = Callable[[UserId, str], "FlextResult[bool]"]

# Monitoring/Observability
MetricCollector = Callable[[str, float], None]
Logger = Callable[[str, str], None]
Tracer = Callable[[str], Callable[[], None]]

# =============================================================================
# ADVANCED GENERIC TYPES
# =============================================================================

class Pipe[T, R]:
    """Type-safe pipe for data transformation."""

    def __init__(self, transformer: Callable[[T], FlextResult[R]]) -> None:
        """Initialize pipe with transformer function."""
        self._transformer = transformer

    def __call__(self, input_value: T) -> FlextResult[R]:
        """Apply transformation to input value."""
        return self._transformer(input_value)

    def then(self, next_pipe: Pipe[R, Any]) -> Pipe[T, Any]:
        """Chain another pipe."""
        def combined(input_value: T) -> FlextResult[Any]:
            result = self(input_value)
            if result.is_failure:
                return result
            return next_pipe(result.data)

        return Pipe(combined)


class Either[T, R]:
    """Either type for representing success/failure without exceptions."""

    def __init__(self, value: T | R, *, is_left: bool) -> None:
        """Initialize Either with value and side indicator."""
        self._value = value
        self._is_left = is_left

    @classmethod
    def left(cls, value: T) -> Either[T, R]:
        """Create a left (error) value."""
        return cls(value, is_left=True)

    @classmethod
    def right(cls, value: R) -> Either[T, R]:
        """Create a right (success) value."""
        return cls(value, is_left=False)

    @property
    def is_left(self) -> bool:
        """Check if this is a left (error) value."""
        return self._is_left

    @property
    def is_right(self) -> bool:
        """Check if this is a right (success) value."""
        return not self._is_left

    def map(self, func: Callable[[R], Any]) -> Either[T, Any]:
        """Transform right value if present."""
        if self.is_right:
            return Either.right(func(self._value))
        return Either.left(self._value)

    def flat_map(self, func: Callable[[R], Either[T, Any]]) -> Either[T, Any]:
        """Transform right value to another Either."""
        if self.is_right:
            return func(self._value)
        return Either.left(self._value)


# =============================================================================
# TYPE CHECKING UTILITIES
# =============================================================================

def is_result_type(obj: object) -> bool:
    """Check if object is a FlextResult."""
    return hasattr(obj, "is_success") and hasattr(obj, "is_failure")


def is_identifiable(obj: object) -> bool:
    """Check if object implements Identifiable protocol."""
    return hasattr(obj, "id")


def is_timestamped(obj: object) -> bool:
    """Check if object implements Timestamped protocol."""
    return hasattr(obj, "created_at") and hasattr(obj, "updated_at")


def is_serializable(obj: object) -> bool:
    """Check if object implements Serializable protocol."""
    return hasattr(obj, "to_dict") and hasattr(obj, "from_dict")


def is_validatable(obj: object) -> bool:
    """Check if object implements Validatable protocol."""
    return hasattr(obj, "validate")


# =============================================================================
# TYPE CONVERSION UTILITIES
# =============================================================================

def ensure_result[T](value: ResultOf[T]) -> FlextResult[T]:
    """Ensure value is wrapped in FlextResult."""
    if is_result_type(value):
        return value
    return FlextResult.ok(value)


def ensure_list(value: T | Sequence[T]) -> list[T]:
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def ensure_dict(value: object) -> FlextDict:
    """Ensure value is a dictionary."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return {"value": value}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AggregateT",
    "Amount",
    "AnyError",
    # Union types
    "AnyId",
    "AsyncPredicate",
    "AsyncResult",
    "AsyncTransformer",
    "Builder",
    "Cacheable",
    "CommandHandler",
    "CommandT",
    "ConfigDict",
    "ConfigValue",
    "CorrelationId",
    "Currency",
    "DataInput",
    "DataOutput",
    "E",
    "Either",
    # Domain-specific types
    "Email",
    # ID and timestamp types
    "EntityId",
    "EntityT",
    "ErrorHandler",
    "ErrorOr",
    "EventHandler",
    "EventT",
    "Factory",
    "FilePath",
    # Data structure types
    "FlextDict",
    "FlextList",
    "FlextMapping",
    "FlextSequence",
    "HeadersDict",
    # Protocols
    "Identifiable",
    "MaybeResult",
    "MetadataDict",
    "PhoneNumber",
    # Advanced types
    "Pipe",
    "Predicate",
    "QueryHandler",
    "QueryParams",
    "QueryT",
    "R",
    "Repository",
    "RequestId",
    # Result types
    "ResultOf",
    "Serializable",
    "Service",
    "ServiceFactory",
    "SessionId",
    # Type variables
    "T",
    "Timestamp",
    "Timestamped",
    "Token",
    "Transformer",
    "UserId",
    "Validatable",
    # Function types
    "Validator",
    "ValueObjectT",
    "ensure_dict",
    "ensure_list",
    "ensure_result",
    "is_identifiable",
    # Utilities
    "is_result_type",
    "is_serializable",
    "is_timestamped",
]
