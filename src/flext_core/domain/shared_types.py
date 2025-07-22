"""Shared typing system for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides the unified typing system that eliminates duplication across
all FLEXT projects. All types here use modern Python 3.13 patterns, Pydantic v2,
composition over inheritance, and follow the FLEXT standards.

ONLY ABSTRACT/GENERIC TYPES - No technology-specific implementations.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Literal
from typing import NotRequired
from typing import Protocol
from typing import TypedDict
from typing import TypeVar
from typing import runtime_checkable

from pydantic import BaseModel
from pydantic import Field
from pydantic import StringConstraints

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from collections.abc import Callable

# Generic type variable for ServiceResult
T = TypeVar("T")

# ==============================================================================
# FUNDAMENTAL BASE TYPES - SINGLE SOURCE OF TRUTH
# ==============================================================================

# Basic constrained types - foundation for all projects
type NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
type PositiveInt = Annotated[int, Field(gt=0)]
type NonNegativeInt = Annotated[int, Field(ge=0)]
type PositiveFloat = Annotated[float, Field(gt=0.0)]
type NonNegativeFloat = Annotated[float, Field(ge=0.0)]

# Network and URL types (generic)
type URL = Annotated[
    str,
    StringConstraints(pattern=r"^https?://[^\s/$.?#].[^\s]*$"),
    Field(description="HTTP/HTTPS URL"),
]

type Host = Annotated[
    str,
    StringConstraints(min_length=1, max_length=255),
    Field(description="Network host"),
]

type Port = Annotated[
    int,
    Field(ge=1, le=65535, description="Network port"),
]

# Generic database types (not specific to any technology)
type DatabaseName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=63),
    Field(description="Database name"),
]

type Username = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Username"),
]

type Password = Annotated[
    str,
    StringConstraints(min_length=1),
    Field(description="Password"),
]

type ApiKey = Annotated[
    str,
    StringConstraints(min_length=1),
    Field(description="API key"),
]

type Token = Annotated[
    str,
    StringConstraints(min_length=1),
    Field(description="Authentication token"),
]

type SchemaName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Schema name"),
]

# Environment and deployment types
type EnvironmentLiteral = Literal["development", "staging", "production", "test"]

# Project and application types
type ProjectName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Project name"),
]

# Generic timeout and performance types
type TimeoutSeconds = Annotated[
    int,
    Field(ge=1, le=3600, description="Timeout in seconds"),
]

type DurationSeconds = Annotated[
    int,
    Field(ge=0, description="Duration in seconds"),
]

type MemoryMB = Annotated[
    int,
    Field(ge=1, description="Memory in megabytes"),
]

type BatchSize = Annotated[
    int,
    Field(ge=1, le=100000, description="Batch size"),
]

type MaxRecords = Annotated[
    int,
    Field(ge=1, description="Maximum records"),
]

type ParallelStreams = Annotated[
    int,
    Field(ge=1, le=50, description="Parallel streams"),
]

# Generic retry types
type RetryCount = Annotated[
    int,
    Field(ge=0, le=10, description="Retry count"),
]

type RetryDelay = Annotated[
    float,
    Field(ge=0.1, le=60.0, description="Retry delay in seconds"),
]

# JSON types
type Json = dict[str, Any] | list[Any] | str | int | float | bool | None
type JsonDict = dict[str, Json]
type JsonSchema = dict[str, Any]

# Timestamp types
type TimestampISO = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$"),
    Field(description="ISO 8601 timestamp"),
]

# Generic ID types
type EntityId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Entity identifier"),
]

type UserId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="User identifier"),
]

type SessionId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Session identifier"),
]

type PluginId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Plugin identifier"),
]

type PipelineId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Pipeline identifier"),
]

# ==============================================================================
# GENERIC ENUMS - NOT TECHNOLOGY SPECIFIC
# ==============================================================================


class ExecutionStatus(StrEnum):
    """Generic execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EntityStatus(StrEnum):
    """Entity status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ARCHIVED = "archived"


class AuthMethod(StrEnum):
    """Generic authentication methods."""

    USERNAME_PASSWORD = "username_password"
    TOKEN = "token"
    API_KEY = "api_key"
    OAUTH = "oauth"
    CERTIFICATE = "certificate"


class ReplicationMethod(StrEnum):
    """Generic replication methods."""

    INCREMENTAL = "incremental"
    FULL_TABLE = "full_table"
    LOG_BASED = "log_based"
    CHANGE_DATA_CAPTURE = "cdc"


class WriteMode(StrEnum):
    """Generic write modes."""

    INSERT = "insert"
    UPDATE = "update"
    UPSERT = "upsert"
    REPLACE = "replace"
    APPEND = "append"


class PageMode(StrEnum):
    """Generic pagination modes."""

    OFFSET = "offset"
    TOKEN = "token"
    CURSOR = "cursor"
    PAGE = "page"


class FilterOperator(StrEnum):
    """Generic filter operators."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    NOT_LIKE = "not_like"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class LogLevel(StrEnum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(StrEnum):
    """Metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(StrEnum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PluginType(StrEnum):
    """Plugin types."""

    EXTRACTOR = "extractor"
    LOADER = "loader"
    TRANSFORMER = "transformer"
    ORCHESTRATOR = "orchestrator"
    UTILITY = "utility"


class TraceStatus(StrEnum):
    """Trace status."""

    STARTED = "started"
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPLETED = "completed"
    FAILED = "failed"


# ==============================================================================
# GENERIC PROTOCOL INTERFACES
# ==============================================================================


@runtime_checkable
class ExtractorInterface(Protocol):
    """Generic data extractor interface."""

    def extract(self, config: JsonDict) -> JsonDict: ...


@runtime_checkable
class TransformerInterface(Protocol):
    """Generic data transformer interface."""

    def transform(self, data: JsonDict) -> JsonDict: ...


@runtime_checkable
class LoaderInterface(Protocol):
    """Generic data loader interface."""

    def load(self, data: JsonDict) -> bool: ...


@runtime_checkable
class ConfigProviderInterface(Protocol):
    """Generic configuration provider interface."""

    def get_config(self, key: str) -> Any: ...
    def set_config(self, key: str, value: Any) -> None: ...


# ==============================================================================
# GENERIC BASE MODELS
# ==============================================================================


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class StatusMixin(BaseModel):
    """Mixin for status tracking."""

    status: ExecutionStatus = ExecutionStatus.PENDING
    error_message: str | None = None


class ConfigBase(BaseModel):
    """Base configuration model."""

    enabled: bool = True
    timeout_seconds: TimeoutSeconds = 30
    retry_count: RetryCount = 3
    retry_delay: RetryDelay = 1.0


class ServiceResult[T](BaseModel):
    """Generic service result wrapper."""

    success: bool
    data: T | None = None
    error: str | None = None
    metadata: JsonDict | None = None

    @property
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self.success

    @property
    def is_failure(self) -> bool:
        """Check if result is a failure."""
        return not self.success

    @classmethod
    def ok(
        cls,
        data: T | None = None,
        metadata: JsonDict | None = None,
    ) -> ServiceResult[T]:
        """Create successful result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str, metadata: JsonDict | None = None) -> ServiceResult[T]:
        """Create failure result."""
        return cls(success=False, error=error, metadata=metadata)

    @property
    def result_data(self) -> dict[str, Any]:
        """Get data as dict for safe access (legacy compatibility)."""
        if self.data is None:
            return {}
        if isinstance(self.data, dict):
            return self.data
        # Handle other types (list, primitive)
        return {"value": self.data}

    @property
    def status(self) -> str:
        """Get status string."""
        return "success" if self.success else "failure"

    def unwrap(self) -> T:
        """Unwrap the data value, raising error if failure."""
        if not self.success:
            raise ValueError(f"Attempted to unwrap failed result: {self.error}")
        if self.data is None:
            raise ValueError("Attempted to unwrap None data")
        return self.data

    def unwrap_or(self, default: T) -> T:
        """Unwrap the data value or return default if failure."""
        if not self.success or self.data is None:
            return default
        return self.data

    def map(self, func: Callable[[T], Any]) -> ServiceResult[Any]:
        """Map the data value using a function."""
        if not self.success or self.data is None:
            return ServiceResult.fail(self.error or "No data to map")
        try:
            mapped_data = func(self.data)
            return ServiceResult.ok(mapped_data, self.metadata)
        except (TypeError, ValueError, AttributeError) as e:
            return ServiceResult.fail(f"Mapping failed: {e}")
        except Exception as e:
            # Generic mapping function can receive any user function - broad exception is intentional
            return ServiceResult.fail(f"Mapping failed: {e}")

    def and_then(self, func: Callable[[T], ServiceResult[Any]]) -> ServiceResult[Any]:
        """Chain operations that return ServiceResult."""
        if not self.success or self.data is None:
            return ServiceResult.fail(self.error or "No data to chain")
        try:
            return func(self.data)
        except (TypeError, ValueError, AttributeError) as e:
            return ServiceResult.fail(f"Chaining failed: {e}")
        except Exception as e:
            # Generic chaining function can receive any user function - broad exception is intentional
            return ServiceResult.fail(f"Chaining failed: {e}")


# ==============================================================================
# GENERIC TYPED DICTIONARIES
# ==============================================================================


class ConnectionInfo(TypedDict):
    """Generic connection information."""

    host: Host
    port: Port
    username: Username
    auth_method: AuthMethod


class EntityInfo(TypedDict):
    """Generic entity information."""

    entity_id: EntityId
    display_name: str
    description: str
    schema: JsonSchema


class PipelineDict(TypedDict):
    """Generic pipeline configuration."""

    pipeline_id: PipelineId
    source: JsonDict
    target: JsonDict
    transformations: NotRequired[list[JsonDict]]


class PluginDict(TypedDict):
    """Generic plugin configuration."""

    plugin_id: PluginId
    plugin_type: str
    config: JsonDict


# ==============================================================================
# EXPORTS - ONLY GENERIC/ABSTRACT TYPES
# ==============================================================================

__all__ = [
    "URL",
    # Enums
    "AlertSeverity",
    "ApiKey",
    "AuthMethod",
    # Base types
    "BatchSize",
    "ConfigBase",
    "ConfigMapping",
    "ConfigProviderInterface",
    "ConnectionInfo",
    "DatabaseName",
    "DurationSeconds",
    "EntityId",
    "EntityInfo",
    "EntityStatus",
    "EnvironmentLiteral",
    "ExecutionStatus",
    "ExtractorInterface",
    "FilterOperator",
    "HandlerFunction",
    "Host",
    "Json",
    "JsonDict",
    "JsonSchema",
    "LoaderInterface",
    "LogLevel",
    "MaxRecords",
    "MemoryMB",
    "MetricType",
    "NonEmptyStr",
    "NonNegativeFloat",
    "NonNegativeInt",
    "PageMode",
    "ParallelStreams",
    "Password",
    "PipelineDict",
    "PipelineId",
    "PluginDict",
    "PluginId",
    "PluginType",
    "Port",
    "PositiveFloat",
    "PositiveInt",
    "ProjectName",
    "ReplicationMethod",
    "RetryCount",
    "RetryDelay",
    "SchemaName",
    "ServiceResult",
    "SessionId",
    "StatusMixin",
    "T",
    "TimeoutSeconds",
    "TimestampISO",
    "TimestampMixin",
    "Token",
    "TraceStatus",
    "TransformerInterface",
    "UserId",
    "Username",
    "WriteMode",
]

# Additional generic types
type ConfigurationKey = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Configuration key"),
]

type ConfigurationValue = Annotated[
    str,
    Field(description="Configuration value"),
]

type FileName = Annotated[
    str,
    StringConstraints(min_length=1),
    Field(description="File name"),
]

type FilePath = Annotated[
    str,
    StringConstraints(min_length=1),
    Field(description="File path"),
]

type Version = Annotated[
    str,
    StringConstraints(pattern=r"^\d+\.\d+\.\d+"),
    Field(description="Version string"),
]

# gRPC and Handler Types (used by flext-grpc)
type ConfigMapping = dict[str, Any]
type HandlerFunction = Callable[..., Awaitable[Any]]
