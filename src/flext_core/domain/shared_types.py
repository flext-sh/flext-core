"""Shared typing system for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides the unified typing system that eliminates duplication across
all FLEXT projects. All types here use modern Python 3.13 patterns, Pydantic v2,
composition over inheritance, and follow the FLEXT standards.
"""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Literal
from typing import NotRequired
from typing import Protocol
from typing import Self
from typing import TypedDict
from typing import TypeVar
from typing import runtime_checkable
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field
from pydantic import StringConstraints
from pydantic import field_validator

from flext_core.domain.constants import RegexPatterns

# ==============================================================================
# FUNDAMENTAL BASE TYPES - SINGLE SOURCE OF TRUTH
# ==============================================================================

# Basic constrained types - foundation for all projects
type NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
type PositiveInt = Annotated[int, Field(gt=0)]
type NonNegativeInt = Annotated[int, Field(ge=0)]
type PositiveFloat = Annotated[float, Field(gt=0.0)]
type NonNegativeFloat = Annotated[float, Field(ge=0.0)]

# Network and URL types
type URL = Annotated[
    str,
    StringConstraints(pattern=RegexPatterns.HTTP_URL),
    Field(description="Valid HTTP/HTTPS URL"),
]
type RedisURL = Annotated[
    str,
    StringConstraints(pattern=RegexPatterns.REDIS_URL),
    Field(description="Valid Redis URL"),
]
type Port = Annotated[int, Field(ge=1, le=65535, description="Network port")]
type IPAddress = Annotated[
    str,
    StringConstraints(pattern=r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"),
    Field(description="IPv4 address"),
]

# File system types
type FilePath = Annotated[Path, Field(description="File system path")]
type DirPath = Annotated[Path, Field(description="Directory path")]
type FileName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=255),
    Field(description="File name"),
]

# Time and duration types
type TimeoutSeconds = Annotated[
    float,
    Field(ge=0.1, le=3600.0, description="Timeout in seconds"),
]
type DurationSeconds = Annotated[
    float,
    Field(ge=0.0, description="Duration in seconds"),
]
type TimestampISO = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
    Field(description="ISO 8601 timestamp"),
]

# Credential and authentication types
type Username = Annotated[
    str,
    StringConstraints(min_length=1, max_length=50),
    Field(description="User name"),
]
type Password = Annotated[
    str,
    StringConstraints(min_length=1),
    Field(description="Password (store securely)"),
]
type Token = Annotated[
    str,
    StringConstraints(min_length=1),
    Field(description="Authentication token"),
]
type ApiKey = Annotated[
    str,
    StringConstraints(min_length=1),
    Field(description="API key"),
]

# Batch and performance types
type BatchSize = Annotated[
    int,
    Field(ge=1, le=10000, description="Batch processing size"),
]
type RetryCount = Annotated[int, Field(ge=0, le=10, description="Number of retries")]
type RetryDelay = Annotated[
    float,
    Field(ge=0.0, le=60.0, description="Delay between retries"),
]

# Memory and resource types
type MemoryMB = Annotated[int, Field(gt=0, description="Memory in megabytes")]
type DiskMB = Annotated[int, Field(gt=0, description="Disk space in megabytes")]
type CpuPercent = Annotated[
    float,
    Field(ge=0.0, le=100.0, description="CPU usage percentage"),
]

# Data size types
type FileSize = Annotated[int, Field(ge=0, description="File size in bytes")]

# Configuration types extending base
type ConfigurationKey = Annotated[
    str,
    StringConstraints(min_length=1, max_length=100, pattern=RegexPatterns.CONFIG_KEY),
    Field(description="Configuration key"),
]
type ConfigurationValue = str | int | float | bool | None

# Database types
type DatabaseName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=63),
    Field(description="Database name"),
]
type DatabaseURL = Annotated[
    str,
    StringConstraints(pattern=r"^(postgresql|mysql|sqlite|oracle)://"),
    Field(description="Database connection URL"),
]

# Oracle Database types
type OracleHost = Annotated[
    str,
    StringConstraints(min_length=1, max_length=255),
    Field(description="Oracle database host"),
]
type OraclePort = Annotated[
    int,
    Field(ge=1, le=65535, description="Oracle database port"),
]
type OracleServiceName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Oracle service name"),
]
type OracleSID = Annotated[
    str,
    StringConstraints(min_length=1, max_length=8),
    Field(description="Oracle SID"),
]
type OracleUsername = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Oracle username"),
]
type OraclePassword = Annotated[
    str,
    StringConstraints(min_length=1),
    Field(description="Oracle password"),
]
type OracleSchema = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Oracle schema name"),
]
type OracleQueryTimeout = Annotated[
    int,
    Field(ge=1, le=3600, description="Oracle query timeout in seconds"),
]
type OracleFetchSize = Annotated[
    int,
    Field(ge=1, le=10000, description="Oracle fetch size for cursor"),
]
type OracleArraySize = Annotated[
    int,
    Field(ge=1, le=10000, description="Oracle array size for batch operations"),
]

# Singer protocol types
type SingerBatchSize = Annotated[
    int,
    Field(ge=1, le=100000, description="Singer batch size"),
]
type SingerMaxRecords = Annotated[
    int,
    Field(ge=1, description="Singer maximum records"),
]
type SingerParallelStreams = Annotated[
    int,
    Field(ge=1, le=50, description="Singer parallel streams"),
]
type SingerReplicationMethod = Literal["INCREMENTAL", "FULL_TABLE", "LOG_BASED"]
type SingerStateInterval = Annotated[
    int,
    Field(ge=1, le=10000, description="Singer state interval"),
]

# JSON types
type Json = dict[str, Any] | list[Any] | str | int | float | bool | None
type JsonDict = dict[str, Json]
type JsonList = list[Json]
type JsonSchema = dict[str, Any]

# ==============================================================================
# BUSINESS DOMAIN TYPES
# ==============================================================================

# Entity identifiers using proper UUID typing
type EntityId = Annotated[UUID, Field(description="Unique entity identifier")]
type UserId = Annotated[UUID, Field(description="User identifier")]
type PipelineId = Annotated[UUID, Field(description="Pipeline identifier")]
type PluginId = Annotated[UUID, Field(description="Plugin identifier")]
type SessionId = Annotated[UUID, Field(description="Session identifier")]
type TransactionId = Annotated[UUID, Field(description="Transaction identifier")]

# Project and versioning types
type ProjectName = Annotated[
    str,
    StringConstraints(min_length=2, max_length=100, pattern=RegexPatterns.PROJECT_NAME),
    Field(description="Project name following naming conventions"),
]
type Version = Annotated[
    str,
    StringConstraints(pattern=RegexPatterns.SEMANTIC_VERSION),
    Field(description="Semantic version (e.g., 1.0.0, 1.0.0-alpha)"),
]

# Environment and log level types
type EnvironmentLiteral = Literal["development", "staging", "production", "test"]
type LogLevelLiteral = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Timestamps with proper typing
type CreatedAt = Annotated[datetime, Field(description="Creation timestamp")]
type UpdatedAt = Annotated[datetime, Field(description="Last update timestamp")]

# ==============================================================================
# ORACLE WMS SPECIFIC TYPES
# ==============================================================================

# WMS company and facility codes with proper validation
type WMSCompanyCode = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z0-9*]{1,10}$", min_length=1, max_length=10),
    Field(description="Oracle WMS company code"),
]

type WMSFacilityCode = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z0-9*]{1,10}$", min_length=1, max_length=10),
    Field(description="Oracle WMS facility code"),
]

# WMS entity name with supported entities
type WMSEntityName = Literal[
    "allocation",
    "order_hdr",
    "order_dtl",
    "item_master",
    "location",
    "inventory",
    "shipment",
    "receipt",
    "pick_slip",
    "task",
]

# WMS field and data types
type WMSFieldName = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z][a-z0-9_]*$", min_length=1, max_length=128),
    Field(description="WMS field name in snake_case"),
]

type WMSFieldMapping = dict[WMSFieldName, WMSFieldName]

# WMS identifier types
type WMSItemID = Annotated[
    str,
    StringConstraints(min_length=1, max_length=50),
    Field(description="WMS item identifier"),
]

type WMSLocationID = Annotated[
    str,
    StringConstraints(min_length=1, max_length=50),
    Field(description="WMS location identifier"),
]

type WMSOrderNumber = Annotated[
    str,
    StringConstraints(min_length=1, max_length=50),
    Field(description="WMS order number"),
]

# ==============================================================================
# SINGER PROTOCOL TYPES
# ==============================================================================

# Singer protocol specific types for data integration
type SingerStreamName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Singer stream name"),
]

type SingerSchemaName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
    Field(description="Singer schema name"),
]

type SingerRecordCount = Annotated[
    int,
    Field(ge=0, description="Number of Singer records"),
]

type SingerBookmark = dict[str, Any]
type SingerCatalog = dict[str, Any]
type SingerState = dict[str, Any]

# ==============================================================================
# ENUMS - COMPREHENSIVE STATUS AND TYPE ENUMS
# ==============================================================================


class Environment(StrEnum):
    """Environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(StrEnum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ResultStatus(StrEnum):
    """Result status enumeration."""

    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class EntityStatus(StrEnum):
    """Entity status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    DRAFT = "draft"
    ARCHIVED = "archived"


class PluginType(StrEnum):
    """Plugin type enumeration."""

    TAP = "tap"
    TARGET = "target"
    TRANSFORM = "transform"
    UTILITY = "utility"
    DBT = "dbt"
    ORCHESTRATOR = "orchestrator"


class MetricType(StrEnum):
    """Metric type enumeration for observability."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(StrEnum):
    """Alert severity enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TraceStatus(StrEnum):
    """Trace status enumeration."""

    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Oracle WMS specific enums
class OracleWMSAuthMethod(StrEnum):
    """Oracle WMS authentication methods."""

    USERNAME_PASSWORD = "username_password"  # noqa: S105
    TOKEN = "token"  # noqa: S105
    API_KEY = "api_key"
    OAUTH = "oauth"


class OracleWMSEntityType(StrEnum):
    """Oracle WMS entity types."""

    ALLOCATION = "allocation"
    ORDER_HDR = "order_hdr"
    ORDER_DTL = "order_dtl"
    ITEM_MASTER = "item_master"
    LOCATION = "location"
    INVENTORY = "inventory"
    SHIPMENT = "shipment"
    RECEIPT = "receipt"
    PICK_SLIP = "pick_slip"
    TASK = "task"


class OracleWMSPageMode(StrEnum):
    """Oracle WMS pagination modes."""

    OFFSET = "offset"
    TOKEN = "token"  # noqa: S105
    CURSOR = "cursor"


class OracleWMSWriteMode(StrEnum):
    """Oracle WMS write modes."""

    INSERT = "insert"
    UPDATE = "update"
    UPSERT = "upsert"
    REPLACE = "replace"


class OracleWMSFilterOperator(StrEnum):
    """Oracle WMS filter operators."""

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


# ==============================================================================
# COMPOSITION MIXINS - REUSABLE COMPONENTS
# ==============================================================================


class TimestampMixin(BaseModel):
    """Mixin for entities with timestamps."""

    created_at: CreatedAt = Field(default_factory=datetime.now)
    updated_at: UpdatedAt | None = None


class IdentifierMixin(BaseModel):
    """Mixin for entities with ID."""

    id: EntityId = Field(description="Unique identifier")


class EntityMixin(IdentifierMixin, TimestampMixin):
    """Mixin combining ID and timestamps."""


class StatusMixin(BaseModel):
    """Mixin for entities with status."""

    status: EntityStatus = Field(default=EntityStatus.ACTIVE)

    def activate(self) -> None:
        """Activate the entity."""
        self.status = EntityStatus.ACTIVE

    def deactivate(self) -> None:
        """Deactivate the entity."""
        self.status = EntityStatus.INACTIVE

    def is_active(self) -> bool:
        """Check if entity is active."""
        return self.status == EntityStatus.ACTIVE


class ProjectMixin(BaseModel):
    """Mixin for project-related entities."""

    project_name: ProjectName
    project_version: Version = Field(default="0.1.0")

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        """Validate project name format."""
        if not v or len(v) < 2:
            msg = "Project name must be at least 2 characters"
            raise ValueError(msg)

        if not v.replace("-", "").replace("_", "").isalnum():
            msg = "Project name must be alphanumeric with hyphens/underscores"
            raise ValueError(
                msg,
            )
        return v


class EnvironmentMixin(BaseModel):
    """Mixin for environment configuration."""

    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT


# ==============================================================================
# PROTOCOLS - INTERFACE DEFINITIONS USING COMPOSITION
# ==============================================================================


@runtime_checkable
class EntityProtocol(Protocol):
    """Protocol for all entities."""

    id: EntityId
    created_at: CreatedAt
    updated_at: UpdatedAt | None

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


# ==============================================================================
# INTERFACE SEGREGATION PRINCIPLE - FOCUSED PROTOCOLS
# ==============================================================================


@runtime_checkable
class ConfigurableProtocol(Protocol):
    """Protocol for objects that can be converted to dictionaries."""

    def to_dict(self, *, exclude_unset: bool = True) -> dict[str, Any]: ...


@runtime_checkable
class MergeableProtocol(Protocol):
    """Protocol for objects that can be merged with other instances."""

    def merge(self, other: Self) -> Self: ...


@runtime_checkable
class ProjectSettingsProtocol(Protocol):
    """Protocol for project configuration data only."""

    project_name: ProjectName
    project_version: Version
    environment: Environment
    debug: bool


@runtime_checkable
class DependencyConfiguratorProtocol(Protocol):
    """Protocol for objects that can configure dependencies."""

    def configure_dependencies(self, container: Any) -> None: ...


@runtime_checkable
class QueryServiceProtocol(Protocol):
    """Protocol for read-only query services."""

    async def query(self, criteria: Any) -> Any: ...


@runtime_checkable
class CommandServiceProtocol(Protocol):
    """Protocol for write operation command services."""

    async def execute(self, command: Any) -> Any: ...


@runtime_checkable
class ValidationServiceProtocol(Protocol):
    """Protocol for validation services."""

    async def validate(self, data: Any) -> Any: ...


# ==============================================================================
# GENERIC COLLECTION TYPES
# ==============================================================================

# Modern Python 3.13 type variables
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
TEntity = TypeVar("TEntity", bound=EntityProtocol)
TConfigurable = TypeVar("TConfigurable", bound=ConfigurableProtocol)

# Generic collections
type EntityCollection[TEntity: EntityProtocol] = Sequence[TEntity]
type ConfigMapping = Mapping[ConfigurationKey, ConfigurationValue]
type HandlerFunction[T, U] = Callable[[T], Awaitable[U]]

# ==============================================================================
# TYPED DICTIONARIES - STRUCTURED DATA
# ==============================================================================


class BaseEntityDict(TypedDict):
    """Base entity dictionary."""

    id: EntityId
    created_at: CreatedAt
    updated_at: NotRequired[UpdatedAt]


class BaseConfigDict(TypedDict):
    """Base configuration dictionary."""

    project_name: ProjectName
    project_version: Version
    environment: EnvironmentLiteral
    debug: bool


class PipelineDict(BaseEntityDict):
    """Pipeline entity dictionary."""

    name: str
    description: str
    status: Literal["active", "inactive", "error"]


class PluginDict(BaseEntityDict):
    """Plugin entity dictionary."""

    name: str
    type: Literal["tap", "target", "transform"]
    version: Version


# WMS specific dictionaries
class WMSConnectionInfo(TypedDict):
    """WMS connection information."""

    base_url: URL
    api_version: str
    auth_method: OracleWMSAuthMethod
    username: Username
    company_code: WMSCompanyCode
    facility_code: WMSFacilityCode


class WMSEntityInfo(TypedDict):
    """WMS entity information."""

    entity_name: OracleWMSEntityType
    display_name: str
    description: str
    primary_key: str
    replication_key: str | None
    schema: JsonSchema


# ==============================================================================
# RESULT PATTERN WITH COMPOSITION
# ==============================================================================


class ServiceResult[T]:
    """Type-safe result pattern with error handling."""

    __slots__ = ("_data", "_error", "_status", "_success")

    def __init__(
        self,
        *,
        success: bool,
        data: T | None = None,
        error: str | None = None,
        status: ResultStatus = ResultStatus.SUCCESS,
    ) -> None:
        """Initialize result."""
        self._success = success
        self._data = data
        self._error = error
        self._status = status if success else ResultStatus.ERROR

    @property
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self._success

    @property
    def is_failure(self) -> bool:
        """Check if result is a failure."""
        return not self._success

    @property
    def data(self) -> T | None:
        """Get result data."""
        return self._data

    @property
    def error(self) -> str | None:
        """Get error message."""
        return self._error

    @property
    def status(self) -> ResultStatus:
        """Get result status."""
        return self._status

    @classmethod
    def ok(cls, data: T) -> Self:
        """Create successful result."""
        return cls(success=True, data=data, status=ResultStatus.SUCCESS)

    @classmethod
    def fail(cls, error: str) -> Self:
        """Create failed result."""
        return cls(success=False, error=error, status=ResultStatus.ERROR)

    @classmethod
    def pending(cls) -> Self:
        """Create pending result."""
        return cls(success=False, status=ResultStatus.PENDING)

    def unwrap(self) -> T:
        """Unwrap result data or raise."""
        if not self._success or self._data is None:
            msg = f"Cannot unwrap failed result: {self._error}"
            raise RuntimeError(msg)
        return self._data

    def unwrap_or(self, default: U) -> T | U:
        """Unwrap result data or return default."""
        return self._data if self._success and self._data is not None else default

    def map[V](self, func: Callable[[T], V]) -> ServiceResult[V]:
        """Map result to new result."""
        if self._success and self._data is not None:
            try:
                return ServiceResult.ok(func(self._data))
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
                return ServiceResult.fail(str(e))
        return ServiceResult.fail(self._error or "No data")

    def and_then[V](self, func: Callable[[T], ServiceResult[V]]) -> ServiceResult[V]:
        """Chain operation if successful."""
        if self._success and self._data is not None:
            return func(self._data)
        return ServiceResult.fail(self._error or "No data")


# ==============================================================================
# EXPORTS - COMPREHENSIVE TYPE SYSTEM
# ==============================================================================

__all__ = [
    "URL",
    "AlertSeverity",
    "ApiKey",
    "BaseConfigDict",
    # Typed dicts
    "BaseEntityDict",
    "BatchSize",
    "CommandServiceProtocol",
    "ConfigMapping",
    "ConfigurableProtocol",
    "ConfigurationKey",
    "ConfigurationValue",
    "CpuPercent",
    "CreatedAt",
    "DatabaseName",
    "DatabaseURL",
    "DependencyConfiguratorProtocol",
    "DirPath",
    "DiskMB",
    "DurationSeconds",
    # Generic types
    "EntityCollection",
    # Entity types
    "EntityId",
    "EntityMixin",
    # Protocols
    "EntityProtocol",
    "EntityStatus",
    # Enums
    "Environment",
    "EnvironmentLiteral",
    "EnvironmentMixin",
    "FileName",
    "FilePath",
    "FileSize",
    "HandlerFunction",
    "IPAddress",
    "IdentifierMixin",
    "Json",
    "JsonDict",
    "JsonList",
    "JsonSchema",
    "LogLevel",
    "LogLevelLiteral",
    "MemoryMB",
    "MergeableProtocol",
    "MetricType",
    # Base types
    "NonEmptyStr",
    "NonNegativeFloat",
    "NonNegativeInt",
    # Oracle database types
    "OracleArraySize",
    "OracleFetchSize",
    "OracleHost",
    "OraclePassword",
    "OraclePort",
    "OracleQueryTimeout",
    "OracleSID",
    "OracleSchema",
    "OracleServiceName",
    "OracleUsername",
    "OracleWMSAuthMethod",
    "OracleWMSEntityType",
    "OracleWMSFilterOperator",
    "OracleWMSPageMode",
    "OracleWMSWriteMode",
    "Password",
    "PipelineDict",
    "PipelineId",
    "PluginDict",
    "PluginId",
    "PluginType",
    "Port",
    "PositiveFloat",
    "PositiveInt",
    "ProjectMixin",
    "ProjectName",
    "ProjectSettingsProtocol",
    "QueryServiceProtocol",
    "RedisURL",
    "ResultStatus",
    "RetryCount",
    "RetryDelay",
    # Result pattern
    "ServiceResult",
    "SessionId",
    # Singer protocol types
    "SingerBatchSize",
    "SingerBookmark",
    "SingerCatalog",
    "SingerMaxRecords",
    "SingerParallelStreams",
    "SingerRecordCount",
    "SingerReplicationMethod",
    "SingerSchemaName",
    "SingerState",
    "SingerStateInterval",
    "SingerStreamName",
    "StatusMixin",
    "TimeoutSeconds",
    "TimestampISO",
    # Mixins
    "TimestampMixin",
    "Token",
    "TraceStatus",
    "TransactionId",
    "UpdatedAt",
    "UserId",
    "Username",
    "ValidationServiceProtocol",
    "Version",
    # WMS types
    "WMSCompanyCode",
    "WMSConnectionInfo",
    "WMSEntityInfo",
    "WMSEntityName",
    "WMSFacilityCode",
    "WMSFieldMapping",
    "WMSFieldName",
    "WMSItemID",
    "WMSLocationID",
    "WMSOrderNumber",
]
