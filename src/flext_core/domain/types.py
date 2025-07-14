"""Advanced typing system for FLEXT.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides advanced typing patterns, protocols, mixins and constants for
maximum code reduction.
"""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
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

from pydantic import Field
from pydantic import StringConstraints

from flext_core.domain.constants import ConfigDefaults
from flext_core.domain.constants import Environments
from flext_core.domain.constants import LogLevels
from flext_core.domain.constants import RegexPatterns

# ==============================================================================
# TYPE CONSTANTS - SINGLE SOURCE OF TRUTH
# ==============================================================================

# String constraints with validation - using aliases compatible with Pydantic
type ProjectName = Annotated[
    str,
    StringConstraints(
        min_length=2,
        max_length=ConfigDefaults.MAX_ENTITY_NAME_LENGTH,
        pattern=RegexPatterns.PROJECT_NAME,
    ),
    Field(description="Project name following naming conventions"),
]

type Version = Annotated[
    str,
    StringConstraints(pattern=RegexPatterns.SEMANTIC_VERSION),
    Field(description="Semantic version (e.g., 1.0.0, 1.0.0-alpha)"),
]

type EnvironmentLiteral = Literal["development", "staging", "production", "test"]

type LogLevelLiteral = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ID types with strong typing - using aliases compatible with Pydantic
EntityId = Annotated[UUID, Field(description="Unique entity identifier")]
UserId = Annotated[UUID, Field(description="User identifier")]
PipelineId = Annotated[UUID, Field(description="Pipeline identifier")]
PluginId = Annotated[UUID, Field(description="Plugin identifier")]

# Timestamp types
CreatedAt = Annotated[datetime, Field(description="Creation timestamp")]
UpdatedAt = Annotated[datetime, Field(description="Last update timestamp")]

# Configuration types
ConfigKey = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=ConfigDefaults.MAX_CONFIG_KEY_LENGTH,
        pattern=RegexPatterns.CONFIG_KEY,
    ),
    Field(description="Configuration key in snake_case"),
]

ConfigValue = str | int | float | bool | None

# ==============================================================================
# ADVANCED GENERIC TYPES
# ==============================================================================

# Modern Python 3.13 type variables
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
TId = TypeVar("TId", bound=UUID)
TEntity = TypeVar("TEntity", bound="EntityProtocol[UUID]")
TConfig = TypeVar("TConfig", bound="ConfigProtocol")
TSettings = TypeVar("TSettings", bound="SettingsProtocol")

# Generic collections with constraints
type EntityCollection[TEntity: "EntityProtocol[UUID]"] = Sequence[TEntity]
type ConfigMapping = Mapping[ConfigKey, ConfigValue]
type HandlerFunction[T, U] = Callable[[T], Awaitable[U]]

# ==============================================================================
# PROTOCOLS - INTERFACE DEFINITIONS
# ==============================================================================


@runtime_checkable
class EntityProtocol(Protocol[TId]):
    """Protocol for all entities - replaces base Entity class.

    Arguments:
        TId: The type of the entity ID.

    """

    id: TId
    created_at: CreatedAt
    updated_at: UpdatedAt | None

    def __eq__(self, other: object) -> bool:
        """Check if the entity is equal to another object.

        Returns:
            True if the entity is equal to the other object, False otherwise.

        """
        ...

    def __hash__(self) -> int:
        """Hash the entity.

        Returns:
            The hash of the entity.

        """
        ...


@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol for configuration objects."""

    def to_dict(self, *, exclude_unset: bool = True) -> dict[str, object]:
        """Convert to dictionary.

        Arguments:
            exclude_unset: Whether to exclude unset values.

        Returns:
            A dictionary containing the configuration.

        """
        ...

    def merge(self, other: Self) -> Self:
        """Merge with another config.

        Arguments:
            other: The other configuration to merge with.

        Returns:
            The merged configuration.

        """
        ...


@runtime_checkable
class SettingsProtocol(Protocol):
    """Protocol for settings objects."""

    project_name: ProjectName
    project_version: Version
    environment: EnvironmentLiteral
    debug: bool

    def configure_dependencies(self, container: object) -> None:
        """Configure dependencies.

        Arguments:
            container: The container to configure.

        """
        ...


@runtime_checkable
class ServiceProtocol(Protocol):
    """Protocol for service layer."""

    async def execute(self, input_data: object) -> object:
        """Execute service operation.

        Arguments:
            input_data: The input data for the service.

        Returns:
            The output data from the service.

        """
        ...


# ==============================================================================
# ENUMS - USING MODERN STRENUM
# ==============================================================================


class ResultStatus(StrEnum):
    """Result status enumeration using StrEnum."""

    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class EntityStatus(StrEnum):
    """Entity status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class PluginType(StrEnum):
    """Plugin type enumeration."""

    TAP = "tap"
    TARGET = "target"
    TRANSFORM = "transform"
    UTILITY = "utility"


class LogLevel(StrEnum):
    """Log level enumeration."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Environment(StrEnum):
    """Environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class MetricType(StrEnum):
    """Metric type enumeration for observability."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(StrEnum):
    """Alert severity enumeration for observability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TraceStatus(StrEnum):
    """Trace status enumeration for observability."""

    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Status(StrEnum):
    """General status enumeration - more generic than EntityStatus."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESOLVED = "resolved"


class ServiceResult[T]:
    """Type-safe result pattern with advanced error handling."""

    __slots__ = ("_data", "_error", "_status", "_success")

    def __init__(
        self,
        *,
        success: bool,
        data: T | None = None,
        error: str | None = None,
        status: ResultStatus = ResultStatus.SUCCESS,
    ) -> None:
        """Initialize the result.

        Arguments:
            success: Whether the result is successful.
            data: The data of the result.
            error: The error of the result.
            status: The status of the result.

        """
        self._success = success
        self._data = data
        self._error = error
        self._status = status if success else ResultStatus.ERROR

    @property
    def is_successful(self) -> bool:
        """Check if the result is successful.

        Returns:
            True if the result is successful, False otherwise.

        """
        return self._success

    @property
    def data(self) -> T | None:
        """Get the data of the result.

        Returns:
            The data of the result.

        """
        return self._data

    @property
    def error(self) -> str | None:
        """Get the error of the result.

        Returns:
            The error of the result.

        """
        return self._error

    @property
    def status(self) -> ResultStatus:
        """Get the status of the result.

        Returns:
            The status of the result.

        """
        return self._status

    # Compatibility aliases for existing code
    @property
    def error_message(self) -> str | None:
        """Get the error message of the result.

        Returns:
            The error message of the result.

        """
        return self._error

    @property
    def value(self) -> T | None:
        """Get the value of the result.

        Returns:
            The value of the result.

        """
        return self._data

    @property
    def is_success(self) -> bool:
        """Check if the result is successful.

        Returns:
            True if the result is successful, False otherwise.

        """
        return self._success

    @property
    def is_failure(self) -> bool:
        """Check if the result is a failure.

        Returns:
            True if the result is a failure, False otherwise.

        """
        return not self._success

    # Remove property to avoid conflict with class method
    # @property
    # def success(self) -> bool:
    #     """Check if result is successful (alias for is_success).:"""
    #     return self._success

    @classmethod
    def ok(cls, data: T) -> Self:
        """Create a successful result.

        Arguments:
            data: The data of the result.

        Returns:
            The successful result.

        """
        return cls(success=True, data=data, status=ResultStatus.SUCCESS)

    @classmethod
    def fail(cls, error: str) -> Self:
        """Create a failed result.

        Arguments:
            error: The error of the result.

        Returns:
            The failed result.

        """
        return cls(success=False, error=error, status=ResultStatus.ERROR)

    # Compatibility aliases for existing code - method variants
    @classmethod
    def success(cls, data: T) -> Self:
        """Create a successful result.

        Arguments:
            data: The data of the result.

        Returns:
            The successful result.

        """
        return cls.ok(data)

    @classmethod
    def failure(cls, error: str) -> Self:
        """Create a failed result.

        Arguments:
            error: The error of the result.

        Returns:
            The failed result.

        """
        return cls.fail(error)

    @classmethod
    def pending(cls) -> Self:
        """Create a pending result.

        Returns:
            The pending result.

        """
        return cls(success=False, status=ResultStatus.PENDING)

    def unwrap(self) -> T:
        """Unwrap the result.

        Returns:
            The data of the result.

        Raises:
            RuntimeError: If the result is not successful or has no data

        """
        if not self._success or self._data is None:
            msg = f"Cannot unwrap failed result: {self._error}"
            raise RuntimeError(msg)
        return self._data

    def unwrap_or(self, default: U) -> T | U:
        """Unwrap the result or return a default value.

        Arguments:
            default: The default value to return if the result is not successful.

        Returns:
            The data of the result or the default value.

        """
        return self._data if self._success and self._data is not None else default

    def map[V](self, func: Callable[[T], V]) -> ServiceResult[V]:
        """Map the result to a new result.

        Arguments:
            func: The function to map the result.

        Returns:
            The mapped result.

        """
        if self._success and self._data is not None:
            try:
                return ServiceResult.ok(func(self._data))
            except (ValueError, TypeError, AttributeError) as e:
                return ServiceResult.fail(str(e))

        return ServiceResult.fail(self._error or "No data")

    def and_then[V](self, func: Callable[[T], ServiceResult[V]]) -> ServiceResult[V]:
        """Chain another operation if this result is successful.

        Args:
            func: Function to apply to the data if successful

        Returns:
            New ServiceResult from the function or failure

        """
        if self._success and self._data is not None:
            return func(self._data)
        return ServiceResult.fail(self._error or "No data")


# ==============================================================================
# TYPED DICTIONARIES - STRUCTURED DATA
# ==============================================================================


class EntityDict(TypedDict):
    """Base entity dictionary."""

    id: EntityId
    created_at: CreatedAt
    updated_at: NotRequired[UpdatedAt]


class ConfigDict(TypedDict):
    """Configuration dictionary."""

    project_name: ProjectName
    project_version: Version
    environment: EnvironmentLiteral
    debug: bool


class PipelineDict(EntityDict):
    """Pipeline entity dictionary."""

    name: str
    description: str
    status: Literal["active", "inactive", "error"]


class PluginDict(EntityDict):
    """Plugin entity dictionary."""

    name: str
    type: Literal["tap", "target", "transform"]
    version: Version


# ==============================================================================
# CONSTANTS - CONFIGURATION VALUES
# ==============================================================================

# Use constants from constants.py - no duplication
FlextConstants = type(
    "FlextConstants",
    (),
    {
        # Framework metadata
        "FRAMEWORK_NAME": "FLEXT",
        "FRAMEWORK_VERSION": "0.7.0",
        "PYTHON_VERSION": "3.13",
        # Default configurations
        "DEFAULT_ENVIRONMENT": Environments.DEFAULT,
        "DEFAULT_LOG_LEVEL": LogLevels.DEFAULT,
        "DEFAULT_TIMEOUT": ConfigDefaults.DEFAULT_TIMEOUT,
        "DEFAULT_TIMEOUT_SECONDS": ConfigDefaults.DEFAULT_TIMEOUT,
        "DEFAULT_RETRY_COUNT": ConfigDefaults.DEFAULT_RETRY_COUNT,
        "DEFAULT_MAX_RETRIES": ConfigDefaults.DEFAULT_RETRY_COUNT,
        "DEFAULT_RETRY_DELAY": 1.0,
        "DEFAULT_POOL_SIZE": 10,
        "DEFAULT_REQUEST_TIMEOUT": 60.0,
        # Limits and constraints
        "MAX_ENTITY_NAME_LENGTH": ConfigDefaults.MAX_ENTITY_NAME_LENGTH,
        "MAX_CONFIG_KEY_LENGTH": ConfigDefaults.MAX_CONFIG_KEY_LENGTH,
        "MAX_ERROR_MESSAGE_LENGTH": ConfigDefaults.MAX_ERROR_MESSAGE_LENGTH,
        "DEFAULT_PAGE_SIZE": ConfigDefaults.DEFAULT_PAGE_SIZE,
        "MAX_PAGE_SIZE": ConfigDefaults.MAX_PAGE_SIZE,
        "MAX_TIMEOUT_SECONDS": 300.0,
        "MAX_POOL_SIZE": 100,
        "MAX_RETRIES": 5,
        "MAX_RETRY_DELAY": 60.0,
        # Environment variable prefixes
        "ENV_PREFIX": ConfigDefaults.ENV_PREFIX,
        "ENV_DELIMITER": ConfigDefaults.ENV_DELIMITER,
        # File extensions
        "CONFIG_FILE_EXTENSION": ConfigDefaults.CONFIG_FILE_EXTENSION,
        "LOG_FILE_EXTENSION": ConfigDefaults.LOG_FILE_EXTENSION,
        # Status constants
        "STATUS_ACTIVE": "active",
        "STATUS_INACTIVE": "inactive",
        "STATUS_ERROR": "error",
        "STATUS_PENDING": "pending",
    },
)


# ==============================================================================
# ANNOTATION HELPERS
# ==============================================================================


def validate_entity_id(v: object) -> UUID:
    """Validate entity ID.

    Arguments:
        v: The value to validate.

    Returns:
        The validated entity ID.

    Raises:
        ValueError: If the value is not a valid UUID

    """
    if isinstance(v, UUID):
        return v
    if isinstance(v, str):
        return UUID(v)
    msg = f"Invalid entity ID: {v}"
    raise ValueError(msg)


def validate_project_name(v: object) -> str:
    """Validate project name.

    Arguments:
        v: The value to validate.

    Returns:
        The validated project name.

    Raises:
        TypeError: If the value is not a string
        ValueError: If the project name format is invalid

    """
    if not isinstance(v, str):
        msg = "Project name must be a string"
        raise TypeError(msg)
    if len(v) < 2 or len(v) > 50:
        msg = "Project name must be 2-50 characters"
        raise ValueError(msg)
    if not v.replace("-", "").replace("_", "").isalnum():
        msg = "Project name must be alphanumeric with hyphens/underscores"
        raise ValueError(msg)
    return v


# Annotation factories
def entity_id_field(description: str) -> Any:
    """Create an entity ID field.

    Arguments:
        description: The description of the field.

    Returns:
        The entity ID field.

    """
    return Field(
        description=description,
        json_schema_extra={"format": "uuid"},
    )


def project_name_field(description: str) -> Any:
    """Create a project name field.

    Arguments:
        description: The description of the field.

    Returns:
        The project name field.

    """
    return Field(
        description=description,
        min_length=2,
        max_length=50,
        pattern=r"^[a-zA-Z][a-zA-Z0-9-_]*$",
    )


def version_field(description: str) -> Any:
    """Create a version field.

    Arguments:
        description: The description of the field.

    Returns:
        The version field.

    """
    return Field(
        description=description,
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$",
    )


# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Enumerations
    "AlertSeverity",
    # Type aliases
    "ConfigDict",
    "ConfigKey",
    "ConfigMapping",
    "ConfigProtocol",
    "ConfigValue",
    "CreatedAt",
    "EntityCollection",
    # Typed dictionaries
    "EntityDict",
    "EntityId",
    # Protocols
    "EntityProtocol",
    "EntityStatus",
    "Environment",
    "EnvironmentLiteral",
    # Constants
    "FlextConstants",
    "HandlerFunction",
    "LogLevel",
    "LogLevelLiteral",
    "MetricType",
    "PipelineDict",
    "PipelineId",
    "PluginDict",
    "PluginId",
    "PluginType",
    # Type aliases
    "ProjectName",
    "ResultStatus",
    # Result types
    "ServiceProtocol",
    "ServiceResult",
    "SettingsProtocol",
    "Status",
    "StrEnum",
    "TraceStatus",
    "UpdatedAt",
    "UserId",
    "Version",
    "entity_id_field",
    "project_name_field",
    # Annotation helpers
    "validate_entity_id",
    "validate_project_name",
    "version_field",
]
