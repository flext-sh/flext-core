"""Type aliases and generics for the FLEXT ecosystem.

Centralizes TypeVar declarations and type aliases for CQRS messages, handlers,
utilities, logging, and validation across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from re import Pattern
from typing import (
    Annotated,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.constants import FlextConstants

# LaxStr compatibility for external integrations (LDAP, etc.)

# ============================================================================
# Module-Level TypeVars - Used in flext-core src/
# ============================================================================
# All TypeVars are defined at module level for clarity and accessibility.
# Only TypeVars actually used in flext-core src/ are declared here.
# Note: ParamSpec cannot be used in type aliases within nested classes.

# ============================================================================
# Core Generic TypeVars
# ============================================================================
# Fundamental type variables used throughout flext-core.
T = TypeVar("T")
"""Generic type variable - most commonly used type parameter.
Used in: decorators, container, mixins, pagination, protocols."""
T_co = TypeVar("T_co", covariant=True)
"""Covariant generic type variable - for read-only types.
Used in: protocols, result."""
T_contra = TypeVar("T_contra", contravariant=True)
"""Contravariant generic type variable - for write-only types.
Used in: protocols (for future extensions)."""
E = TypeVar("E")
"""Element type - for collections and sequences.
Used in: collection, enum, args utilities."""
U = TypeVar("U")
"""Utility type - for utility functions and helpers.
Used in: result (for map/flat_map operations)."""
R = TypeVar("R")
"""Return type - for function return values and decorators.
Used in: decorators, args utilities."""

# ============================================================================
# ParamSpec
# ============================================================================
P = ParamSpec("P")
"""ParamSpec for decorator patterns and variadic function signatures.
Used in: decorators, args utilities."""

# ============================================================================
# Handler TypeVars
# ============================================================================
# Type variables for CQRS handlers and message processing.
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
"""Contravariant message type - for message objects.
Used in: handlers (CQRS message processing)."""
ResultT = TypeVar("ResultT")
"""Result type - generic result type variable.
Used in: handlers (handler return types)."""

# ============================================================================
# Config/Model TypeVars
# ============================================================================
# Type variables for configuration and model types.
T_Model = TypeVar("T_Model", bound=BaseModel)
"""Model type - for Pydantic model types (bound to BaseModel).
Used in: configuration utilities."""
T_Namespace = TypeVar("T_Namespace")
"""Namespace type - for namespace objects.
Used in: config (namespace configuration)."""
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
"""Settings type - for Pydantic settings types (bound to BaseSettings).
Used in: config (settings configuration)."""


# ============================================================================
# Module-Level Recursive Types (PEP 695 required for Pydantic recursive compat)
# ============================================================================

type GeneralValueType = (
    str
    | int
    | float
    | bool
    | datetime
    | None
    | BaseModel
    | Path
    | Sequence[GeneralValueType]
    | Mapping[str, GeneralValueType]
)

type JsonPrimitive = str | int | float | bool | None

type JsonValue = (
    JsonPrimitive | Sequence[GeneralValueType] | Mapping[str, GeneralValueType]
)


class FlextTypes:
    """Type system foundation for FLEXT ecosystem - Complex Type Aliases Only.

    This class serves as the centralized type system for all FLEXT projects.
    All complex types (type aliases) are organized within this class.
    TypeVars are defined at module level (see above).

    All aliases use ``TypeAlias`` annotation (not PEP 695 ``type`` statement)
    so that basedpyright resolves them as proper class attributes at workspace scope.
    """

    # =====================================================================
    # COMPLEX TYPE ALIASES (TypeAlias for basedpyright compatibility)
    # =====================================================================

    LaxStr: TypeAlias = str | bytes | bytearray
    """LaxStr compatibility for ldap3 integration."""

    # =========================================================================
    # ALIGNED TYPE HIERARCHY (Pydantic-safe, no 'object' types)
    # =========================================================================

    # Tier 1: Scalar primitives (immutable, JSON-safe)
    ScalarValue: TypeAlias = str | int | float | bool | datetime | None

    # Tier 2: Pydantic-safe metadata values
    MetadataScalarValue: TypeAlias = str | int | float | bool | None
    MetadataListValue: TypeAlias = list[str | int | float | bool | None]

    # Tier 2.5: Pydantic-safe config types for Field() annotations
    PydanticConfigValue: TypeAlias = (
        str | int | float | bool | None | list[str | int | float | bool | None]
    )

    # Tier 3: General value types (superset including BaseModel, Path, datetime)
    GeneralScalarValue: TypeAlias = str | int | float | bool | datetime | None
    GeneralListValue: TypeAlias = list[str | int | float | bool | datetime | None]

    # =========================================================================
    # GeneralValueType - Re-exported from module level (recursive)
    # =========================================================================
    GeneralValueType: TypeAlias = GeneralValueType

    # RegisterableService - Type for services registerable in FlextContainer
    RegisterableService: TypeAlias = GeneralValueType | object

    # RegistrablePlugin - Type for plugins registerable in FlextRegistry
    RegistrablePlugin: TypeAlias = GeneralValueType | Callable[..., GeneralValueType]

    # Constant value type - all possible constant types in FlextConstants
    ConstantValue: TypeAlias = (
        str
        | int
        | float
        | bool
        | ConfigDict
        | SettingsConfigDict
        | frozenset[str]
        | tuple[str, ...]
        | Mapping[str, str | int]
        | StrEnum
        | type[StrEnum]
        | Pattern[str]
        | type
    )

    # Object list type - sequence of general value types for batch operations
    ObjectList: TypeAlias = Sequence[GeneralValueType]

    # File content type supporting all serializable formats
    FileContent: TypeAlias = str | bytes | BaseModel | Sequence[Sequence[str]]

    # Sortable object type
    SortableObjectType: TypeAlias = str | int | float

    # =========================================================================
    # Conversion Mode Types
    # =========================================================================
    ConversionMode: TypeAlias = Literal[
        FlextConstants.Utilities.ConversionMode.TO_STR,
        FlextConstants.Utilities.ConversionMode.TO_STR_LIST,
        FlextConstants.Utilities.ConversionMode.NORMALIZE,
        FlextConstants.Utilities.ConversionMode.JOIN,
    ]

    # MetadataAttributeValue - ALIGNED with GeneralValueType primitive types
    MetadataAttributeValue: TypeAlias = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | list[str | int | float | bool | datetime | None]
    )

    # Flexible value type for protocol methods
    FlexibleValue: TypeAlias = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | Sequence[str | int | float | bool | datetime | None]
    )

    # Type hint specifier - used for type introspection
    TypeHintSpecifier: TypeAlias = (
        type | str | Callable[[GeneralValueType], GeneralValueType]
    )

    # Generic type argument
    GenericTypeArgument: TypeAlias = str | type[GeneralValueType]

    # Message type specifier
    MessageTypeSpecifier: TypeAlias = str | type

    # Type origin specifier
    TypeOriginSpecifier: TypeAlias = (
        str | type[GeneralValueType] | Callable[[GeneralValueType], GeneralValueType]
    )

    # JSON types - re-exported from module level
    JsonPrimitive: TypeAlias = JsonPrimitive
    JsonValue: TypeAlias = JsonValue

    # Single consolidated callable type for handlers and validators
    HandlerCallable: TypeAlias = Callable[[GeneralValueType], GeneralValueType]

    # Acceptable message types for handlers
    AcceptableMessageType: TypeAlias = (
        GeneralValueType | Sequence[GeneralValueType] | ScalarValue
    )

    # Conditional execution callable types
    ConditionCallable: TypeAlias = Callable[[GeneralValueType], bool]

    # Handler type union
    HandlerType: TypeAlias = HandlerCallable | BaseModel

    # =========================================================================
    # Configuration mapping types
    # =========================================================================
    ConfigurationMapping: TypeAlias = Mapping[str, GeneralValueType]
    ConfigurationDict: TypeAlias = dict[str, GeneralValueType]
    PydanticConfigDict: TypeAlias = dict[str, PydanticConfigValue]

    class Dispatcher:
        """Dispatcher configuration types namespace."""

    IncEx: TypeAlias = set[str] | dict[str, set[str] | bool]

    DecoratorType: TypeAlias = Callable[[HandlerCallable], HandlerCallable]

    ServiceInstanceType: TypeAlias = GeneralValueType
    FactoryCallable: TypeAlias = Callable[[], GeneralValueType]
    ResourceCallable: TypeAlias = Callable[[], GeneralValueType]

    FactoryRegistrationCallable: TypeAlias = Callable[
        [], ScalarValue | Sequence[ScalarValue]
    ]
    FactoryMapping: TypeAlias = Mapping[str, FactoryRegistrationCallable]
    ResourceMapping: TypeAlias = Mapping[str, ResourceCallable]

    # =========================================================================
    # Validation mapping types (used in _models/validation.py)
    # =========================================================================
    FieldValidatorMapping: TypeAlias = Mapping[str, Callable[[GeneralValueType], bool]]
    ConsistencyRuleMapping: TypeAlias = Mapping[str, Callable[[GeneralValueType], bool]]
    EventValidatorMapping: TypeAlias = Mapping[str, Callable[[GeneralValueType], bool]]

    # Error/Exception types (used in exceptions.py)
    ErrorTypeMapping: TypeAlias = dict[str, int | str | dict[str, int]]
    ExceptionKwargsType: TypeAlias = str | int | float | bool | datetime | None

    # General nested dict type (used in guards.py)
    GeneralNestedDict: TypeAlias = dict[str, GeneralValueType]

    # ServiceMapping for service registration
    ServiceMapping: TypeAlias = Mapping[str, GeneralValueType]

    # =====================================================================
    # Pydantic Models
    # =====================================================================
    class BatchResultDict(BaseModel):
        """Result payload model for batch operation outputs."""

        model_config = ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        results: list[GeneralValueType] = Field(default_factory=list)
        errors: list[tuple[int, str]] = Field(default_factory=list)
        total: int = Field(default=0)
        success_count: int = Field(default=0)
        error_count: int = Field(default=0)

        def __getitem__(
            self,
            key: str,
        ) -> list[GeneralValueType] | list[tuple[int, str]] | int:
            """Provide mapping-style compatibility for legacy batch result access."""
            if key == "results":
                return self.results
            if key == "errors":
                return self.errors
            if key == "total":
                return self.total
            if key == "success_count":
                return self.success_count
            if key == "error_count":
                return self.error_count
            msg = f"Invalid key: {key}"
            raise KeyError(msg)

    # =====================================================================
    # VALIDATION TYPES (Annotated with Pydantic constraints)
    # =====================================================================

    class Validation:
        """Validation type aliases with Pydantic constraints."""

        PortNumber: TypeAlias = Annotated[int, Field(ge=1, le=65535)]
        """Port number type alias (1-65535)."""

        PositiveTimeout: TypeAlias = Annotated[float, Field(gt=0.0, le=300.0)]
        """Positive timeout in seconds (0-300)."""

        RetryCount: TypeAlias = Annotated[int, Field(ge=0, le=10)]
        """Retry count (0-10)."""

        WorkerCount: TypeAlias = Annotated[int, Field(ge=1, le=100)]
        """Worker count (1-100)."""


t_core = FlextTypes
t = FlextTypes


__all__ = [
    "FlextTypes",
    "GeneralValueType",
    "JsonPrimitive",
    "JsonValue",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "T",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "U",
    "t",
    "t_core",
]
