"""FLEXT Type System - Pure type definitions without external dependencies.

This module provides ONLY basic type definitions using Python built-in types.
NO imports from other flext modules to avoid circular dependencies.
All types are based on built-in Python types: dict, list, str, int, etc.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from enum import StrEnum
from typing import Literal, ParamSpec, TypeVar


class FlextTypes:
    """Hierarchical type system for FLEXT types."""

    # =========================================================================
    # CORE TYPES - Fundamental building blocks
    # =========================================================================

    class Core:
        """Core fundamental types used across flext-core modules."""

        # Type alias for Optional to avoid Union syntax
        type Optional[T] = T | None

        # Basic functional types - use typing imports directly in code

        # Basic collection types (heavily used in handlers, commands, models)
        type Dict = dict[str, object]
        type List = list[object]
        type StringList = list[str]
        type Headers = dict[str, str]
        type CounterDict = dict[str, int]
        type IntList = list[int]
        type FloatList = list[float]
        type BoolList = list[bool]

        # Advanced collection types
        type NestedDict = dict[str, dict[str, object]]
        type PathDict = dict[str, str]
        type ConfigDict = dict[str, str | int | float | bool | None]
        type MetadataDict = dict[str, str]
        type ParameterDict = dict[str, object]
        type AttributeDict = dict[str, object]

        type ConfigValue = (
            str | int | float | bool | None | list[object] | dict[str, object]
        )
        type JsonData = (
            dict[str, object] | list[object] | str | int | float | bool | None
        )

        # JSON types (used in handlers, commands, result serialization)
        # JsonValue union type for flexible JSON data handling
        # The nested list[str | int | float | bool | None | list[object] | dict[str, object]] is unreadable
        type JsonValue = (
            str
            | int
            | float
            | bool
            | None
            | list[str | int | float | bool | None | list[object] | dict[str, object]]
            | dict[
                str,
                str | int | float | bool | None | list[object] | dict[str, object],
            ]
        )
        type JsonObject = dict[str, JsonValue]
        JsonDict = dict[str, JsonValue]

        # Value type - Union type for domain operations
        type Value = str | int | float | bool | None | object

        # Operation callable - specific operation type
        OperationCallable = Callable[[object], object]

        # Serialization
        # Serialization function type
        Serializer = Callable[[object], dict[str, object]]

    # =========================================================================
    # DOMAIN TYPES - Domain-Driven Design patterns
    # =========================================================================

    class Domain:
        """Domain types for DDD patterns."""

        # Entity types
        type EntityId = str
        type Entity = object

        # Value object types
        type ValueObject = object

        # Aggregate types
        type AggregateRoot = object

        # Domain event types
        type DomainEvent = dict[str, object]

    # =========================================================================
    # SERVICE TYPES - Service layer patterns - with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Service:
        """Service layer types."""

        # Container registry types
        type ServiceDict = dict[str, object]  # Services registry mapping
        type FactoryDict = dict[str, Callable[[], object]]  # Factory registry
        type ServiceListDict = dict[
            str,
            Literal["instance", "factory"],
        ]  # Service type mapping

    # =========================================================================
    # SERIALIZATION TYPES - Data serialization patterns
    # =========================================================================

    class Serialization:
        """Basic serialization types."""

        # Serialized forms
        type JsonString = str
        type XmlString = str
        type CsvString = str
        type YamlString = str

        # Binary forms
        type Bytes = bytes
        type ByteArray = bytearray
        type Base64String = str

    # =========================================================================
    # IDENTIFIERS - Basic ID and key types
    # =========================================================================

    class Identifiers:
        """Basic identifier types."""

        # Primary identifiers
        type Id = str
        type Key = str
        type Name = str
        type Code = str
        type Tag = str

        # Compound identifiers
        type Path = str
        type Url = str
        type Uri = str
        type Urn = str

    # =========================================================================
    # CONFIG TYPES - Configuration and settings
    # =========================================================================

    class Config:
        """Configuration types."""

        # Primary configuration types (heavily used across flext-core)
        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type ConfigDict = dict[str, ConfigValue]  # Core config dictionary type

        # Environment type (used in config, handlers, commands)
        type Environment = Literal[
            "development",
            "production",
            "staging",
            "test",
            "local",
        ]

        # Logging levels (used in loggings, handlers, observability)
        type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Config serialization
        type ConfigSerializer = Callable[[ConfigDict], str]

    # =========================================================================
    # MODELS TYPES - Models system types - with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Models:
        """Model system types."""

        # =====================================================================
        # HEAVILY USED MODELS TYPES - Core model patterns (21 total usages)
        # =====================================================================

        # Primary configuration dictionary type (15 usages) - Enhanced with Python 3.13+ union syntax
        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type ConfigDict = dict[str, ConfigValue]  # Primary models config type

        # Basic configuration types
        type PerformanceConfig = dict[str, ConfigValue]  # Performance config input

    # =========================================================================
    # FUNCTION TYPES - Function patterns actually used
    # =========================================================================

    class Function:
        """Function types used across modules."""

        type Validator = Callable[[object], bool]
        type Processor = Callable[[object], object]

    # =========================================================================
    # VALIDATION TYPES - Validation patterns used across modules
    # =========================================================================

    class Validation:
        """Validation types used in guards, validations, commands."""

        type Validator = Callable[[object], bool]
        type ValidationRule = Callable[[object], bool]
        type BusinessRule = Callable[[object], bool]

    # =========================================================================
    # PROJECT TYPES - Project and workspace management
    # =========================================================================

    class Project:
        """Project and workspace types consolidated from across ecosystem."""

        # Enum is now imported at module level

        class ProjectType(StrEnum):
            """Project type enumeration."""

            PYTHON = "python"
            JAVASCRIPT = "javascript"
            GO = "go"
            RUST = "rust"
            DOCUMENTATION = "documentation"
            MIXED = "mixed"

        class WorkspaceStatus(StrEnum):
            """Workspace status enumeration."""

            INITIALIZING = "initializing"
            READY = "ready"
            ERROR = "error"

        # Project-related type aliases
        type ProjectPath = str
        type ProjectName = str
        type WorkspacePath = str
        type WorkspaceName = str

    # =========================================================================
    # LEGACY COMPATIBILITY - Types that modules still reference
    # =========================================================================
    # LEGACY HELL: Multiple compatibility classes with duplicate ConfigValue definitions
    # This creates 4 different ConfigDict types that are all the same - architectural mess!

    class Aggregates:
        """Aggregate types for core.py compatibility."""

        # DUPLICATE: ConfigValue defined again - same as Core.ConfigValue, Models.ConfigValue, Commands.ConfigValue

        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type ConfigDict = dict[str, ConfigValue]
        type AggregatesConfigDict = dict[
            str, ConfigValue
        ]  # DUPLICATE: Same as ConfigDict
        type AggregatesConfig = dict[str, ConfigValue]  # DUPLICATE: Same as ConfigDict
        type SystemConfig = dict[str, ConfigValue]  # DUPLICATE: Same as ConfigDict
        type PerformanceConfig = dict[str, ConfigValue]  # DUPLICATE: Same as ConfigDict
        type PerformanceLevel = Literal["low", "balanced", "high", "extreme"]

    class Commands:
        """CQRS Command and Query types for Flext CQRS components."""

        # Handler function types (avoiding FlextResult to prevent circular imports)
        type CommandHandlerFunc[T, R] = Callable[[T], R]
        type QueryHandlerFunc[T, R] = Callable[[T], R]

        # Bus configuration types
        type BusConfig = dict[
            str, str | int | float | bool | list[object] | dict[str, object]
        ]
        type MiddlewareConfig = dict[
            str, str | int | float | bool | list[object] | dict[str, object]
        ]
        type HandlerRegistry = dict[str, object]

        # Command/Query metadata types
        type CommandMetadata = dict[str, str | int | float | bool | None]
        type QueryMetadata = dict[str, str | int | float | bool | None]

    class Container:
        """Container types for core.py compatibility."""

        type ServiceKey = str
        type ServiceInstance = object
        type ServiceRegistration = object  # FlextResult[None]
        type ServiceRetrieval = object  # FlextResult[object]
        type FactoryFunction = Callable[[], object]
        type FactoryRegistration = object  # FlextResult[None]

    class Handler:
        """Handler types for validations.py compatibility."""

        type Context = dict[str, object]
        type HandlerMetadata = dict[str, object]

    # =========================================================================
    # RESULT TYPES - Result pattern types for test compatibility
    # =========================================================================

    class Result:
        """Result types for test compatibility."""

        type ResultData = object
        type ResultError = str | None
        type ResultValue = object


# =========================================================================
# BACKWARD COMPATIBILITY EXPORTS - Maintain ecosystem compatibility
# =========================================================================

# Export TypeVars for backward compatibility (32+ projects depend on these)
# Note: Direct assignment to maintain proper TypeVar types for MyPy
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
R = TypeVar("R")
E = TypeVar("E")
F = TypeVar("F")
K = TypeVar("K")
P = ParamSpec("P")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

MessageT = TypeVar("MessageT")
ResultT = TypeVar("ResultT")
T_Service = TypeVar("T_Service")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
TAggregate = TypeVar("TAggregate")
TCommand = TypeVar("TCommand")
TDomain = TypeVar("TDomain")
TDomainResult = TypeVar("TDomainResult")
TEntity = TypeVar("TEntity")
TEntry = TypeVar("TEntry")
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TItem = TypeVar("TItem")
TMessage = TypeVar("TMessage")
TOutput_co = TypeVar("TOutput_co", covariant=True)
TQuery = TypeVar("TQuery")
TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")
TResult = TypeVar("TResult")
TUtil = TypeVar("TUtil")
TValueObject = TypeVar("TValueObject")


# Export Optional type alias for backward compatibility
# Re-export the type alias from FlextTypes.Core to maintain backward compatibility
type Optional[T] = FlextTypes.Core.Optional[T]


__all__: list[str] = [
    "T1",
    "T2",
    "T3",
    "E",
    "F",
    "FlextTypes",
    "K",
    "Optional",
    "P",
    "R",
    "T",
    "TAggregate",
    "TCommand",
    "TDomain",
    "TDomainResult",
    "TEntity",
    "TEntry",
    "TInput_contra",
    "TItem",
    "TMessage",
    "TOutput_co",
    "TQuery",
    "TRequest",
    "TResponse",
    "TResult",
    "TUtil",
    "TValueObject",
    "T_Service",
    "T_co",
    "T_contra",
    "U",
    "V",
]
