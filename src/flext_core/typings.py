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
from typing import Literal, ParamSpec, TypeVar


class FlextTypes:
    """Hierarchical type system for FLEXT types."""

    # =========================================================================

    # =========================================================================

    class TypeVars:
        """Generic type variables for ecosystem-wide use."""

        # Specialized type variables
        TEntity = TypeVar("TEntity")  # Entity types
        TValueObject = TypeVar("TValueObject")  # Value object types
        TAggregate = TypeVar("TAggregate")  # Aggregate root types
        TMessage = TypeVar("TMessage")  # Message types for handlers
        TRequest = TypeVar("TRequest")  # Request types
        TResponse = TypeVar("TResponse")  # Response types
        TCommand = TypeVar("TCommand")  # Command types for CQRS
        TQuery = TypeVar("TQuery")  # Query types for CQRS
        TResult = TypeVar("TResult")  # Result types for handlers
        TEntry = TypeVar("TEntry")  # Entry types for schema processing

        # Primary type variables for generic programming (Python 3.13+ syntax)
        # T and P are defined at module level per user requirement
        U = TypeVar("U")  # Secondary generic type for FLEXT ecosystem
        V = TypeVar("V")  # Tertiary generic type
        K = TypeVar("K")  # Key type for mappings
        R = TypeVar("R")  # Return type for functions
        E = TypeVar("E", bound=Exception)  # Exception type
        F = TypeVar("F")  # Function/field type variable
        # P is defined at module level per user requirement

    # =========================================================================
    # CORE TYPES - Fundamental building blocks
    # =========================================================================

    class Core:
        """Core fundamental types used across flext-core modules."""

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

        # Operation callable
        type OperationCallable = Callable[[object], object]

        # Serialization
        type Serializer = Callable[[object], dict[str, object]]

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
    # SERVICE TYPES - Service layer patterns - REFACTORED with Pydantic 2.11+ & Python 3.13+
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
    # MODELS TYPES - Models system types - REFACTORED with Pydantic 2.11+ & Python 3.13+
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
    # LEGACY COMPATIBILITY - Types that modules still reference
    # =========================================================================

    class Aggregates:
        """Aggregate types for core.py compatibility."""

        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type ConfigDict = dict[str, ConfigValue]
        type AggregatesConfigDict = dict[str, ConfigValue]
        type AggregatesConfig = dict[str, ConfigValue]
        type SystemConfig = dict[str, ConfigValue]
        type PerformanceConfig = dict[str, ConfigValue]
        type PerformanceLevel = Literal["low", "balanced", "high", "extreme"]

    class Commands:
        """Command types for core.py compatibility."""

        type CommandsConfigDict = dict[
            str, str | int | float | bool | list[object] | dict[str, object]
        ]
        type CommandsConfig = dict[
            str, str | int | float | bool | list[object] | dict[str, object]
        ]

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


# Generic type variables

T = TypeVar("T")  # Generic type
U = TypeVar("U")  # Generic type
V = TypeVar("V")  # Generic type
R = TypeVar("R")  # Generic result type
E = TypeVar("E", bound=Exception)  # Exception type
F = TypeVar("F")  # Generic function type
K = TypeVar("K")  # Generic key type
P = ParamSpec("P")  # Parameter specification


# Convenience top-level aliases for common types used in tests/examples
type ConfigDict = FlextTypes.Config.ConfigDict

__all__: list[str] = [
    "ConfigDict",
    "E",
    "F",
    "FlextTypes",
    "K",
    "P",
    "R",
    "T",
    "U",
    "V",
]
