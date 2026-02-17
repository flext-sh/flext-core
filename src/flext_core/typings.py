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


class FlextTypes:
    """Type system foundation for FLEXT ecosystem - Complex Type Aliases Only.

    This class serves as the centralized type system for all FLEXT projects.
    All complex types (type aliases, TypedDict classes) are organized in nested
    namespaces within this class. TypeVars are defined at module level (see above).

    LaxStr compatibility for ldap3 integration
    """

    type LaxStr = str | bytes | bytearray

    # =====================================================================
    # COMPLEX TYPE ALIASES (Python 3.13+ PEP 695 strict)
    # =====================================================================
    # Complex types that require composition or domain-specific logic

    # =========================================================================
    # ALIGNED TYPE HIERARCHY (Pydantic-safe, no 'object' types)
    # =========================================================================
    # Business Rule: All value types are JSON-serializable and Pydantic-compatible.
    # Types are aligned from MetadataAttributeValue up to GeneralValueType.

    # Tier 1: Scalar primitives (immutable, JSON-safe)
    type ScalarValue = str | int | float | bool | datetime | None

    # Tier 2: Pydantic-safe metadata values (used in Metadata.attributes)
    # Must match _ConfigValue in _models/base.py for Pydantic schema compatibility
    type MetadataScalarValue = str | int | float | bool | None
    type MetadataListValue = list[str | int | float | bool | None]
    type MetadataNestedDict = dict[
        str,
        str | int | float | bool | list[str | int | float | bool | None] | None,
    ]

    # Tier 2.5: Pydantic-safe config types for Field() annotations
    # These types avoid recursion issues in Pydantic schema generation
    type PydanticConfigValue = (
        str
        | int
        | float
        | bool
        | None
        | list[str | int | float | bool | None]
        | dict[
            str,
            str | int | float | bool | list[str | int | float | bool | None] | None,
        ]
    )
    type PydanticConfigDict = dict[str, FlextTypes.PydanticConfigValue]

    # Tier 3: General value types (superset including BaseModel, Path, datetime)
    # Used throughout the codebase for flexible value handling
    type GeneralScalarValue = str | int | float | bool | datetime | None
    type GeneralListValue = list[str | int | float | bool | datetime | None]
    type GeneralNestedDict = dict[
        str,
        str
        | int
        | float
        | bool
        | datetime
        | None
        | list[str | int | float | bool | datetime | None],
    ]

    # =========================================================================
    # GeneralValueType - RECURSIVE type for JSON-like values (PEP 695)
    # =========================================================================
    # Python 3.13+ recursive type using forward reference
    # Allows Mapping[str, GeneralValueType] to be a GeneralValueType
    type GeneralValueType = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | BaseModel
        | Path
        | Sequence[FlextTypes.GeneralValueType]  # Recursive: lists of values
        | Mapping[str, FlextTypes.GeneralValueType]  # Recursive: dicts of values
    )

    # RegisterableService - Type for services registerable in FlextContainer
    # Extends GeneralValueType to include protocol types (Config, Ctx, Logger, etc.)
    # which are plain classes (not BaseModel) implementing protocol interfaces
    RegisterableService: TypeAlias = GeneralValueType | object

    # RegistrablePlugin - Type for plugins registerable in FlextRegistry
    # Includes callables (factories, handlers) and service instances (servers, quirks)
    # Used for class-level plugin storage where arbitrary objects are registered
    # Callables return GeneralValueType or can be service instances themselves
    RegistrablePlugin: TypeAlias = GeneralValueType | Callable[..., GeneralValueType]

    # Constant value type - all possible constant types in FlextConstants
    # Used for type-safe constant access via __getitem__ method
    # Includes all types that can be stored as constants: primitives, collections,
    # Pydantic ConfigDict, SettingsConfigDict, and StrEnum types
    type ConstantValue = (
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
    # Reuses "FlextTypes.GeneralValueType" (forward reference managed by __future__ annotations)
    type ObjectList = Sequence[FlextTypes.GeneralValueType]

    # File content type supporting all serializable formats including Pydantic models
    # Used for file operations that can handle different content types
    # Reuses "FlextTypes.GeneralValueType" for configuration mappings
    type FileContent = (
        str
        | bytes
        | Mapping[str, FlextTypes.GeneralValueType]
        | Sequence[Sequence[str]]
        | BaseModel
    )

    # Sortable object type - types that can be sorted (str, int, float, Mapping)
    # Note: Uses ScalarValue but excludes bool/None for sorting compatibility
    # PEP 695 recursive types work with __future__ annotations
    # Use string forward reference for recursive types to avoid pyrefly errors
    type SortableObjectType = (
        str | int | float | Mapping[str, FlextTypes.SortableObjectType]
    )

    # =========================================================================
    # Conversion Mode Types - Used in FlextUtilitiesConversion.conversion()
    # =========================================================================
    # Literal types for conversion function mode parameter overloads
    # Values from FlextConstants.Utilities.ConversionMode StrEnum
    type ConversionModeToStr = Literal[FlextConstants.Utilities.ConversionMode.TO_STR]
    type ConversionModeToStrList = Literal[
        FlextConstants.Utilities.ConversionMode.TO_STR_LIST
    ]
    type ConversionModeNormalize = Literal[
        FlextConstants.Utilities.ConversionMode.NORMALIZE
    ]
    type ConversionModeJoin = Literal[FlextConstants.Utilities.ConversionMode.JOIN]
    type ConversionMode = (
        FlextTypes.ConversionModeToStr
        | FlextTypes.ConversionModeToStrList
        | FlextTypes.ConversionModeNormalize
        | FlextTypes.ConversionModeJoin
    )

    # MetadataAttributeValue - ALIGNED with GeneralValueType primitive types
    # Includes datetime for proper subtyping (MetadataAttributeValue <: GeneralValueType)
    # Excludes: BaseModel, Path, Callable (those are GeneralValueType extensions)
    MetadataAttributeValue = (
        str
        | int
        | float
        | bool
        | datetime  # CRITICAL: Must include datetime for subtype compatibility
        | None
        | list[str | int | float | bool | datetime | None]
        | dict[
            str,
            str
            | int
            | float
            | bool
            | datetime
            | None
            | list[str | int | float | bool | datetime | None],
        ]
    )

    # Generic metadata dictionary type - read-only interface for metadata containers
    type Metadata = Mapping[str, FlextTypes.MetadataAttributeValue]
    """Structural type for metadata dictionaries using Mapping (read-only interface).

    Represents flexible metadata containers without importing models.
    Used in protocol definitions where metadata must be stored or passed.
    This is a generic type available to all FLEXT projects.
    """

    # Flexible value type for protocol methods - contains scalars,
    # sequences, or mappings
    type FlexibleValue = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | Sequence[str | int | float | bool | datetime | None]
        | Mapping[str, str | int | float | bool | datetime | None]
    )

    # Mapping of string keys to flexible values
    type FlexibleMapping = Mapping[str, FlextTypes.FlexibleValue]

    # NOTE: ParameterValueType was removed - use t.FlextTypes.GeneralValueType directly
    # No aliases for convenience - use types directly per FLEXT standards

    # Type hint specifier - used for type introspection
    # Can be any type hint: type, type alias, generic, or string
    # Note: For runtime type introspection, we accept type, str,
    # or callable types
    # Note: Removed Callable[..., T] to avoid variadic callables - use specific
    # callable types instead
    type TypeHintSpecifier = (
        type
        | str
        | Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType]
    )

    # Generic type argument - used for extracting generic type arguments
    # Can be a string type name or a type class representing "FlextTypes.GeneralValueType"
    # Reuses "FlextTypes.GeneralValueType" from parent t class (forward reference)
    type GenericTypeArgument = str | type[FlextTypes.GeneralValueType]

    # Message type specifier - used for handler type checking
    # Can be a string type name or any type class
    # Using type (not type[GeneralValueType]) to accept any runtime type hints
    type MessageTypeSpecifier = str | type

    # Type origin specifier - used for generic type origin checking
    # Can be a string type name, type class, or callable with __origin__ attribute
    # Reuses "FlextTypes.GeneralValueType" from parent t class (forward reference)
    # Note: Removed Callable[..., T] to avoid variadic callables - use specific callable types instead
    type TypeOriginSpecifier = (
        str
        | type[FlextTypes.GeneralValueType]
        | Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType]
    )

    # Primitive JSON types
    type JsonPrimitive = str | int | float | bool | None

    # Complex JSON types using recursive "FlextTypes.GeneralValueType"
    # Reuses "FlextTypes.GeneralValueType" from parent t class (no duplication)
    type JsonValue = (
        FlextTypes.JsonPrimitive
        | Sequence[FlextTypes.GeneralValueType]
        | Mapping[str, FlextTypes.GeneralValueType]
    )
    type JsonList = Sequence[FlextTypes.JsonValue]
    type JsonDict = Mapping[str, FlextTypes.GeneralValueType]

    # Single consolidated callable type for handlers and validators
    # Reuses "FlextTypes.GeneralValueType" from outer t class
    type HandlerCallable = Callable[
        [FlextTypes.GeneralValueType],
        FlextTypes.GeneralValueType,
    ]
    # Middleware uses same callable signature as handlers
    # Use HandlerCallable directly - no aliases per FLEXT standards

    # Handler type for registry - union of all possible handler types
    # This includes callables, objects with handle() method, and handler instances
    # Note: Handler protocol is defined in p.Handler for proper protocol organization

    # Middleware configuration type
    type MiddlewareConfig = Mapping[str, FlextTypes.GeneralValueType]

    # Acceptable message types for handlers - union of common message types
    # Reuses "FlextTypes.GeneralValueType" and ScalarValue from parent t class
    type AcceptableMessageType = (
        FlextTypes.GeneralValueType
        | Mapping[str, FlextTypes.GeneralValueType]
        | Sequence[FlextTypes.GeneralValueType]
        | FlextTypes.ScalarValue
    )

    # Conditional execution callable types (PEP 695)
    # Reuses "FlextTypes.GeneralValueType" from parent t class (forward reference)
    type ConditionCallable = Callable[[FlextTypes.GeneralValueType], bool]

    # Variadic callable protocol is defined in p.VariadicCallable
    # Use that instead of redeclaring here

    # Handler type union - all possible handler representations
    # This is used for handler registries where handlers can be callables,
    # handler instances, or configuration dicts
    # Note: Handler and VariadicCallable protocols are defined in p
    # but cannot be imported here due to circular dependency. For type checking,
    # we use HandlerCallable which covers most use cases.
    # Reuses "FlextTypes.GeneralValueType" from parent t class (no duplication)
    # Note: Removed Callable[..., T] to avoid variadic callables - HandlerCallable covers variadic cases
    type HandlerType = (
        FlextTypes.HandlerCallable
        | Mapping[str, FlextTypes.GeneralValueType]  # Configuration dict
    )

    class Dispatcher:
        """Dispatcher configuration types namespace."""

        # DispatcherConfig moved to Pydantic model in _models/settings.py
        # Use type alias for backward compatibility
        # from flext_core._models.settings import FlextModelsConfig
        # DispatcherConfig = FlextModelsConfig.DispatcherConfig

    # =====================================================================
    # MAPPING TYPE ALIASES (Python 3.13+ PEP 695)
    # =====================================================================
    # Using PEP 695 type keyword for better type checking and IDE support
    type ServiceMetadataMapping = Mapping[str, FlextTypes.GeneralValueType]
    """Mapping for service metadata (attribute names to values)."""

    type FieldMetadataMapping = Mapping[str, FlextTypes.GeneralValueType]
    """Mapping for field metadata (field names to metadata objects)."""

    type SummaryDataMapping = Mapping[str, int | float | str]
    """Mapping for summary data (category names to summary values)."""

    type CategoryGroupsMapping = Mapping[str, Sequence[FlextTypes.GeneralValueType]]
    """Mapping for category groups (category names to entry lists)."""

    type ContainerConfigDict = dict[str, FlextTypes.GeneralValueType]

    type SharedContainersMapping = Mapping[str, FlextTypes.ContainerConfigDict]
    """Mapping for shared containers (container IDs to container objects)."""

    type EventDataMapping = Mapping[str, FlextTypes.GeneralValueType]
    # Mutable dict version for Pydantic models (DomainEvent.data)
    type EventDataDict = dict[str, FlextTypes.GeneralValueType]
    """Mapping for event data (event properties to values)."""

    type ContextMetadataMapping = Mapping[str, FlextTypes.GeneralValueType]
    """Mapping for context metadata (context properties to values)."""

    type ConfigurationMapping = Mapping[str, FlextTypes.GeneralValueType]
    """Mapping for configuration (configuration keys to values)."""

    type ConfigurationDict = dict[str, FlextTypes.GeneralValueType]
    """Mutable dict for configuration (configuration keys to values)."""

    type IncEx = set[str] | dict[str, set[str] | bool]
    """Pydantic include/exclude type for model_dump() and serialization.

    Used for include and exclude parameters in BaseModel.model_dump().
    Can be a set of field names or a dict mapping field names to nested include/exclude.
    """

    # Common string-to-value mappings
    type StringMapping = Mapping[str, str]
    """Mapping for string-to-string associations (e.g., key mappings, flags)."""

    type StringDict = dict[str, str]
    """Mutable dict for string-to-string associations (e.g., key mappings, flags)."""

    type StringIntMapping = Mapping[str, int]
    """Mapping for string-to-integer associations (e.g., metrics, counts)."""

    type StringIntDict = dict[str, int]
    """Mutable dict for string-to-integer associations (e.g., metrics, counts)."""

    type StringFloatMapping = Mapping[str, float]
    """Mapping for string-to-float associations (e.g., metrics, measurements)."""

    type StringFloatDict = dict[str, float]
    """Mutable dict for string-to-float associations (e.g., metrics, measurements)."""

    type StringBoolMapping = Mapping[str, bool]
    """Mapping for string-to-boolean associations (e.g., flags, features)."""

    type StringBoolDict = dict[str, bool]
    """Mutable dict for string-to-boolean associations (e.g., flags, permissions)."""

    type StringNumericMapping = Mapping[str, int | float]
    """Mapping for string-to-numeric associations (e.g., metrics)."""

    type StringNumericDict = dict[str, int | float]
    """Mutable dict for string-to-numeric associations (e.g., metrics)."""

    type NestedStringIntMapping = Mapping[str, Mapping[str, int]]
    """Nested mapping for string-to-string-to-integer (e.g., nested metrics)."""

    type NestedStringIntDict = dict[str, dict[str, int]]
    """Mutable nested dict for string-to-string-to-integer (e.g., nested metrics)."""

    type StringConfigurationDictDict = dict[str, dict[str, FlextTypes.GeneralValueType]]
    """Mutable nested dict for string-to-configuration-dict (e.g., scoped contexts)."""

    type StringListMapping = Mapping[str, list[FlextTypes.GeneralValueType]]
    """Mapping for string-to-list associations (e.g., categories)."""

    type StringListDict = dict[str, list[FlextTypes.GeneralValueType]]
    """Mutable dict for string-to-list associations (e.g., categories)."""

    type StringSequenceMapping = Mapping[str, Sequence[str]]
    """Mapping for string-to-string-sequence associations."""

    type StringSequenceDict = dict[str, Sequence[str]]
    """Mutable dict for string-to-string-sequence associations."""

    type StringListSequenceDict = dict[str, list[str]]
    """Mutable dict for string-to-string-list associations (e.g., hooks)."""

    type MetadataAttributeDict = dict[str, FlextTypes.MetadataAttributeValue]
    """Mutable dict for metadata attributes (string keys to metadata-compatible values)."""

    type StringGenericTypeArgumentTupleDict = dict[
        str,
        tuple[FlextTypes.GenericTypeArgument, ...],
    ]
    """Mutable dict for string-to-generic-type-argument-tuple mappings."""

    type GeneralValueDict = dict[str, FlextTypes.GeneralValueType]
    """Mutable dict for general value types (alias for ConfigurationDict)."""

    type DecoratorType = Callable[
        [FlextTypes.HandlerCallable],
        FlextTypes.HandlerCallable,
    ]
    """Type for decorators that wrap handler callables."""

    type FloatListDict = dict[str, list[float]]
    """Mutable dict for string-to-float-list associations (e.g., execution times)."""

    type HandlerTypeDict = dict[str, FlextTypes.HandlerType]
    """Mutable dict for handler type mappings."""

    type HandlerCallableDict = dict[str, FlextTypes.HandlerCallable]
    """Mutable dict for handler callable mappings."""

    type StringFlextLoggerDict = dict[str, FlextTypes.GeneralValueType]
    """Mutable dict for string-to-logger mappings.

    Note: Uses GeneralValueType as base type to allow any logger type
    (typically FlextLogger) without circular import.
    """

    type StringFlextExceptionTypeDict = dict[str, type[object]]
    """Mutable dict for string-to-exception-type mappings.

    Note: Uses 'type[object]' as base type to allow any exception type
    (typically e.BaseError) without circular import.
    """

    type StringCallableBoolStrTupleDict = dict[
        str,
        tuple[Callable[[object], bool], str],
    ]
    """Mutable dict for string-to-callable-bool-str-tuple mappings.

    Used for guard shortcuts mapping names to (check_fn, type_desc) tuples.
    """

    type StringStrEnumTypeDict = dict[str, type[StrEnum]]
    """Mutable dict for string-to-StrEnum-type mappings."""

    type StringStrEnumInstanceDict = dict[str, StrEnum]
    """Mutable dict for string-to-StrEnum-instance mappings (e.g., __members__)."""

    type StringPathDict = dict[str, Path]
    """Mutable dict for string-to-Path mappings (e.g., file paths)."""

    type StringTypeDict = dict[str, type]
    """Mutable dict for string-to-type mappings (generic type registry)."""

    type StringHandlerCallableListDict = dict[str, list[FlextTypes.HandlerCallable]]
    """Mutable dict for string-to-handler-callable-list mappings (e.g., hooks)."""

    type StringBaseSettingsTypeDict = dict[str, type[BaseSettings]]
    """Mutable dict for string-to-BaseSettings-type mappings (e.g., namespace registry)."""

    type StringSequenceGeneralValueMapping = Mapping[
        str,
        Sequence[FlextTypes.GeneralValueType],
    ]
    """Mapping for string-to-sequence-of-t.FlextTypes.GeneralValueType associations."""

    type StringSequenceGeneralValueDict = dict[
        str,
        Sequence[FlextTypes.GeneralValueType],
    ]
    """Mutable dict for string-to-sequence-of-t.FlextTypes.GeneralValueType associations."""

    type StringTupleFloatIntDict = dict[str, tuple[float, int]]
    """Mutable dict for string-to-(float, int)-tuple mappings (e.g., rate limit windows)."""

    type StringTupleGeneralValueDict = dict[
        str,
        tuple[FlextTypes.GeneralValueType, FlextTypes.GeneralValueType],
    ]
    """Mutable dict for string-to-(t.FlextTypes.GeneralValueType, t.FlextTypes.GeneralValueType)-tuple mappings (e.g., differences)."""

    # Container-specific types (forward references to avoid circular imports)
    # Service instance type - union of all types accepted by container.register()
    # ARCHITECTURAL EXCEPTION: DI containers must accept any Python object
    # This uses GeneralValueType + Protocol for type-safe service storage
    type ServiceInstanceType = FlextTypes.GeneralValueType
    """Type for service instances accepted by FlextContainer.register().

    Includes all GeneralValueType members plus Callables:
    - Primitives: str, int, float, bool, datetime, None
    - BaseModel: Pydantic models (covers FlextService, FlextConfig, etc.)
    - Path: File system paths
    - Sequences and Mappings: Recursive container types
    - Callable: Service classes, factory functions, loggers

    ARCHITECTURAL NOTE: DI containers require broad type acceptance.
    The ServiceRegistration model uses arbitrary_types_allowed=True
    to accept any Python object at runtime while maintaining type hints.
    """

    # Factory callable type - zero-argument factory returning GeneralValueType
    # Uses GeneralValueType for type-safe factory returns
    type FactoryCallable = Callable[[], FlextTypes.GeneralValueType]
    """Callable type for container factories.

    Zero-argument callable that returns GeneralValueType.
    Used by FlextContainer.register_factory() and with_factory().
    Supports factories returning BaseModel, primitives, sequences, and mappings.
    """

    # Resource callable type - same as factory but for lifecycle-managed resources
    type ResourceCallable = Callable[[], FlextTypes.GeneralValueType]
    """Callable type for container resources.

    Zero-argument callable that returns GeneralValueType.
    Used by FlextContainer.register_resource() and with_resource().
    """

    # Factory registration callable - type used by m.ContainerFactoryRegistration
    # More restricted than FactoryCallable (non-recursive scalar types)
    type FactoryRegistrationCallable = Callable[
        [],
        FlextTypes.ScalarValue
        | Sequence[FlextTypes.ScalarValue]
        | Mapping[str, FlextTypes.ScalarValue],
    ]
    """Callable type for factory registrations in m.ContainerFactoryRegistration.

    Zero-argument callable returning non-recursive scalar types.
    More restricted than FactoryCallable for Pydantic serialization compatibility.
    """

    # Service mapping type - for scoped container services parameter
    type ServiceMapping = Mapping[str, FlextTypes.ServiceInstanceType]
    """Mapping type for service registrations in scoped container creation."""

    # Factory mapping type - for scoped container factories parameter
    type FactoryMapping = Mapping[str, FlextTypes.FactoryRegistrationCallable]
    """Mapping type for factory registrations in scoped container creation."""

    # Resource mapping type - for scoped container resources parameter
    type ResourceMapping = Mapping[str, FlextTypes.ResourceCallable]
    """Mapping type for resource registrations in scoped container creation."""

    type ServiceRegistrationDict = dict[str, FlextTypes.GeneralValueType]
    """Mutable dict for service registration mappings.

    Note: Uses GeneralValueType as base type to allow any service registration type
    (typically m.ContainerServiceRegistration) without circular import.
    """

    type FactoryRegistrationDict = dict[str, FlextTypes.GeneralValueType]
    """Mutable dict for factory registration mappings.

    Note: Uses GeneralValueType as base type to allow any factory registration type
    (typically m.ContainerFactoryRegistration) without circular import.
    """

    # Additional mapping types for validation and exceptions
    # Validators return Result-like objects with is_failure/error attributes
    # Using object to avoid circular import with protocols.py
    type FieldValidatorMapping = Mapping[
        str,
        Callable[[FlextTypes.GeneralValueType], object],
    ]
    """Mapping for field validators (field names to validator returning Result-like)."""

    type ConsistencyRuleMapping = Mapping[
        str,
        Callable[[FlextTypes.GeneralValueType], object],
    ]
    """Mapping for consistency rules (rule names to validator returning Result-like)."""

    type ResourceRegistrationDict = dict[str, FlextTypes.GeneralValueType]
    """Mutable dict for resource registration mappings.

    Uses GeneralValueType to avoid circular imports (typically ResourceRegistration).
    """

    type EventValidatorMapping = Mapping[
        str,
        Callable[[FlextTypes.GeneralValueType], object],
    ]
    """Mapping for event validators (event types to validator returning Result-like)."""

    type ErrorTypeMapping = Mapping[
        str,
        str
        | int
        | float
        | Mapping[
            str,
            str
            | int
            | float
            | bool
            | Sequence[str | int | float | bool | None]
            | Mapping[str, str | int | float | bool | None]
            | None,
        ]
        | None,
    ]
    """Mapping for error type data with details, metadata, correlation IDs, timestamps."""

    type ExceptionKwargsType = (
        FlextTypes.ScalarValue
        | Sequence[FlextTypes.ScalarValue]
        | Mapping[
            str,
            FlextTypes.ScalarValue
            | Sequence[FlextTypes.ScalarValue]
            | Mapping[str, FlextTypes.ScalarValue],
        ]
    )
    """Type alias for exception kwargs - flexible value handling with proper parametrization."""

    # =====================================================================
    # TYPEDDICT CLASSES (Python 3.13+ PEP 695)
    # =====================================================================
    class BatchResultDict(BaseModel):
        """Result payload model for batch operation outputs."""

        model_config = ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        results: list[FlextTypes.GeneralValueType] = Field(default_factory=list)
        errors: list[tuple[int, str]] = Field(default_factory=list)
        total: int = Field(default=0)
        success_count: int = Field(default=0)
        error_count: int = Field(default=0)

        def __getitem__(
            self,
            key: str,
        ) -> list[FlextTypes.GeneralValueType] | list[tuple[int, str]] | int:
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
    # VALIDATION TYPES (Python 3.13+ Annotated with Pydantic constraints)
    # =====================================================================

    class Validation:
        """Validation type aliases with Pydantic constraints."""

        # Port number type with constraints (1-65535)
        type PortNumber = Annotated[int, Field(ge=1, le=65535)]
        """Port number type alias (1-65535)."""

        # Timeout type with constraints (positive float)
        type PositiveTimeout = Annotated[float, Field(gt=0.0, le=300.0)]
        """Positive timeout in seconds (0-300)."""

        # Retry count type with constraints
        type RetryCount = Annotated[int, Field(ge=0, le=10)]
        """Retry count (0-10)."""

        # Worker count type with constraints
        type WorkerCount = Annotated[int, Field(ge=1, le=100)]
        """Worker count (1-100)."""

    # Commonly used type aliases


t_core = FlextTypes
t = FlextTypes


__all__ = [
    "FlextTypes",
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
