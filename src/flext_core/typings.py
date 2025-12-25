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
from types import ModuleType
from typing import (
    ParamSpec,
    TypeAlias,
    TypedDict,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    LaxStr: TypeAlias = str | bytes | bytearray

    # =====================================================================
    # COMPLEX TYPE ALIASES (Python 3.13+ PEP 695 strict)
    # =====================================================================
    # Complex types that require composition or domain-specific logic

    # Core scalar and value types using Python 3.13+ PEP 695 strict syntax
    type ScalarValue = str | int | float | bool | datetime | None

    # Recursive general value type - PEP 695 syntax for Pydantic compatibility
    # Supports JSON-serializable values: primitives, sequences, and mappings
    # Note: Recursive type uses forward reference (managed by __future__ annotations)
    # Reuses ScalarValue defined above
    # PEP 695 recursive types work with __future__ annotations
    # Use string forward reference for recursive types to avoid pyrefly errors
    type GeneralValueType = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | BaseModel
        | Path
        | Sequence[FlextTypes.GeneralValueType]
        | Mapping[str, FlextTypes.GeneralValueType]
    )

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
        | Pattern[str]  # For regex pattern constants
        | type  # For nested namespace classes (e.g., FlextConstants.Network)
    )

    # Object list type - sequence of general value types for batch operations
    # Reuses "GeneralValueType" (forward reference managed by __future__ annotations)
    type ObjectList = Sequence[GeneralValueType]

    # File content type supporting all serializable formats including Pydantic models
    # Used for file operations that can handle different content types
    # Reuses "GeneralValueType" for configuration mappings
    type FileContent = (
        str
        | bytes
        | Mapping[str, GeneralValueType]
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

    # Metadata-compatible attribute value type - composed for strict
    # validation
    # Reuses ScalarValue defined above (forward reference managed by
    # __future__ annotations)
    type MetadataAttributeValue = (
        str
        | int
        | float
        | bool
        | datetime
        | BaseModel
        | Path
        | None
        | Sequence[FlextTypes.MetadataAttributeValue]
        | Mapping[str, FlextTypes.MetadataAttributeValue]
    )

    # Generic metadata dictionary type - read-only interface for metadata containers
    type Metadata = Mapping[str, MetadataAttributeValue]
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
    type FlexibleMapping = Mapping[str, FlexibleValue]

    # NOTE: ParameterValueType was removed - use t.GeneralValueType directly
    # No aliases for convenience - use types directly per FLEXT standards

    # Type hint specifier - used for type introspection
    # Can be any type hint: type, type alias, generic, or string
    # Note: For runtime type introspection, we accept type, str,
    # or callable types
    # Note: Removed Callable[..., T] to avoid variadic callables - use specific
    # callable types instead
    type TypeHintSpecifier = type | str | Callable[[GeneralValueType], GeneralValueType]

    # Generic type argument - used for extracting generic type arguments
    # Can be a string type name or a type class representing "GeneralValueType"
    # Reuses "GeneralValueType" from parent t class (forward reference)
    type GenericTypeArgument = str | type[GeneralValueType]

    # Message type specifier - used for handler type checking
    # Can be a string type name or a type class
    # Reuses "GeneralValueType" from parent t class (forward reference)
    type MessageTypeSpecifier = str | type[GeneralValueType]

    # Type origin specifier - used for generic type origin checking
    # Can be a string type name, type class, or callable with __origin__ attribute
    # Reuses "GeneralValueType" from parent t class (forward reference)
    # Note: Removed Callable[..., T] to avoid variadic callables - use specific callable types instead
    type TypeOriginSpecifier = (
        str | type[GeneralValueType] | Callable[[GeneralValueType], GeneralValueType]
    )

    # Primitive JSON types
    type JsonPrimitive = str | int | float | bool | None

    # Complex JSON types using recursive "GeneralValueType"
    # Reuses "GeneralValueType" from parent t class (no duplication)
    type JsonValue = (
        JsonPrimitive | Sequence[GeneralValueType] | Mapping[str, GeneralValueType]
    )
    type JsonList = Sequence[JsonValue]
    type JsonDict = Mapping[str, GeneralValueType]

    # Single consolidated callable type for handlers and validators
    # Reuses "GeneralValueType" from outer t class
    type HandlerCallable = Callable[
        [GeneralValueType],
        GeneralValueType,
    ]
    # Middleware uses same callable signature as handlers
    # Use HandlerCallable directly - no aliases per FLEXT standards

    # Handler type for registry - union of all possible handler types
    # This includes callables, objects with handle() method, and handler instances
    # Note: Handler protocol is defined in p.Handler for proper protocol organization

    # Middleware configuration type
    type MiddlewareConfig = Mapping[str, GeneralValueType]

    # Acceptable message types for handlers - union of common message types
    # Reuses "GeneralValueType" and ScalarValue from parent t class
    type AcceptableMessageType = (
        "GeneralValueType"
        | Mapping[str, GeneralValueType]
        | Sequence[GeneralValueType]
        | ScalarValue
    )

    # Conditional execution callable types (PEP 695)
    # Reuses "GeneralValueType" from parent t class (forward reference)
    type ConditionCallable = Callable[[GeneralValueType], bool]

    # Variadic callable protocol is defined in p.VariadicCallable
    # Use that instead of redeclaring here

    # Handler type union - all possible handler representations
    # This is used for handler registries where handlers can be callables,
    # handler instances, or configuration dicts
    # Note: Handler and VariadicCallable protocols are defined in p
    # but cannot be imported here due to circular dependency. For type checking,
    # we use HandlerCallable which covers most use cases.
    # Reuses "GeneralValueType" from parent t class (no duplication)
    # Note: Removed Callable[..., T] to avoid variadic callables - HandlerCallable covers variadic cases
    type HandlerType = (
        HandlerCallable | Mapping[str, GeneralValueType]  # Configuration dict
    )

    class Dispatcher:
        """Dispatcher configuration types namespace."""

        class DispatcherConfig(TypedDict, total=True):
            """Typed dictionary for dispatcher configuration values.

            Business Rule: TypedDict uses dict[str, ...] for field types because
            TypedDict defines the structure of a dictionary with known keys.
            All fields required (total=True) for complete dispatcher configuration.

            Audit Implication: Used for dispatcher configuration in message processing
            systems. Complete configuration ensures proper dispatcher lifecycle management
            and audit trail completeness for dispatcher operations.
            """

            dispatcher_timeout_seconds: float
            """Timeout in seconds for dispatcher operations."""

            executor_workers: int
            """Number of worker threads for executor."""

            circuit_breaker_threshold: int
            """Threshold for circuit breaker activation."""

            rate_limit_max_requests: int
            """Maximum number of requests per rate limit window."""

            rate_limit_window_seconds: float
            """Time window in seconds for rate limiting."""

            max_retry_attempts: int
            """Maximum number of retry attempts for failed operations."""

            retry_delay: float
            """Delay in seconds between retry attempts."""

            enable_timeout_executor: bool
            """Enable timeout executor for operations."""

            dispatcher_enable_logging: bool
            """Enable logging for dispatcher operations."""

            dispatcher_auto_context: bool
            """Enable automatic context creation for dispatcher."""

            dispatcher_enable_metrics: bool
            """Enable metrics collection for dispatcher."""

    # =====================================================================
    # MAPPING TYPE ALIASES (Python 3.13+ PEP 695)
    # =====================================================================
    # Using PEP 695 type keyword for better type checking and IDE support
    type ServiceMetadataMapping = Mapping[str, GeneralValueType]
    """Mapping for service metadata (attribute names to values)."""

    type FieldMetadataMapping = Mapping[str, GeneralValueType]
    """Mapping for field metadata (field names to metadata objects)."""

    type SummaryDataMapping = Mapping[str, int | float | str]
    """Mapping for summary data (category names to summary values)."""

    type CategoryGroupsMapping = Mapping[str, Sequence[GeneralValueType]]
    """Mapping for category groups (category names to entry lists)."""

    # ContainerConfigDict must be defined before SharedContainersMapping
    class ContainerConfigDict(TypedDict, total=True):
        """Container configuration for Docker services.

        Business Rule: TypedDict uses dict[str, ...] for field types because
        TypedDict defines the structure of a dictionary with known keys.
        All fields required (total=True) for complete container configuration.

        Audit Implication: Used for Docker container configuration in deployment
        systems. Complete configuration ensures proper container lifecycle management
        and audit trail completeness for container operations.
        """

        compose_file: str
        service: str
        port: int

    type SharedContainersMapping = Mapping[str, ContainerConfigDict]
    """Mapping for shared containers (container IDs to container objects)."""

    type EventDataMapping = Mapping[str, GeneralValueType]
    # Mutable dict version for Pydantic models (DomainEvent.data)
    type EventDataDict = dict[str, GeneralValueType]
    """Mapping for event data (event properties to values)."""

    type ContextMetadataMapping = Mapping[str, GeneralValueType]
    """Mapping for context metadata (context properties to values)."""

    type ConfigurationMapping = Mapping[str, GeneralValueType]
    """Mapping for configuration (configuration keys to values)."""

    type ConfigurationDict = dict[str, GeneralValueType]
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

    type StringConfigurationDictDict = dict[str, dict[str, GeneralValueType]]
    """Mutable nested dict for string-to-configuration-dict (e.g., scoped contexts)."""

    type StringListMapping = Mapping[str, list[GeneralValueType]]
    """Mapping for string-to-list associations (e.g., categories)."""

    type StringListDict = dict[str, list[GeneralValueType]]
    """Mutable dict for string-to-list associations (e.g., categories)."""

    type StringSequenceMapping = Mapping[str, Sequence[str]]
    """Mapping for string-to-string-sequence associations."""

    type StringSequenceDict = dict[str, Sequence[str]]
    """Mutable dict for string-to-string-sequence associations."""

    type StringListSequenceDict = dict[str, list[str]]
    """Mutable dict for string-to-string-list associations (e.g., hooks)."""

    type MetadataAttributeDict = dict[str, MetadataAttributeValue]
    """Mutable dict for metadata attributes (string keys to metadata-compatible values)."""

    type StringGenericTypeArgumentTupleDict = dict[
        str,
        tuple[GenericTypeArgument, ...],
    ]
    """Mutable dict for string-to-generic-type-argument-tuple mappings."""

    type GeneralValueDict = dict[str, GeneralValueType]
    """Mutable dict for general value types (alias for ConfigurationDict)."""

    type DecoratorType = Callable[[HandlerCallable], HandlerCallable]
    """Type for decorators that wrap handler callables."""

    type FloatListDict = dict[str, list[float]]
    """Mutable dict for string-to-float-list associations (e.g., execution times)."""

    type HandlerTypeDict = dict[str, HandlerType]
    """Mutable dict for handler type mappings."""

    type HandlerCallableDict = dict[str, HandlerCallable]
    """Mutable dict for handler callable mappings."""

    type StringFlextLoggerDict = dict[str, object]
    """Mutable dict for string-to-logger mappings.

    Note: Uses 'object' as base type to allow any logger type
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

    type StringHandlerCallableListDict = dict[str, list[HandlerCallable]]
    """Mutable dict for string-to-handler-callable-list mappings (e.g., hooks)."""

    type StringBaseSettingsTypeDict = dict[str, type[BaseSettings]]
    """Mutable dict for string-to-BaseSettings-type mappings (e.g., namespace registry)."""

    type StringSequenceGeneralValueMapping = Mapping[
        str,
        Sequence[GeneralValueType],
    ]
    """Mapping for string-to-sequence-of-t.GeneralValueType associations."""

    type StringSequenceGeneralValueDict = dict[str, Sequence[GeneralValueType]]
    """Mutable dict for string-to-sequence-of-t.GeneralValueType associations."""

    type StringTupleFloatIntDict = dict[str, tuple[float, int]]
    """Mutable dict for string-to-(float, int)-tuple mappings (e.g., rate limit windows)."""

    type StringTupleGeneralValueDict = dict[
        str,
        tuple[GeneralValueType, GeneralValueType],
    ]
    """Mutable dict for string-to-(t.GeneralValueType, t.GeneralValueType)-tuple mappings (e.g., differences)."""

    # Container-specific types (forward references to avoid circular imports)
    # Service instance type - union of all types accepted by container.register()
    # Uses 'object' as base to accept any service type (protocols, config, context, etc.)
    type ServiceInstanceType = object
    """Type for service instances accepted by FlextContainer.register().

    Includes:
    - t.GeneralValueType: Primitives, sequences, mappings
    - BaseModel: Pydantic models
    - Callable[..., t.GeneralValueType]: Callable services (variadic signature)
    - object: Arbitrary object instances (protocols, loggers, configs, contexts, etc.)

    Note: Using 'object' allows registration of protocol instances (p.Config,
    p.Ctx, etc.) and other arbitrary services.
    """

    # Factory callable type - zero-argument factory returning any object
    # Uses 'object' to support factories returning BaseModel, protocols, etc.
    type FactoryCallable = Callable[[], object]
    """Callable type for container factories.

    Zero-argument callable that returns any object.
    Used by FlextContainer.register_factory() and with_factory().
    Supports factories returning BaseModel, protocols, loggers, etc.
    """

    # Resource callable type - same as factory but for lifecycle-managed resources
    type ResourceCallable = Callable[[], object]
    """Callable type for container resources.

    Zero-argument callable that returns any object.
    Used by FlextContainer.register_resource() and with_resource().
    """

    # Factory registration callable - type used by m.ContainerFactoryRegistration
    # More restricted than FactoryCallable (non-recursive scalar types)
    type FactoryRegistrationCallable = Callable[
        [],
        ScalarValue | Sequence[ScalarValue] | Mapping[str, ScalarValue],
    ]
    """Callable type for factory registrations in m.ContainerFactoryRegistration.

    Zero-argument callable returning non-recursive scalar types.
    More restricted than FactoryCallable for Pydantic serialization compatibility.
    """

    # Service mapping type - for scoped container services parameter
    type ServiceMapping = Mapping[str, ServiceInstanceType]
    """Mapping type for service registrations in scoped container creation."""

    # Factory mapping type - for scoped container factories parameter
    type FactoryMapping = Mapping[str, FactoryRegistrationCallable]
    """Mapping type for factory registrations in scoped container creation."""

    # Resource mapping type - for scoped container resources parameter
    type ResourceMapping = Mapping[str, ResourceCallable]
    """Mapping type for resource registrations in scoped container creation."""

    type ServiceRegistrationDict = dict[str, object]
    """Mutable dict for service registration mappings.

    Note: Uses 'object' as base type to allow any service registration type
    (typically m.ContainerServiceRegistration) without circular import.
    """

    type FactoryRegistrationDict = dict[str, object]
    """Mutable dict for factory registration mappings.

    Note: Uses 'object' as base type to allow any factory registration type
    (typically m.ContainerFactoryRegistration) without circular import.
    """

    # Additional mapping types for validation and exceptions
    # Validators return Result-like objects with is_failure/error attributes
    # Using object to avoid circular import with protocols.py
    type FieldValidatorMapping = Mapping[
        str,
        Callable[[GeneralValueType], object],
    ]
    """Mapping for field validators (field names to validator returning Result-like)."""

    type ConsistencyRuleMapping = Mapping[
        str,
        Callable[[GeneralValueType], object],
    ]
    """Mapping for consistency rules (rule names to validator returning Result-like)."""

    type ResourceRegistrationDict = dict[str, object]
    """Mutable dict for resource registration mappings.

    Uses object to avoid circular imports (typically ResourceRegistration).
    """

    type EventValidatorMapping = Mapping[
        str,
        Callable[[GeneralValueType], object],
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
        ScalarValue
        | Sequence[ScalarValue]
        | Mapping[
            str,
            ScalarValue | Sequence[ScalarValue] | Mapping[str, ScalarValue],
        ]
    )
    """Type alias for exception kwargs - flexible value handling with proper parametrization."""

    # =====================================================================
    # TYPEDDICT CLASSES (Python 3.13+ PEP 695)
    # =====================================================================
    class BatchResultDictBase(TypedDict, total=True):
        """Base TypedDict for batch processing operations.

        Business Rule: TypedDict uses dict[str, ...] for field types because
        TypedDict defines the structure of a dictionary with known keys.
        All fields required (total=True) for complete batch result representation.
        Note: TypedDict doesn't support generics directly, so results field uses
        list[t.GeneralValueType] for type flexibility. Type narrowing can be done
        at usage site based on operation return type.

        Audit Implication: Used for batch operation result tracking and auditing.
        Complete result structure ensures audit trail completeness for batch operations.
        """

        results: list[object]
        """List of successful processing results.

        Business Rule: TypedDict fields must specify dict structure.
        Contains successfully processed items in order.
        Type is t.GeneralValueType for flexibility; actual type depends on operation.
        """

        errors: list[tuple[int, str]]
        """List of error tuples (index, error_message).

        Business Rule: TypedDict fields must specify dict structure.
        Contains (index, error_message) tuples for failed items.
        """

        total: int
        """Total number of items processed.

        Business Rule: TypedDict fields must specify dict structure.
        Total count includes both successful and failed items.
        """

        success_count: int
        """Number of successfully processed items.

        Business Rule: TypedDict fields must specify dict structure.
        Count of items in results list.
        """

        error_count: int
        """Number of failed items.

        Business Rule: TypedDict fields must specify dict structure.
        Count of items in errors list.
        """

    # Public type alias for batch result (avoids SLF001 violation)
    # Use PEP 695 type alias for proper type checking support
    type BatchResultDict = BatchResultDictBase

    # Note: TypedDict cannot be generic, so BatchResultDict is used directly
    # Type narrowing for results: list[T] is done at usage site based on operation return type
    # Users should use BatchResultDict directly

    class RuntimeBootstrapOptions(TypedDict, total=False):
        """Typed dictionary for runtime bootstrap options.

        Business Rule: TypedDict uses dict[str, ...] for field types because
        TypedDict defines the structure of a dictionary with known keys.
        All fields are optional (total=False) as subclasses can override
        only specific options. This TypedDict matches the signature of
        FlextRuntime.create_service_runtime() to reduce casts and improve
        type safety.

        Audit Implication: Used for runtime bootstrap configuration in service
        initialization. Complete configuration ensures proper runtime lifecycle
        management and audit trail completeness for service operations.
        """

        config_type: type[BaseModel]
        """Config type for service runtime (defaults to FlextSettings)."""

        config_overrides: Mapping[str, FlextTypes.FlexibleValue]
        """Configuration overrides to apply to the config instance."""

        context: object
        """Context instance (p.Ctx) for service runtime.

        Note: Uses object to avoid circular import with protocols.py.
        Actual type is p.Ctx, but protocols cannot be imported here.
        """

        subproject: str
        """Subproject identifier for scoped container creation."""

        services: Mapping[
            str,
            FlextTypes.GeneralValueType | BaseModel | object,
        ]
        """Service registrations for container.

        Note: Uses object for p.VariadicCallable["GeneralValueType"] to avoid
        circular import with protocols.py. Actual type includes callables.
        """

        factories: Mapping[
            str,
            Callable[
                [],
                (
                    FlextTypes.ScalarValue
                    | Sequence[FlextTypes.ScalarValue]
                    | Mapping[str, FlextTypes.ScalarValue]
                ),
            ],
        ]
        """Factory registrations for container."""

        resources: Mapping[str, Callable[[], FlextTypes.GeneralValueType]]
        """Resource registrations for container."""

        container_overrides: Mapping[str, FlextTypes.FlexibleValue]
        """Container configuration overrides."""

        wire_modules: Sequence[ModuleType]
        """Modules to wire for dependency injection."""

        wire_packages: Sequence[str]
        """Packages to wire for dependency injection."""

        wire_classes: Sequence[type]
        """Classes to wire for dependency injection."""

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
