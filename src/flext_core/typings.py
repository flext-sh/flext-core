"""Type aliases and generics used across dispatcher-ready components.

``FlextTypes`` centralizes ``TypeVar`` declarations and nested namespaces of
aliases for CQRS messages, handlers, utilities, logging, and validation. The
module keeps typing consistent for dispatcher pipelines, services, and examples
without importing higher-layer implementations while retaining compatibility
with legacy bus naming where necessary.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import socket
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import (
    Annotated,
    ParamSpec,
    TypedDict,
    TypeVar,
)

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

# ============================================================================
# Module-Level TypeVars - Core and Domain-Specific
# ============================================================================
# All TypeVars are defined at module level for clarity and accessibility

# Core TypeVars - covariant, contravariant, and generic type variables
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
P = ParamSpec("P")  # ParamSpec for decorator patterns
R = TypeVar("R")  # Return type for decorators

# Handler-specific TypeVars for protocol compliance
TInput_Handler_Protocol_contra = TypeVar(
    "TInput_Handler_Protocol_contra",
    contravariant=True,
)
TResult_Handler_Protocol = TypeVar("TResult_Handler_Protocol")
CallableInputT = TypeVar("CallableInputT")  # Input type for callable handlers
CallableOutputT = TypeVar("CallableOutputT")  # Output type for callable handlers

# Domain TypeVars for CQRS and domain-driven design patterns
TEntity_co = TypeVar("TEntity_co", covariant=True)
TResult_co = TypeVar("TResult_co", covariant=True)
TCommand_contra = TypeVar("TCommand_contra", contravariant=True)
TQuery_contra = TypeVar("TQuery_contra", contravariant=True)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
TResult_Handler_co = TypeVar("TResult_Handler_co", covariant=True)
TInput_Handler_contra = TypeVar("TInput_Handler_contra", contravariant=True)

# Generic TypeVars for utility and collection operations
E = TypeVar("E")  # Element type
F = TypeVar("F")  # Function type
K = TypeVar("K")  # Key type
ResultT = TypeVar("ResultT")  # Result type
U = TypeVar("U")  # Utility type
V = TypeVar("V")  # Value type
W = TypeVar("W")  # Wrapper type

# Additional domain TypeVars for advanced patterns
T1_co = TypeVar("T1_co", covariant=True)
T2_co = TypeVar("T2_co", covariant=True)
T3_co = TypeVar("T3_co", covariant=True)
TAggregate_co = TypeVar("TAggregate_co", covariant=True)
TCacheKey_contra = TypeVar("TCacheKey_contra", contravariant=True)
TCacheValue_co = TypeVar("TCacheValue_co", covariant=True)
TConfigKey_contra = TypeVar("TConfigKey_contra", contravariant=True)
TDomainEvent_co = TypeVar("TDomainEvent_co", covariant=True)
TEvent_contra = TypeVar("TEvent_contra", contravariant=True)
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TItem_contra = TypeVar("TItem_contra", contravariant=True)
TState_co = TypeVar("TState_co", covariant=True)
TUtil_contra = TypeVar("TUtil_contra", contravariant=True)
TValue_co = TypeVar("TValue_co", covariant=True)
TValueObject_co = TypeVar("TValueObject_co", covariant=True)
TResult_contra = TypeVar("TResult_contra", contravariant=True)

# Factory TypeVar for dependency injection patterns
# Uses forward reference to FlextTypes.GeneralValueType
FactoryT = TypeVar("FactoryT", bound=Callable[[], "FlextTypes.GeneralValueType"])

# Config-specific TypeVars
T_Config = TypeVar("T_Config")  # Bound removed - forward ref causes circular import
T_Namespace = TypeVar("T_Namespace")

# Model-specific TypeVars
T_Model = TypeVar("T_Model", bound=BaseModel)


class FlextTypes:
    """Type system foundation for FLEXT ecosystem - Complex Type Aliases Only.

    NOTE: All TypeVars are now defined at module level (see above).
    This class contains only complex type aliases, domain validation types,
    and nested helper classes.

    Architecture: Nested type definitions for:
    - Scalar and base types ("ScalarValue", FlextTypes.GeneralValueType, etc.)
    - Domain validation types (PortNumber, TimeoutSeconds, etc.)
    - JSON types (JsonPrimitive, JsonValue, etc.)
    - Handler types (HandlerCallable with dispatcher/bus-compatible aliases)
    - Utility types (SerializableType, MetadataDict, etc.)
    - CQRS pattern types (Command, Event, Message, Query)
    - Configuration models (RetryConfig)
    """

    # =====================================================================
    # SCALAR AND BASE TYPES (Python 3.13+ PEP 695 Type Aliases)
    # =====================================================================
    # Base scalar types for value handling across FLEXT ecosystem

    # Core scalar and value types using Python 3.13+ PEP 695 strict syntax
    # No aliases - direct composition for strict typing

    # Scalar value type - base types (no containers)
    type ScalarValue = str | int | float | bool | None

    # Recursive general value type - PEP 695 syntax for Pydantic compatibility
    # Supports JSON-serializable values: primitives, sequences, and mappings
    type GeneralValueType = (
        str
        | int
        | float
        | bool
        | Sequence[FlextTypes.GeneralValueType]
        | Mapping[str, FlextTypes.GeneralValueType]
        | None
    )

    # Sortable object type for cache utilities - objects that can be sorted/compared
    # Expanded from object to specific types for type safety
    type SortableObjectType = (
        str
        | int
        | float
        | bool
        | Sequence[FlextTypes.GeneralValueType]
        | Mapping[str, FlextTypes.GeneralValueType]
    )

    # Metadata-compatible attribute value type - composed for strict validation
    type MetadataAttributeValue = (
        str
        | int
        | float
        | bool
        | list[str | int | float | bool | None]
        | Mapping[str, str | int | float | bool | None]
        | None
    )

    # Generic metadata value type - supports nested structures for flexible metadata
    # Used across all FLEXT projects for metadata storage and exchange
    # Uses MetadataAttributeValue (non-recursive) to avoid Pydantic recursion issues
    # while maintaining type safety. For deeply nested structures, use
    # Mapping[str, FlextTypes.GeneralValueType] in model fields and validate manually if needed.
    # Direct type definition - no aliases per FLEXT standards
    type MetadataValue = MetadataAttributeValue

    # Generic metadata dictionary type - read-only interface for metadata containers
    type Metadata = Mapping[str, MetadataAttributeValue]
    """Structural type for metadata dictionaries using Mapping (read-only interface).

    Represents flexible metadata containers without importing models.
    Used in protocol definitions where metadata must be stored or passed.
    This is a generic type available to all FLEXT projects.
    """

    # Flexible value type for protocol methods - contains scalars, sequences, or mappings
    # Used in protocols for flexible input/output handling
    type FlexibleValue = ScalarValue | Sequence[ScalarValue] | Mapping[str, ScalarValue]

    # Constant value type - all possible constant types in FlextConstants
    # Used for type-safe constant access via __getitem__ method
    # Includes all types that can be stored as constants: primitives, collections,
    # Pydantic ConfigDict, and StrEnum types
    type ConstantValue = (
        str
        | int
        | float
        | bool
        | ConfigDict
        | frozenset[str]
        | tuple[str, ...]
        | Mapping[str, str | int]
        | StrEnum
        | type[StrEnum]
    )

    # Collection type aliases - fully parameterized for strict mode
    # NOTE: Using forward references for recursive FlextTypes.GeneralValueType
    type ObjectList = Sequence[FlextTypes.GeneralValueType]
    # Direct composition - no simple aliases for simple types

    class Validation:
        """Domain validation types using Pydantic Field annotations."""

        # Network validation types
        type PortNumber = Annotated[
            int,
            Field(ge=1, le=65535, description="Network port"),
        ]
        type TimeoutSeconds = Annotated[
            float,
            Field(gt=0, le=300, description="Timeout in seconds"),
        ]
        type RetryCount = Annotated[
            int,
            Field(ge=0, le=10, description="Retry attempts"),
        ]

        # String validation types
        type NonEmptyStr = Annotated[
            str,
            Field(min_length=1, description="Non-empty string"),
        ]

        @staticmethod
        def _validate_hostname(value: str) -> str:
            """Validate hostname by attempting DNS resolution."""
            try:
                _ = socket.gethostbyname(value)
                return value
            except socket.gaierror as e:
                msg = f"Cannot resolve hostname '{value}': {e}"
                raise ValueError(msg) from e

        type HostName = Annotated[
            str,
            Field(min_length=1, max_length=253, description="Valid hostname"),
            AfterValidator(_validate_hostname),
        ]

    class Json:
        """JSON serialization types for API and data exchange."""

        # Primitive JSON types
        type JsonPrimitive = str | int | float | bool | None

        # Complex JSON types using recursive FlextTypes.GeneralValueType
        type JsonValue = (
            JsonPrimitive
            | Sequence[FlextTypes.GeneralValueType]
            | Mapping[str, FlextTypes.GeneralValueType]
        )
        type JsonList = Sequence[JsonValue]
        type JsonDict = Mapping[str, FlextTypes.GeneralValueType]

    class Handler:
        """Handler and middleware type definitions for CQRS patterns."""

        # Single consolidated callable type for handlers and validators
        # All handler callables share the same signature: input GeneralValueType -> output GeneralValueType
        type HandlerCallable = Callable[
            [FlextTypes.GeneralValueType],
            FlextTypes.GeneralValueType,
        ]
        # Middleware uses same callable signature as handlers
        # Use HandlerCallable directly - no aliases per FLEXT standards

        # Handler type for registry - union of all possible handler types
        # This includes callables, objects with handle() method, and handler instances
        # Note: Handler protocol is defined in FlextProtocols.Handler for proper protocol organization

        # Middleware configuration types
        type MiddlewareConfig = FlextTypes.Types.ConfigurationMapping

        # Acceptable message types for handlers - union of common message types
        type AcceptableMessageType = (
            FlextTypes.GeneralValueType
            | Mapping[str, FlextTypes.GeneralValueType]
            | Sequence[FlextTypes.GeneralValueType]
            | str
            | int
            | float
            | bool
            | None
        )

        # Conditional execution callable types (PEP 695)
        type ConditionCallable = Callable[[FlextTypes.GeneralValueType], bool]

        # Variadic callable protocol is defined in FlextProtocols.VariadicCallable
        # Use that instead of redeclaring here

        # Handler type union - all possible handler representations
        # This is used for handler registries where handlers can be callables,
        # handler instances, or configuration dicts
        # Note: Handler and VariadicCallable protocols are defined in FlextProtocols
        # but cannot be imported here due to circular dependency. For type checking,
        # we use Callable as a fallback that works for most use cases.
        type HandlerType = (
            HandlerCallable
            | Callable[..., FlextTypes.GeneralValueType]  # Variadic callable fallback
            | Mapping[str, FlextTypes.GeneralValueType]  # Configuration dict
        )

    class Processor:
        """Processor types for data transformation pipelines."""

        # Input/output types for processing operations
        # Use GeneralValueType directly - no aliases per FLEXT standards
        # For output, use GeneralValueType | None directly where needed

    class Factory:
        """Factory pattern types for dependency injection."""

        # Factory callable and type variable definitions
        type FactoryCallableType = Callable[[], FlextTypes.GeneralValueType]

    class Predicate:
        """Predicate types for filtering and validation operations."""

        type PredicateType = Callable[[FlextTypes.GeneralValueType], bool]

    class Decorator:
        """Decorator pattern types for function enhancement."""

        type DecoratorReturnType = Callable[
            [Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType]],
            Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType],
        ]

    class Utility:
        """General utility types for common operations."""

        # Direct composition - no simple aliases for simple types
        type TypeOriginSpecifier = str | type[FlextTypes.GeneralValueType]
        type TypeHintSpecifier = str | type[FlextTypes.GeneralValueType]
        type MessageTypeSpecifier = str | type[FlextTypes.GeneralValueType]

        type GenericTypeArgument = str | type[FlextTypes.GeneralValueType]

        # Object types for caching and messaging
        # Use GeneralValueType directly - no aliases per FLEXT standards
        # For cached objects, serializable types, and parameter values, use GeneralValueType directly
        # For JSON objects, use Mapping[str, GeneralValueType] directly

        # Collection and caching types
        type ObjectList = Sequence[FlextTypes.GeneralValueType]

        # Registry and configuration types
        type MetadataDict = FlextTypes.Types.ServiceMetadataMapping
        type ServiceRegistry = Mapping[str, FlextTypes.GeneralValueType]
        type FactoryRegistry = Mapping[str, Callable[[], FlextTypes.GeneralValueType]]
        type ContainerConfig = FlextTypes.Types.ConfigurationMapping

        # Event and payload types - use types directly, no aliases
        # EventPayload uses EventDataMapping (complex type, keep)
        # CommandPayload and QueryPayload use GeneralValueType directly

    class Bus:
        """Dispatcher routing type aliases (legacy "Bus" namespace retained)."""

        # Message and handler types for bus operations
        # Use GeneralValueType directly - no aliases per FLEXT standards
        # For message types, use GeneralValueType directly
        # For handler types, use Handler.HandlerCallable directly
        # For configuration, use Types.ConfigurationMapping | None directly

    class Logging:
        """Logging and context type definitions."""

        # Context and processor types for logging operations
        # Use Types.ContextMetadataMapping directly - no aliases per FLEXT standards
        # For context values, args, and kwargs, use GeneralValueType directly
        # For processor and bound logger types, use Callable directly with proper signatures

    class Hook:
        """Hook and registry types for extensibility."""

        # Hook callable and registry types
        # Use Handler.HandlerCallable directly for hook callables - no aliases per FLEXT standards
        # For registries, use Mapping[str, GeneralValueType] directly

    class Config:
        """Configuration models for operational settings."""

        class RetryConfig(BaseModel):
            """Configuration for retry operations with exponential backoff.

            Defines retry behavior for operations that may fail and need
            automatic retry with configurable backoff strategies.
            """

            model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

            # Retry attempt limits
            max_attempts: int = Field(ge=1, description="Maximum retry attempts")

            # Timing configuration
            initial_delay_seconds: float = Field(
                gt=0.0,
                description="Initial delay in seconds",
            )
            max_delay_seconds: float = Field(
                gt=0.0,
                description="Maximum delay in seconds",
            )

            # Backoff strategy configuration
            exponential_backoff: bool = Field(description="Enable exponential backoff")
            backoff_multiplier: float = Field(
                default=2.0,
                gt=0.0,
                description="Backoff multiplier",
            )

            # Exception handling configuration
            retry_on_exceptions: list[type[Exception]] = Field(
                description="Exception types to retry on",
            )

    class Cqrs:
        """CQRS pattern types - use GeneralValueType directly."""

        # CQRS types use GeneralValueType directly - no aliases per FLEXT standards
        # For commands, events, messages, and queries, use GeneralValueType directly

    class Types:
        """Type aliases for structured mappings and TypedDict definitions.

        Provides Python 3.13+ PEP 695 type aliases for Mapping types and
        TypedDict class definitions used throughout the FLEXT ecosystem.
        All types use proper annotations for strict type checking.

        **Mapping Type Aliases**:
        - ServiceMetadataMapping: Service metadata structure
        - FieldMetadataMapping: Field metadata structure
        - SummaryDataMapping: Summary data structure
        - CategoryGroupsMapping: Category groups structure
        - SharedContainersMapping: Shared container configuration
        - EventDataMapping: Event data structure
        - ContextMetadataMapping: Context metadata structure
        - ConfigurationMapping: Configuration structure
        - ConfigInstanceMapping: Config instance structure
        - NamespaceRegistryMapping: Namespace registry structure

        **TypedDict Classes**:
        - ServiceRegistrationMetadata: Service registration metadata
        - FactoryRegistrationMetadata: Factory registration metadata
        - ValidationContextDict: Validation context data
        - EventDataDict: Domain event data
        - EntityDataDict: Entity data structure
        - CategoryGroupDict: Category grouping data
        - ContainerConfigDict: Container configuration
        - DependencyContainerConfigDict: Dependency injection container config
        - DockerServiceConfigDict: Docker service configuration
        """

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

        # ContainerConfigDict must be defined before SharedContainersMapping
        class ContainerConfigDict(TypedDict, total=True):
            """Container configuration for Docker services."""

            compose_file: str
            service: str
            port: int

        type SharedContainersMapping = Mapping[str, ContainerConfigDict]
        """Mapping for shared containers (container IDs to container objects)."""

        type EventDataMapping = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping for event data (event properties to values)."""

        type ContextMetadataMapping = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping for context metadata (context properties to values)."""

        type ConfigurationMapping = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping for configuration (configuration keys to values)."""

        type ConfigInstanceMapping = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping for configuration instances (instance IDs to instances)."""

        type NamespaceRegistryMapping = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping for namespace registry (namespace names to registries)."""

        # Additional mapping types for validation and exceptions
        type FieldValidatorMapping = Mapping[
            str,
            Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType],
        ]
        """Mapping for field validators (field names to validator functions)."""

        type ConsistencyRuleMapping = Mapping[
            str,
            Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType],
        ]
        """Mapping for consistency rules (rule names to validator functions)."""

        type EventValidatorMapping = Mapping[
            str,
            Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType],
        ]
        """Mapping for event validators (event types to validator functions)."""

        type NestedExceptionLevelMapping = dict[str, dict[type, str]]
        """Nested dict for exception levels (library → exception_type → level)."""

        type ExceptionLevelMapping = dict[str, str]
        """Mutable dict for exception levels (container names to levels)."""

        type ErrorTypeMapping = dict[
            str,
            str
            | int
            | float
            | dict[
                str,
                str
                | int
                | float
                | bool
                | list[str | int | float | bool | None]
                | dict[str, str | int | float | bool | None]
                | None,
            ]
            | None,
        ]
        """Dict for error type data with details, metadata, correlation IDs, timestamps."""

        type ExceptionMetricsMapping = dict[type, int]
        """Mutable dict for exception metrics (exception types to occurrence counts)."""

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
        class ServiceRegistrationMetadata(TypedDict, total=True):
            """Metadata for service registration in the container.

            Used for tracking service registration details including
            handler type, registration status, and metadata.
            """

            handler_type: str
            """Type of handler (command, query, event, etc.)."""

            registration_status: str
            """Status of registration (active, inactive, error)."""

            metadata: dict[str, FlextTypes.GeneralValueType]
            """Additional service metadata (JSON-serializable)."""

        class FactoryRegistrationMetadata(TypedDict, total=True):
            """Metadata for factory registration in the container.

            Used for tracking factory registration details.
            """

            factory_type: str
            """Type of factory."""

            registration_status: str
            """Status of registration (active, inactive, error)."""

            metadata: dict[str, FlextTypes.GeneralValueType]
            """Additional factory metadata (JSON-serializable)."""

        class ValidationContextDict(TypedDict, total=False):
            """Context dictionary for validation operations.

            Used for passing validation context data between validators.
            """

            field_name: str
            """Name of the field being validated."""

            field_value: FlextTypes.GeneralValueType
            """Value of the field being validated."""

            parent_data: dict[str, FlextTypes.GeneralValueType]
            """Parent data context for cross-field validation."""

        class EventDataDict(TypedDict, total=True):
            """Data dictionary for event objects.

            Used for structured event data in domain events.
            """

            event_type: str
            """Type of event."""

            event_data: dict[str, FlextTypes.GeneralValueType]
            """Event payload data."""

            metadata: dict[str, FlextTypes.GeneralValueType]
            """Event metadata (timestamp, correlation ID, etc.)."""

        class EntityDataDict(TypedDict, total=False):
            """Data dictionary for entity objects.

            Used for structured entity data representation.
            """

            entity_id: str
            """Unique entity identifier."""

            entity_type: str
            """Type classification of the entity."""

            data: dict[str, FlextTypes.GeneralValueType]
            """Entity data payload."""

        class CategoryGroupDict(TypedDict, total=True):
            """Data dictionary for category groups.

            Used for categorized data grouping.
            """

            category_name: str
            """Name of the category."""

            items: list[FlextTypes.GeneralValueType]
            """List of items in this category."""

        class DependencyContainerConfigDict(TypedDict, total=True):
            """Configuration dictionary for dependency injection containers.

            Used for dependency injection container configuration.
            """

            service_name: str
            """Name of the service."""

            service_type: str
            """Type of service."""

            config: dict[str, FlextTypes.GeneralValueType]
            """Service configuration data."""

        # DockerServiceConfigDict uses ContainerConfigDict directly - no aliases


# NOTE: All TypeVars are defined at module level (lines 28-81)
# GeneralValueType is defined in FlextTypes class (line 130)
# Use FlextTypes.GeneralValueType for type annotations
# Module-level aliases are not needed - use FlextTypes namespace directly

__all__ = [
    "CallableInputT",
    "CallableOutputT",
    "E",
    "F",
    "FactoryT",
    "FlextTypes",
    "K",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "T",
    "T1_co",
    "T2_co",
    "T3_co",
    "TAggregate_co",
    "TCacheKey_contra",
    "TCacheValue_co",
    "TCommand_contra",
    "TConfigKey_contra",
    "TDomainEvent_co",
    "TEntity_co",
    "TEvent_contra",
    "TInput_Handler_Protocol_contra",
    "TInput_Handler_contra",
    "TInput_contra",
    "TItem_contra",
    "TQuery_contra",
    "TResult_Handler_Protocol",
    "TResult_Handler_co",
    "TResult_co",
    "TResult_contra",
    "TState_co",
    "TUtil_contra",
    "TValueObject_co",
    "TValue_co",
    "T_Config",
    "T_Model",
    "T_Namespace",
    "T_co",
    "T_contra",
    "U",
    "V",
    "W",
]
