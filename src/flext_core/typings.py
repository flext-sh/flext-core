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
from datetime import datetime
from enum import StrEnum
from re import Pattern
from typing import (
    Annotated,
    ParamSpec,
    TypedDict,
    TypeVar,
)

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
FactoryT = TypeVar("FactoryT")

# Config-specific TypeVars
T_Config = TypeVar("T_Config")
T_Namespace = TypeVar("T_Namespace")
T_Settings = TypeVar("T_Settings", bound=BaseSettings)

# Model-specific TypeVars
T_Model = TypeVar("T_Model", bound=BaseModel)

# ============================================================================
# FLEXT TYPES - All complex types defined inside FlextTypes class
# ============================================================================
# NOTE: Only TypeVars are defined at module level (see above)
# All complex type aliases MUST be inside FlextTypes class - NO loose types!


class FlextTypes:
    """Type system foundation for FLEXT ecosystem - Complex Type Aliases Only.

    NOTE: All TypeVars are defined at module level (see above).
    This class contains complex type aliases and nested helper classes.
    All complex types are defined at module level and referenced here.

    Architecture: Nested type definitions for:
    - Scalar and base types ("FlextTypes.ScalarValue", FlextTypes.GeneralValueType, etc.)
    - Domain validation types (PortNumber, TimeoutSeconds, etc.)
    - JSON types (JsonPrimitive, JsonValue, etc.)
    - Handler types (HandlerCallable with dispatcher/bus-compatible aliases)
    - Utility types (SerializableType, MetadataDict, etc.)
    - CQRS pattern types (Command, Event, Message, Query)
    - Configuration models (RetryConfig)
    """

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
    type GeneralValueType = (
        FlextTypes.ScalarValue
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
    # Reuses FlextTypes.GeneralValueType (forward reference managed by __future__ annotations)
    type ObjectList = Sequence[FlextTypes.GeneralValueType]

    # Sortable object type - types that can be sorted (str, int, float, Mapping)
    # Note: Uses FlextTypes.ScalarValue but excludes bool/None for sorting compatibility
    type SortableObjectType = (
        str | int | float | Mapping[str, FlextTypes.SortableObjectType]
    )

    # Metadata-compatible attribute value type - composed for strict validation
    # Reuses FlextTypes.ScalarValue defined above (forward reference managed by __future__ annotations)
    type MetadataAttributeValue = (
        FlextTypes.ScalarValue
        | Sequence[FlextTypes.ScalarValue]
        | Mapping[str, FlextTypes.ScalarValue]
    )

    # Generic metadata dictionary type - read-only interface for metadata containers
    type Metadata = Mapping[str, MetadataAttributeValue]
    """Structural type for metadata dictionaries using Mapping (read-only interface).

    Represents flexible metadata containers without importing models.
    Used in protocol definitions where metadata must be stored or passed.
    This is a generic type available to all FLEXT projects.
    """

    # Flexible value type for protocol methods - contains scalars, sequences, or mappings
    type FlexibleValue = (
        FlextTypes.ScalarValue
        | Sequence[FlextTypes.ScalarValue]
        | Mapping[str, FlextTypes.ScalarValue]
    )

    # Mapping of string keys to flexible values
    type FlexibleMapping = Mapping[str, FlexibleValue]

    # =========================================================================
    # TOP-LEVEL JSON TYPE ALIASES (for backward compatibility with ecosystem)
    # =========================================================================
    # These aliases provide direct access to JSON types at FlextTypes level
    # instead of requiring FlextTypes.Json.JsonValue etc.

    # JSON value - primitive or complex value (alias to Json.JsonValue)
    type JsonValue = (
        str
        | int
        | float
        | bool
        | Sequence[GeneralValueType]
        | Mapping[str, GeneralValueType]
        | None
    )

    # JSON dictionary - mapping with string keys and JSON values
    # NOTE: Use FlextTypes.Json.JsonDict for JSON-specific operations
    # This top-level alias is kept for backward compatibility but prefer Json.JsonDict
    type JsonDict = Mapping[str, GeneralValueType]

    # NOTE: ParameterValueType was removed - use GeneralValueType directly
    # No aliases for convenience - use types directly per FLEXT standards

    class Utility:
        """Utility type definitions for type checking and introspection."""

        # Type hint specifier - used for type introspection
        # Can be any type hint: type, type alias, generic, or string
        # Note: For runtime type introspection, we accept type, str, or callable types
        type TypeHintSpecifier = type | str | Callable[..., FlextTypes.GeneralValueType]

        # Generic type argument - used for extracting generic type arguments
        # Can be a string type name or a type class representing FlextTypes.GeneralValueType
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class (forward reference)
        type GenericTypeArgument = str | type[FlextTypes.GeneralValueType]

        # Message type specifier - used for handler type checking
        # Can be a string type name or a type class
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class (forward reference)
        type MessageTypeSpecifier = str | type[FlextTypes.GeneralValueType]

        # Type origin specifier - used for generic type origin checking
        # Can be a string type name, type class, or callable with __origin__ attribute
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class (forward reference)
        type TypeOriginSpecifier = (
            str
            | type[FlextTypes.GeneralValueType]
            | Callable[..., FlextTypes.GeneralValueType]
        )

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
            """Validate hostname by attempting DNS resolution.

            Business Rule: Validates hostname strings by attempting DNS resolution
            using socket.gethostbyname(). Ensures hostnames are resolvable before
            being used in network configurations. Raises ValueError if hostname cannot
            be resolved, preventing invalid network configurations.

            Audit Implication: Hostname validation ensures network configurations
            are valid before being used in production systems. Failed validations
            are logged with error messages for audit trail completeness. Used by
            Pydantic 2 AfterValidator for type-safe hostname validation.

            Args:
                value: Hostname string to validate

            Returns:
                Validated hostname string (same as input if valid)

            Raises:
                ValueError: If hostname cannot be resolved via DNS

            """
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
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class (no duplication)
        type JsonValue = (
            JsonPrimitive
            | Sequence[FlextTypes.GeneralValueType]
            | Mapping[str, FlextTypes.GeneralValueType]
        )
        type JsonList = Sequence[JsonValue]
        type JsonDict = Mapping[str, FlextTypes.GeneralValueType]

    class HandlerAliases:
        """Handler and middleware type alias definitions for CQRS patterns.

        NOTE: Renamed from 'Handler' to avoid conflict with Handler TypeVars class.
        """

        # Single consolidated callable type for handlers and validators
        # Reuses FlextTypes.GeneralValueType from outer FlextTypes class
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
        # Reuses FlextTypes.GeneralValueType and FlextTypes.ScalarValue from parent FlextTypes class
        type AcceptableMessageType = (
            FlextTypes.GeneralValueType
            | Mapping[str, FlextTypes.GeneralValueType]
            | Sequence[FlextTypes.GeneralValueType]
            | FlextTypes.ScalarValue
        )

        # Conditional execution callable types (PEP 695)
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class (forward reference)
        type ConditionCallable = Callable[[FlextTypes.GeneralValueType], bool]

        # Variadic callable protocol is defined in FlextProtocols.VariadicCallable
        # Use that instead of redeclaring here

        # Handler type union - all possible handler representations
        # This is used for handler registries where handlers can be callables,
        # handler instances, or configuration dicts
        # Note: Handler and VariadicCallable protocols are defined in FlextProtocols
        # but cannot be imported here due to circular dependency. For type checking,
        # we use Callable as a fallback that works for most use cases.
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class (no duplication)
        type HandlerType = (
            HandlerCallable
            | Callable[..., FlextTypes.GeneralValueType]  # Variadic callable fallback
            | Mapping[str, FlextTypes.GeneralValueType]  # Configuration dict
        )

    class Config:
        """Configuration models for operational settings."""

        class RetryConfig(BaseModel):
            """Configuration for retry operations with exponential backoff.

            Business Rule: Defines retry behavior for operations that may fail and
            need automatic retry with configurable backoff strategies. Uses Pydantic 2
            BaseModel with frozen=True for immutability. All fields are validated at
            model instantiation, ensuring valid retry configurations.

            Audit Implication: Retry configuration is immutable after creation, ensuring
            consistent retry behavior throughout operation lifecycle. All retry attempts
            are logged with configuration details for audit trail completeness. Used by
            reliability systems for configurable retry patterns.

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
        """CQRS pattern types - use FlextTypes.GeneralValueType directly."""

        # CQRS types use FlextTypes.GeneralValueType directly - no aliases per FLEXT standards
        # For commands, events, messages, and queries, use FlextTypes.GeneralValueType directly

    class Processor:
        """Processor type definitions for pipeline processing."""

        # Processor callable type - takes input and returns processed output
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class
        type ProcessorCallable = Callable[
            [FlextTypes.GeneralValueType],
            FlextTypes.GeneralValueType,
        ]

        # Processor configuration type
        type ProcessorConfig = Mapping[str, FlextTypes.GeneralValueType]

    class Factory:
        """Factory type definitions for service creation patterns."""

        # Factory callable type - creates instances
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class
        type FactoryCallable = Callable[[], FlextTypes.GeneralValueType]

        # Factory with arguments callable type
        type FactoryWithArgsCallable = Callable[..., FlextTypes.GeneralValueType]

        # Factory configuration type
        type FactoryConfig = Mapping[str, FlextTypes.GeneralValueType]

    class Bus:
        """Message bus type definitions for event-driven patterns."""

        # Bus message type - uses GeneralValueType for flexibility
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class
        type BusMessage = FlextTypes.GeneralValueType

        # Bus handler callable type
        type BusHandler = Callable[[FlextTypes.GeneralValueType], None]

        # Bus subscription identifier
        type SubscriptionId = str

    class Logging:
        """Logging type definitions for structured logging."""

        # Log level type - string representation of log levels
        type LogLevel = str

        # Log context type - additional context for log messages
        type LogContext = Mapping[str, FlextTypes.GeneralValueType]

        # Log formatter callable type
        type LogFormatter = Callable[[str, LogContext], str]

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

        type DatabaseQueryResultMapping = Mapping[str, str | int]
        """Mapping for database query results (column names to values)."""

        type DIPatternsResultMapping = Mapping[str, int | Sequence[str] | str]
        """Mapping for dependency injection patterns demonstration results."""

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

        type NestedExceptionLevelMapping = Mapping[str, Mapping[type, str]]
        """Nested mapping for exception levels (library → exception_type → level)."""

        type ExceptionLevelMapping = Mapping[str, str]
        """Mapping for exception levels (container names to levels)."""

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

        type ExceptionMetricsMapping = Mapping[type, int]
        """Mapping for exception metrics (exception types to occurrence counts)."""

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

            Business Rule: TypedDict uses dict[str, ...] for field types because
            TypedDict defines the structure of a dictionary with known keys.
            This is correct - TypedDict fields must use dict, not Mapping.
            When used as function parameters/returns, prefer Mapping for read-only
            interface, but TypedDict itself correctly uses dict.

            Audit Implication: This metadata is used for service registration tracking.
            All fields are required (total=True), ensuring complete registration info.
            """

            handler_type: str
            """Type of handler (command, query, event, etc.)."""

            registration_status: str
            """Status of registration (active, inactive, error)."""

            metadata: dict[str, FlextTypes.GeneralValueType]
            """Additional service metadata (JSON-serializable).

            Business Rule: dict[str, GeneralValueType] is correct here because
            TypedDict fields must specify dict structure. This is not a type annotation
            for a function parameter/return, but a field definition within TypedDict.
            """

        class FactoryRegistrationMetadata(TypedDict, total=True):
            """Metadata for factory registration in the container.

            Used for tracking factory registration details.

            Business Rule: TypedDict uses dict[str, ...] for field types because
            TypedDict defines the structure of a dictionary with known keys.
            This is correct - TypedDict fields must use dict, not Mapping.

            Audit Implication: This metadata tracks factory registration state.
            All fields are required (total=True) for complete factory tracking.
            """

            factory_type: str
            """Type of factory."""

            registration_status: str
            """Status of registration (active, inactive, error)."""

            metadata: dict[str, FlextTypes.GeneralValueType]
            """Additional factory metadata (JSON-serializable).

            Business Rule: dict[str, GeneralValueType] is correct here because
            TypedDict fields must specify dict structure.
            """

        class ValidationContextDict(TypedDict, total=False):
            """Context dictionary for validation operations.

            Used for passing validation context data between validators.

            Business Rule: TypedDict uses dict[str, ...] for field types because
            TypedDict defines the structure of a dictionary with known keys.
            Fields are optional (total=False) to allow partial context passing.

            Audit Implication: Used for cross-field validation context.
            Optional fields allow flexible validation scenarios.
            """

            field_name: str
            """Name of the field being validated."""

            field_value: FlextTypes.GeneralValueType
            """Value of the field being validated."""

            parent_data: dict[str, FlextTypes.GeneralValueType]
            """Parent data context for cross-field validation.

            Business Rule: dict[str, GeneralValueType] is correct here because
            TypedDict fields must specify dict structure.
            """

        class EventDataDict(TypedDict, total=True):
            """Data dictionary for event objects.

            Used for structured event data in domain events.

            Business Rule: TypedDict uses dict[str, ...] for field types because
            TypedDict defines the structure of a dictionary with known keys.
            All fields required (total=True) for complete event representation.

            Audit Implication: Used for domain event serialization and storage.
            Complete event structure ensures audit trail completeness.
            """

            event_type: str
            """Type of event."""

            event_data: dict[str, FlextTypes.GeneralValueType]
            """Event payload data.

            Business Rule: dict[str, GeneralValueType] is correct here because
            TypedDict fields must specify dict structure.
            """

            metadata: dict[str, FlextTypes.GeneralValueType]
            """Event metadata (timestamp, correlation ID, etc.).

            Business Rule: dict[str, GeneralValueType] is correct here because
            TypedDict fields must specify dict structure.
            """

        class EntityDataDict(TypedDict, total=False):
            """Data dictionary for entity objects.

            Used for structured entity data representation.

            Business Rule: TypedDict uses dict[str, ...] for field types because
            TypedDict defines the structure of a dictionary with known keys.
            Fields are optional (total=False) for flexible entity representation.

            Audit Implication: Used for entity serialization and persistence.
            Optional fields allow partial entity updates.
            """

            entity_id: str
            """Unique entity identifier."""

            entity_type: str
            """Type classification of the entity."""

            data: dict[str, FlextTypes.GeneralValueType]
            """Entity data payload.

            Business Rule: dict[str, GeneralValueType] is correct here because
            TypedDict fields must specify dict structure.
            """

        class CategoryGroupDict(TypedDict, total=True):
            """Data dictionary for category groups.

            Business Rule: TypedDict uses dict[str, ...] for field types because
            TypedDict defines the structure of a dictionary with known keys.
            All fields required (total=True) for complete category representation.

            Audit Implication: Used for categorized data grouping in audit trails.
            Complete category structure ensures audit trail completeness for grouped
            data operations.

            Used for categorized data grouping.
            """

            category_name: str
            """Name of the category."""

            items: list[FlextTypes.GeneralValueType]
            """List of items in this category."""

        class DependencyContainerConfigDict(TypedDict, total=True):
            """Configuration dictionary for dependency injection containers.

            Used for dependency injection container configuration.

            Business Rule: TypedDict uses dict[str, ...] for field types because
            TypedDict defines the structure of a dictionary with known keys.
            All fields required (total=True) for complete container configuration.

            Audit Implication: Used for DI container service registration.
            Complete configuration ensures proper service lifecycle management.
            """

            service_name: str
            """Name of the service."""

            service_type: str
            """Type of service."""

            config: dict[str, FlextTypes.GeneralValueType]
            """Service configuration data.

            Business Rule: dict[str, GeneralValueType] is correct here because
            TypedDict fields must specify dict structure.
            """

            # DockerServiceConfigDict uses ContainerConfigDict directly - no aliases

        class _BatchResultDictBase(TypedDict, total=True):
            """Base TypedDict for batch processing operations.

            Business Rule: TypedDict uses dict[str, ...] for field types because
            TypedDict defines the structure of a dictionary with known keys.
            All fields required (total=True) for complete batch result representation.
            Note: TypedDict doesn't support generics directly, so results field uses
            list[GeneralValueType] for type flexibility. Type narrowing can be done
            at usage site based on operation return type.

            Audit Implication: Used for batch operation result tracking and auditing.
            Complete result structure ensures audit trail completeness for batch operations.
            """

            results: list[FlextTypes.GeneralValueType]
            """List of successful processing results.

            Business Rule: TypedDict fields must specify dict structure.
            Contains successfully processed items in order.
            Type is GeneralValueType for flexibility; actual type depends on operation.
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
        type BatchResultDict = _BatchResultDictBase

        # Note: TypedDict cannot be generic, so BatchResultDict is used directly
        # Type narrowing for results: list[T] is done at usage site based on operation return type
        # Users should use FlextTypes.Types.BatchResultDict directly

    class Example:
        """Example-specific types for demonstrating FLEXT features.

        Provides type-safe type aliases for example configurations and operations,
        using advanced Python 3.13+ patterns and PEP 695 syntax.
        """

        # Configuration data types using advanced mapping patterns
        # Reuses FlextTypes.ScalarValue from parent FlextTypes class (forward reference)
        type ConfigDataMapping = Mapping[str, FlextTypes.ScalarValue]
        """Mapping for configuration data (keys to scalar values)."""

        type EnvVarsMapping = Mapping[str, str]
        """Mapping for environment variables (variable names to string values)."""

        # Service metadata for configuration demonstrations
        type ConfigDemoMetadata = Mapping[str, Sequence[str] | Mapping[str, str]]
        """Metadata mapping for configuration demonstration results."""

        # Advanced validation types using Pydantic Field annotations
        type DatabaseUrl = Annotated[
            str,
            Field(
                min_length=10,
                pattern=r"^(sqlite|postgresql|mysql)://.*$",
                description="Database URL",
            ),
        ]
        type PortNumber = Annotated[
            int,
            Field(ge=1, le=65535, description="Network port"),
        ]
        type TimeoutSeconds = Annotated[
            float,
            Field(gt=0, le=300, description="Timeout in seconds"),
        ]
        type WorkerCount = Annotated[
            int,
            Field(ge=1, le=100, description="Number of workers"),
        ]

        # Configuration instance type using advanced union
        # Reuses FlextTypes.GeneralValueType from parent FlextTypes class (forward reference)
        type ConfigInstance = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping type for configuration instances."""

        # Validation result types for configuration operations
        type ValidationResult = tuple[bool, Sequence[str]]
        """Tuple of (is_valid, error_messages)."""

        # Environment configuration context
        type EnvConfigContext = Mapping[str, str | None]
        """Context mapping for environment variable configurations."""

        # Types for demonstration data
        # Reuses FlextTypes.ScalarValue and FlextTypes.GeneralValueType from parent FlextTypes class (forward references)
        type UserDataMapping = Mapping[str, FlextTypes.ScalarValue]
        """Mapping for user data in demonstrations."""

        type ValidationDataMapping = Mapping[str, Sequence[str]]
        """Mapping for validation data in demonstrations."""

        # Result metadata for demonstration outcomes
        type ResultMetadataMapping = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping for demonstration result metadata."""

        # Dependency injection service types
        type ServiceMetadataMapping = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping for service metadata in DI demonstrations."""

        type DatabaseQueryResultMapping = Mapping[str, FlextTypes.ScalarValue]
        """Mapping for database query results."""

        type ServiceConfigMapping = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping for service configuration data."""

        type DIPatternsResultMapping = Mapping[str, FlextTypes.GeneralValueType]
        """Mapping for DI patterns demonstration results."""


# NOTE: All TypeVars are defined at module level
# All complex types are defined in FlextTypes class only (no loose module-level aliases)
# Use FlextTypes.ScalarValue, FlextTypes.GeneralValueType, FlextTypes.ConstantValue directly
# Use FlextTypes.Example.ResultMetadataMapping, etc. for example-specific types


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
    "T_Settings",
    "T_co",
    "T_contra",
    "U",
    "V",
    "W",
]
