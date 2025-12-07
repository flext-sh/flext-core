"""Type aliases and generics used across dispatcher-ready components.

``t`` centralizes ``TypeVar`` declarations and nested namespaces of
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
from pathlib import Path
from re import Pattern
from types import ModuleType
from typing import (
    Annotated,
    ClassVar,
    ParamSpec,
    TypedDict,
    TypeVar,
)

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.constants import c

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
# FLEXT TYPES - All complex types defined inside t class
# ============================================================================
# NOTE: Only TypeVars are defined at module level (see above)
# All complex type aliases MUST be inside t class - NO loose types!


class FlextTypes:
    """Type system foundation for FLEXT ecosystem - Complex Type Aliases Only.

    This class serves as the centralized type system for all FLEXT projects.
    All complex types (type aliases, TypedDict classes) are organized in nested
    namespaces within this class. TypeVars are defined at module level (see above).

    **Architecture Principles:**
    - TypeVars: Defined at module level (Python limitation)
    - Complex types: All in FlextTypes nested namespaces
    - No loose types: All type aliases must be in appropriate namespaces
    - Extension: Other projects extend via composition, not modification

    **Namespace Structure:**

    - **Top-level types**: Core scalar and value types (ScalarValue, GeneralValueType,
      ConstantValue, ObjectList, SortableObjectType, MetadataAttributeValue, Metadata,
      FlexibleValue, FlexibleMapping, JsonValue, JsonDict)

    - **Utility**: Type introspection helpers (TypeHintSpecifier, GenericTypeArgument,
      MessageTypeSpecifier, TypeOriginSpecifier)

    - **Validation**: Domain validation types with Pydantic Field annotations
      (PortNumber, TimeoutSeconds, RetryCount, NonEmptyStr, HostName)

    - **Json**: JSON serialization types (JsonPrimitive, JsonValue, JsonList, JsonDict)

    - **Handler**: Handler and middleware type definitions for CQRS patterns
      (HandlerCallable, MiddlewareConfig, AcceptableMessageType, ConditionCallable,
      HandlerType)

    - **Config**: Configuration type aliases (use m.Config.* for model classes)

    - **Dispatcher**: Dispatcher type definitions for message dispatching
      (DispatcherConfig TypedDict)

    - **Types**: TypedDict classes and mapping type aliases
      - Mapping aliases: ServiceMetadataMapping, FieldMetadataMapping,
        SummaryDataMapping, CategoryGroupsMapping, SharedContainersMapping,
        EventDataMapping, ContextMetadataMapping, ConfigurationMapping,
        FieldValidatorMapping, ConsistencyRuleMapping,
        EventValidatorMapping, ErrorTypeMapping, ExceptionKwargsType
      - TypedDict classes: ContainerConfigDict, BatchResultDict


    **Guidelines for Adding New Types:**

    1. **TypeVars**: Add at module level with clear category comments
    2. **Domain validation types**: Add to Validation namespace
    3. **JSON types**: Add to Json namespace
    4. **Handler types**: Add to Handler namespace
    5. **Configuration types**: Add to Config namespace (BaseModel) or Types namespace (TypedDict)
    6. **Dispatcher types**: Add to Dispatcher namespace
    7. **TypedDict classes**: Add to Types namespace
    8. **Mapping type aliases**: Add to Types namespace

    **Extending FlextTypes in Other Projects:**

    Other FLEXT projects should extend FlextTypes via composition, not modification:

    ```python
    from flext_core.typings import t


    class MyProjectTypes:
        # Extend via composition
        type MyCustomType = t.GeneralValueType | MySpecificType
        # Use base types from t
        type MyConfig = t.Types.ConfigurationMapping
    ```

    Do NOT modify FlextTypes directly. Instead, create project-specific type namespaces
    that reference FlextTypes types via composition.
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
        ScalarValue | Sequence[GeneralValueType] | Mapping[str, GeneralValueType]
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
    # Reuses t.GeneralValueType (forward reference managed by __future__ annotations)
    type ObjectList = Sequence[GeneralValueType]

    # Sortable object type - types that can be sorted (str, int, float, Mapping)
    # Note: Uses t.ScalarValue but excludes bool/None for sorting compatibility
    type SortableObjectType = str | int | float | Mapping[str, SortableObjectType]

    # Metadata-compatible attribute value type - composed for strict
    # validation
    # Reuses t.ScalarValue defined above (forward reference managed by
    # __future__ annotations)
    type MetadataAttributeValue = (
        ScalarValue | Sequence[ScalarValue] | Mapping[str, ScalarValue]
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
    type FlexibleValue = ScalarValue | Sequence[ScalarValue] | Mapping[str, ScalarValue]

    # Mapping of string keys to flexible values
    type FlexibleMapping = Mapping[str, FlexibleValue]

    # =========================================================================
    # TOP-LEVEL JSON TYPE ALIASES
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

    # NOTE: Use t.Json.JsonDict for JSON-specific operations

    # NOTE: ParameterValueType was removed - use GeneralValueType directly
    # No aliases for convenience - use types directly per FLEXT standards

    class Utility:
        """Utility type definitions for type checking and introspection."""

        # Type hint specifier - used for type introspection
        # Can be any type hint: type, type alias, generic, or string
        # Note: For runtime type introspection, we accept type, str,
        # or callable types
        # Note: Removed Callable[..., T] to avoid Any - use specific
        # callable types instead
        type TypeHintSpecifier = (
            type | str | Callable[[t.GeneralValueType], t.GeneralValueType]
        )

        # Generic type argument - used for extracting generic type arguments
        # Can be a string type name or a type class representing t.GeneralValueType
        # Reuses t.GeneralValueType from parent t class (forward reference)
        type GenericTypeArgument = str | type[t.GeneralValueType]

        # Message type specifier - used for handler type checking
        # Can be a string type name or a type class
        # Reuses t.GeneralValueType from parent t class (forward reference)
        type MessageTypeSpecifier = str | type[t.GeneralValueType]

        # Type origin specifier - used for generic type origin checking
        # Can be a string type name, type class, or callable with __origin__ attribute
        # Reuses t.GeneralValueType from parent t class (forward reference)
        # Note: Removed Callable[..., T] to avoid Any - use specific callable types instead
        type TypeOriginSpecifier = (
            str
            | type[t.GeneralValueType]
            | Callable[[t.GeneralValueType], t.GeneralValueType]
        )

    class Validation:
        """Domain validation types using Pydantic Field annotations."""

        # Network validation types
        # NOTE: Field() requires literal values, so we use constants directly
        # These constants are centralized in FlextConstants for reuse
        type PortNumber = Annotated[
            int,
            Field(
                ge=c.Network.MIN_PORT,
                le=c.Network.MAX_PORT,
                description="Network port",
            ),
        ]
        type TimeoutSeconds = Annotated[
            float,
            Field(
                gt=c.ZERO,
                le=int(c.Network.DEFAULT_TIMEOUT),
                description="Timeout in seconds",
            ),
        ]
        type RetryCount = Annotated[
            int,
            Field(
                ge=c.ZERO,
                le=c.Validation.RETRY_COUNT_MAX,
                description="Retry attempts",
            ),
        ]

        # String validation types
        type NonEmptyStr = Annotated[
            str,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Non-empty string",
            ),
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
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                max_length=c.Network.MAX_HOSTNAME_LENGTH,
                description="Valid hostname",
            ),
            AfterValidator(_validate_hostname),
        ]

    class Json:
        """JSON serialization types for API and data exchange."""

        # Primitive JSON types
        type JsonPrimitive = str | int | float | bool | None

        # Complex JSON types using recursive t.GeneralValueType
        # Reuses t.GeneralValueType from parent t class (no duplication)
        type JsonValue = (
            JsonPrimitive
            | Sequence[t.GeneralValueType]
            | Mapping[str, t.GeneralValueType]
        )
        type JsonList = Sequence[JsonValue]
        type JsonDict = Mapping[str, t.GeneralValueType]

    class Handler:
        """Handler and middleware type definitions for CQRS patterns."""

        # Single consolidated callable type for handlers and validators
        # Reuses t.GeneralValueType from outer t class
        type HandlerCallable = Callable[
            [t.GeneralValueType],
            t.GeneralValueType,
        ]
        # Middleware uses same callable signature as handlers
        # Use HandlerCallable directly - no aliases per FLEXT standards

        # Handler type for registry - union of all possible handler types
        # This includes callables, objects with handle() method, and handler instances
        # Note: Handler protocol is defined in p.Application.Handler for proper protocol organization

        # Middleware configuration type
        type MiddlewareConfig = t.Types.ConfigurationMapping

        # Acceptable message types for handlers - union of common message types
        # Reuses t.GeneralValueType and t.ScalarValue from parent t class
        type AcceptableMessageType = (
            t.GeneralValueType
            | Mapping[str, t.GeneralValueType]
            | Sequence[t.GeneralValueType]
            | t.ScalarValue
        )

        # Conditional execution callable types (PEP 695)
        # Reuses t.GeneralValueType from parent t class (forward reference)
        type ConditionCallable = Callable[[t.GeneralValueType], bool]

        # Variadic callable protocol is defined in p.Utility.Callable
        # Use that instead of redeclaring here

        # Handler type union - all possible handler representations
        # This is used for handler registries where handlers can be callables,
        # handler instances, or configuration dicts
        # Note: Handler and Utility.Callable protocols are defined in p
        # but cannot be imported here due to circular dependency. For type checking,
        # we use HandlerCallable which covers most use cases.
        # Reuses t.GeneralValueType from parent t class (no duplication)
        # Note: Removed Callable[..., T] to avoid Any - HandlerCallable covers variadic cases
        type HandlerType = (
            HandlerCallable | Mapping[str, t.GeneralValueType]  # Configuration dict
        )

    class Config:
        """Configuration models for operational settings.

        Note: BaseModel classes have been moved to models.py
        per FLEXT architecture standards. Use m.Config.* instead.
        This namespace now only contains type aliases, not model classes.
        """

    class Dispatcher:
        """Dispatcher type definitions for message dispatching and processing."""

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

        **TypedDict Classes**:
        - ContainerConfigDict: Container configuration
        - DockerServiceConfigDict: Docker service configuration
        """

        # =====================================================================
        # MAPPING TYPE ALIASES (Python 3.13+ PEP 695)
        # =====================================================================
        # Using PEP 695 type keyword for better type checking and IDE support
        type ServiceMetadataMapping = Mapping[str, t.GeneralValueType]
        """Mapping for service metadata (attribute names to values)."""

        type FieldMetadataMapping = Mapping[str, t.GeneralValueType]
        """Mapping for field metadata (field names to metadata objects)."""

        type SummaryDataMapping = Mapping[str, int | float | str]
        """Mapping for summary data (category names to summary values)."""

        type CategoryGroupsMapping = Mapping[str, Sequence[t.GeneralValueType]]
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

        type EventDataMapping = Mapping[str, t.GeneralValueType]
        """Mapping for event data (event properties to values)."""

        type ContextMetadataMapping = Mapping[str, t.GeneralValueType]
        """Mapping for context metadata (context properties to values)."""

        type ConfigurationMapping = Mapping[str, t.GeneralValueType]
        """Mapping for configuration (configuration keys to values)."""

        type ConfigurationDict = dict[str, t.GeneralValueType]
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

        type StringConfigurationDictDict = dict[str, dict[str, t.GeneralValueType]]
        """Mutable nested dict for string-to-configuration-dict (e.g., scoped contexts)."""

        type StringListMapping = Mapping[str, list[t.GeneralValueType]]
        """Mapping for string-to-list associations (e.g., categories)."""

        type StringListDict = dict[str, list[t.GeneralValueType]]
        """Mutable dict for string-to-list associations (e.g., categories)."""

        type StringSequenceMapping = Mapping[str, Sequence[str]]
        """Mapping for string-to-string-sequence associations."""

        type StringSequenceDict = dict[str, Sequence[str]]
        """Mutable dict for string-to-string-sequence associations."""

        type StringListSequenceDict = dict[str, list[str]]
        """Mutable dict for string-to-string-list associations (e.g., hooks)."""

        type MetadataAttributeDict = dict[str, t.MetadataAttributeValue]
        """Mutable dict for metadata attributes (string keys to metadata-compatible values)."""

        type StringGenericTypeArgumentTupleDict = dict[
            str,
            tuple[t.Utility.GenericTypeArgument, ...],
        ]
        """Mutable dict for string-to-generic-type-argument-tuple mappings."""

        type GeneralValueDict = dict[str, t.GeneralValueType]
        """Mutable dict for general value types (alias for ConfigurationDict)."""

        type FloatListDict = dict[str, list[float]]
        """Mutable dict for string-to-float-list associations (e.g., execution times)."""

        type HandlerTypeDict = dict[str, t.Handler.HandlerType]
        """Mutable dict for handler type mappings."""

        type HandlerCallableDict = dict[str, t.Handler.HandlerCallable]
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

        type StringHandlerCallableListDict = dict[str, list[t.Handler.HandlerCallable]]
        """Mutable dict for string-to-handler-callable-list mappings (e.g., hooks)."""

        type StringBaseSettingsTypeDict = dict[str, type[BaseSettings]]
        """Mutable dict for string-to-BaseSettings-type mappings (e.g., namespace registry)."""

        type StringSequenceGeneralValueMapping = Mapping[
            str,
            Sequence[t.GeneralValueType],
        ]
        """Mapping for string-to-sequence-of-GeneralValueType associations."""

        type StringSequenceGeneralValueDict = dict[str, Sequence[t.GeneralValueType]]
        """Mutable dict for string-to-sequence-of-GeneralValueType associations."""

        type StringTupleFloatIntDict = dict[str, tuple[float, int]]
        """Mutable dict for string-to-(float, int)-tuple mappings (e.g., rate limit windows)."""

        type StringTupleGeneralValueDict = dict[
            str,
            tuple[t.GeneralValueType, t.GeneralValueType],
        ]
        """Mutable dict for string-to-(GeneralValueType, GeneralValueType)-tuple mappings (e.g., differences)."""

        # Container-specific types (forward references to avoid circular imports)
        type ServiceRegistrationDict = dict[str, object]
        """Mutable dict for service registration mappings.

        Note: Uses 'object' as base type to allow any service registration type
        (typically m.Container.ServiceRegistration) without circular import.
        """

        type FactoryRegistrationDict = dict[str, object]
        """Mutable dict for factory registration mappings.

        Note: Uses 'object' as base type to allow any factory registration type
        (typically m.Container.FactoryRegistration) without circular import.
        """

        # Additional mapping types for validation and exceptions
        # Validators return Result-like objects with is_failure/error attributes
        # Using object to avoid circular import with protocols.py
        type FieldValidatorMapping = Mapping[
            str,
            Callable[[t.GeneralValueType], object],
        ]
        """Mapping for field validators (field names to validator returning Result-like)."""

        type ConsistencyRuleMapping = Mapping[
            str,
            Callable[[t.GeneralValueType], object],
        ]
        """Mapping for consistency rules (rule names to validator returning Result-like)."""

        type ResourceRegistrationDict = dict[str, object]
        """Mutable dict for resource registration mappings.

        Uses object to avoid circular imports (typically ResourceRegistration).
        """

        type EventValidatorMapping = Mapping[
            str,
            Callable[[t.GeneralValueType], object],
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
            t.ScalarValue
            | Sequence[t.ScalarValue]
            | Mapping[
                str,
                t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue],
            ]
        )
        """Type alias for exception kwargs - flexible value handling with proper parametrization."""

        # =====================================================================
        # TYPEDDICT CLASSES (Python 3.13+ PEP 695)
        # =====================================================================
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

            results: list[t.GeneralValueType]
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
        # Users should use t.Types.BatchResultDict directly

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
            """Config type for service runtime (defaults to FlextConfig)."""

            config_overrides: Mapping[str, t.FlexibleValue]
            """Configuration overrides to apply to the config instance."""

            context: object
            """Context instance (p.Context.Ctx) for service runtime.

            Note: Uses object to avoid circular import with protocols.py.
            Actual type is p.Context.Ctx, but protocols cannot be imported here.
            """

            subproject: str
            """Subproject identifier for scoped container creation."""

            services: Mapping[
                str,
                t.GeneralValueType | BaseModel | object,
            ]
            """Service registrations for container.

            Note: Uses object for p.Utility.Callable[t.GeneralValueType] to avoid
            circular import with protocols.py. Actual type includes callables.
            """

            factories: Mapping[
                str,
                Callable[
                    [],
                    (
                        t.ScalarValue
                        | Sequence[t.ScalarValue]
                        | Mapping[str, t.ScalarValue]
                    ),
                ],
            ]
            """Factory registrations for container."""

            resources: Mapping[str, Callable[[], t.GeneralValueType]]
            """Resource registrations for container."""

            container_overrides: Mapping[str, t.FlexibleValue]
            """Container configuration overrides."""

            wire_modules: Sequence[ModuleType]
            """Modules to wire for dependency injection."""

            wire_packages: Sequence[str]
            """Packages to wire for dependency injection."""

            wire_classes: Sequence[type]
            """Classes to wire for dependency injection."""

    # =========================================================================
    # CORE NAMESPACE - For cross-project access pattern consistency
    # =========================================================================
    # Provides a .Core. namespace class for accessing core types from other projects
    # Similar pattern to how flext-ldap uses .Ldif. namespace to access flext-ldif types
    # This enables consistent namespace patterns across the FLEXT ecosystem

    class Core:
        """Core types namespace for cross-project access.

        Provides organized access to all core types for other FLEXT projects.
        Usage: Other projects can reference `t.Core.ScalarValue`, `t.Core.Json.JsonValue`, etc.
        This enables consistent namespace patterns (e.g., `.Core.`, `.Ldif.`, `.Ldap.`).
        """

        # Core scalar and value types
        type ScalarValue = FlextTypes.ScalarValue
        type GeneralValueType = FlextTypes.GeneralValueType
        type ConstantValue = FlextTypes.ConstantValue
        type ObjectList = FlextTypes.ObjectList
        type SortableObjectType = FlextTypes.SortableObjectType
        type MetadataAttributeValue = FlextTypes.MetadataAttributeValue
        type Metadata = FlextTypes.Metadata
        type FlexibleValue = FlextTypes.FlexibleValue
        type FlexibleMapping = FlextTypes.FlexibleMapping
        type JsonValue = FlextTypes.JsonValue

        # Nested namespace classes - defined as ClassVar for proper type checking
        # These are assigned after class definition to avoid forward reference issues
        # Note: With from __future__ import annotations, these are automatically strings
        Utility: ClassVar[type[FlextTypes.Utility]]
        Validation: ClassVar[type[FlextTypes.Validation]]
        Json: ClassVar[type[FlextTypes.Json]]
        Handler: ClassVar[type[FlextTypes.Handler]]
        Config: ClassVar[type[FlextTypes.Config]]
        Dispatcher: ClassVar[type[FlextTypes.Dispatcher]]
        Types: ClassVar[type[FlextTypes.Types]]

    # =========================================================================
    # CORE NAMESPACE - Access nested classes (defined after Core class)
    # =========================================================================
    # Note: These are assigned after all nested classes are defined
    # to enable .Core. namespace pattern for cross-project access

    # =========================================================================
    # ROOT-LEVEL ALIASES (Minimize nesting for common types)
    # Usage: t.PortNumber instead of t.Validation.PortNumber
    # Both access patterns work - aliases for convenience, namespaces for clarity
    # =========================================================================

    # Validation aliases (most commonly used)
    PortNumber = Validation.PortNumber
    TimeoutSeconds = Validation.TimeoutSeconds
    RetryCount = Validation.RetryCount
    NonEmptyStr = Validation.NonEmptyStr
    HostName = Validation.HostName

    # Json aliases
    JsonPrimitive = Json.JsonPrimitive
    JsonList = Json.JsonList
    JsonDict = Json.JsonDict

    # Handler aliases
    HandlerCallable = Handler.HandlerCallable
    MiddlewareConfig = Handler.MiddlewareConfig
    AcceptableMessageType = Handler.AcceptableMessageType
    ConditionCallable = Handler.ConditionCallable

    # Types aliases (frequently used mappings)
    StringDict = Types.StringDict
    StringMapping = Types.StringMapping
    ConfigurationDict = Types.ConfigurationDict
    ConfigurationMapping = Types.ConfigurationMapping
    StringIntDict = Types.StringIntDict
    StringBoolDict = Types.StringBoolDict
    GeneralValueDict = Types.GeneralValueDict
    BatchResultDict = Types.BatchResultDict
    ContainerConfigDict = Types.ContainerConfigDict

    # ClassVar annotations in Core class ensure proper type checking
    Core.Utility = Utility
    Core.Validation = Validation
    Core.Json = Json
    Core.Handler = Handler
    Core.Config = Config
    Core.Dispatcher = Dispatcher
    Core.Types = Types


t = FlextTypes
t_core = FlextTypes

__all__ = [
    "E",
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
