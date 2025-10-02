"""Centralized type system for FLEXT ecosystem with Python 3.13+ patterns.

This module provides the complete type system for the FLEXT ecosystem,
centralizing all TypeVars and type aliases in a single location following
strict Flext standards and Clean Architecture principles.

Key Principles:
- Single source of truth for all types
- Python 3.13+ syntax with modern union types
- Pydantic 2 patterns and validation
- No wrappers, aliases, or legacy patterns
- Centralized TypeVars exported as public API

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable as CollectionsCallable
from typing import (
    Literal,
    ParamSpec,
    TypedDict,
    TypeVar,
)

# Basic type aliases at module level
Callable = CollectionsCallable[..., object]  # Generic callable for any signature

# =============================================================================
# CENTRALIZED TYPE VARIABLES - All TypeVars for the entire FLEXT ecosystem
# =============================================================================

# Core generic type variables
P = ParamSpec("P")

# Core TypeVars
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

# Plugin-specific TypeVars
TPlugin = TypeVar("TPlugin")
TPluginConfig = TypeVar("TPluginConfig")
TPluginContext = TypeVar("TPluginContext")
TPluginData = TypeVar("TPluginData")
TPluginDiscovery = TypeVar("TPluginDiscovery")
TPluginHandler = TypeVar("TPluginHandler")
TPluginLoader = TypeVar("TPluginLoader")
TPluginManager = TypeVar("TPluginManager")
TPluginMetadata = TypeVar("TPluginMetadata")
TPluginPlatform = TypeVar("TPluginPlatform")
TPluginRegistry = TypeVar("TPluginRegistry")
TPluginService = TypeVar("TPluginService")
TPluginSystem = TypeVar("TPluginSystem")
TPluginValidator = TypeVar("TPluginValidator")

# Covariant type variables (read-only)
T1_co = TypeVar("T1_co", covariant=True)
T2_co = TypeVar("T2_co", covariant=True)
T3_co = TypeVar("T3_co", covariant=True)

# Contravariant type variables (write-only)
TItem = TypeVar("TItem")
TItem_contra = TypeVar("TItem_contra", contravariant=True)
TResult = TypeVar("TResult")
TResult_contra = TypeVar("TResult_contra", contravariant=True)
TUtil = TypeVar("TUtil")
TUtil_contra = TypeVar("TUtil_contra", contravariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
MessageT = TypeVar("MessageT")
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
TCommand_contra = TypeVar("TCommand_contra", contravariant=True)
TEvent_contra = TypeVar("TEvent_contra", contravariant=True)
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TQuery_contra = TypeVar("TQuery_contra", contravariant=True)
TState_co = TypeVar("TState_co", covariant=True)
TState = TypeVar("TState")
E = TypeVar("E")
F = TypeVar("F")
K = TypeVar("K")
R = TypeVar("R")
# P already defined as ParamSpec above

# Type aliases - Python 3.13+ syntax
type WorkspaceStatus = Literal["initializing", "ready", "error", "maintenance"]

# Module-level type aliases for common types (can be imported directly)
type Dict = dict[str, object]
type List = list[object]

# Domain-specific type variables
Message = TypeVar("Message")
Command = TypeVar("Command")
Query = TypeVar("Query")
Event = TypeVar("Event")
ResultT = TypeVar("ResultT")
TCommand = TypeVar("TCommand")
TEvent = TypeVar("TEvent")
TQuery = TypeVar("TQuery")

# Service and infrastructure type variables
U = TypeVar("U")
V = TypeVar("V")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
TAccumulate = TypeVar("TAccumulate")
TAggregate = TypeVar("TAggregate")
TAggregate_co = TypeVar("TAggregate_co", covariant=True)
TCacheKey = TypeVar("TCacheKey")
TCacheKey_contra = TypeVar("TCacheKey_contra", contravariant=True)
TCacheValue = TypeVar("TCacheValue")
TCacheValue_co = TypeVar("TCacheValue_co", covariant=True)
W = TypeVar("W")
TConcurrent = TypeVar("TConcurrent")
TConfigKey = TypeVar("TConfigKey")
TConfigKey_contra = TypeVar("TConfigKey_contra", contravariant=True)
TConfigValue = TypeVar("TConfigValue")
TConfigValue_co = TypeVar("TConfigValue_co", covariant=True)
TDomainEvent = TypeVar("TDomainEvent")
TDomainEvent_co = TypeVar("TDomainEvent_co", covariant=True)
TEntity = TypeVar("TEntity")
TEntity_co = TypeVar("TEntity_co", covariant=True)
TKey = TypeVar("TKey")
TKey_contra = TypeVar("TKey_contra", contravariant=True)
TMessage = TypeVar("TMessage")
TParallel = TypeVar("TParallel")
TResource = TypeVar("TResource")
TResult_co = TypeVar("TResult_co", covariant=True)
TService = TypeVar("TService")
TTimeout = TypeVar("TTimeout")
TValue = TypeVar("TValue")
TValue_co = TypeVar("TValue_co", covariant=True)
TValueObject_co = TypeVar("TValueObject_co", covariant=True)
UParallel = TypeVar("UParallel")
UResource = TypeVar("UResource")


# =============================================================================
# TYPED DICTIONARIES - Structured data types
# =============================================================================


class RateLimiterState(TypedDict):
    """Rate limiter state tracking structure."""

    requests: list[float]
    last_reset: float


# =============================================================================
# FLEXT TYPES NAMESPACE - Centralized type system for the FLEXT ecosystem
# =============================================================================


class FlextTypes:
    """Centralized type system namespace for the FLEXT ecosystem.

    Provides comprehensive, organized type system following strict
    Flext standards with single source of truth for all definitions.

    **Function**: Type definitions for ecosystem-wide consistency
        - Core fundamental types (Dict, List, Headers)
        - Configuration types (ConfigValue, ConfigDict)
        - JSON types with Python 3.13+ syntax
        - Message types for CQRS patterns
        - Handler types for command/query handlers
        - Service types for domain services
        - Protocol types for interface definitions
        - Plugin types for extensibility
        - Generic type variables (T, U, V, W)
        - Covariant and contravariant type variables

    **Uses**: Core Python type system infrastructure
        - typing.TypeVar for generic type variables
        - typing.ParamSpec for parameter specifications
        - typing.Literal for literal type hints
        - collections.abc.Callable for callable types
        - Python 3.13+ union syntax (X | Y)
        - Modern type aliases with type keyword
        - Covariant and contravariant variance
        - Generic type constraints

    **How to use**: Type annotations and type safety
        ```python
        from flext_core import FlextTypes


        # Example 1: Use core dict type
        def process_data(data: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
            return {"processed": True, **data}


        # Example 2: Use configuration types
        def load_config() -> FlextTypes.Core.ConfigDict:
            return {
                "timeout": FlextConstants.Network.DEFAULT_TIMEOUT,
                "retries": FlextConstants.Reliability.DEFAULT_MAX_RETRIES,
            }


        # Example 3: Use headers type
        def build_headers(token: str) -> FlextTypes.Core.Headers:
            return {"Authorization": f"Bearer {token}"}


        # Example 4: Use message types for CQRS
        class CreateUserCommand(FlextTypes.Message.Command):
            email: str
            name: str


        # Example 5: Use handler types
        def create_handler(
            cmd: FlextTypes.Message.Command,
        ) -> FlextTypes.Handler.HandlerResult:
            return FlextResult[User].ok(User())


        # Example 6: Use generic type variables
        def transform[T](items: list[T]) -> list[T]:
            return [item for item in items]


        # Example 7: Use string lists
        tags: FlextTypes.Core.StringList = ["tag1", "tag2"]
        ```

    **TODO**: Enhanced type features for 1.0.0+ releases
        - [ ] Add type validation decorators
        - [ ] Implement runtime type checking utilities
        - [ ] Support type narrowing helpers
        - [ ] Add type transformation utilities
        - [ ] Implement type compatibility checking
        - [ ] Support type documentation generation
        - [ ] Add type migration tools
        - [ ] Implement type testing utilities
        - [ ] Support type versioning
        - [ ] Add type analysis and inspection

    Attributes:
        Core: Core fundamental types for ecosystem.
        Message: Message types for CQRS patterns.
        Handler: Handler types for command/query.
        Service: Service types for domain logic.
        Protocol: Protocol types for interfaces.
        Plugin: Plugin types for extensibility.

    Note:
        All types use Python 3.13+ modern syntax.
        Type variables support covariance/contravariance.
        Type aliases use modern type keyword syntax.
        No wrappers or legacy patterns allowed.
        Single source of truth for all ecosystem types.

    Warning:
        Type changes may impact entire ecosystem.
        Generic type variables must match usage patterns.
        Covariance/contravariance must be used correctly.
        Type aliases should not wrap existing types.

    Example:
        Complete type usage with generics:

        >>> def process[T](data: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
        ...     return data
        >>> result = process({"key": "value"})
        >>> print(result)
        {'key': 'value'}

    See Also:
        FlextResult: For result type patterns.
        FlextModels: For domain model types.
        FlextHandlers: For handler type usage.
        FlextProtocols: For protocol definitions.

    """

    # =========================================================================
    # CORE TYPES - Fundamental building blocks
    # =========================================================================

    class Core:
        """Core fundamental types used across all FLEXT modules."""

        # Generic type variables (re-exported for convenience)
        T = T
        U = U
        V = V
        W = W

        # Note: Use standard T | None syntax directly

        # Basic types - moved to module level

        # Basic collection types - simple type aliases (not using TypeAlias in class scope)
        Dict = dict[str, object]
        List = list[object]
        StringList = list[str]
        IntList = list[int]
        FloatList = list[float]
        BoolList = list[bool]

        # Advanced collection types
        NestedDict = dict[str, dict[str, object]]
        type Headers = dict[str, str]
        type Metadata = dict[str, str]
        type Parameters = dict[str, object]
        type CounterDict = dict[str, int]

        # Configuration types
        type ConfigValue = (
            str | int | float | bool | list[object] | dict[str, object] | None
        )
        type ConfigDict = dict[str, ConfigValue]

        # JSON types with modern Python 3.13+ syntax
        type JsonValue = (
            str | int | float | bool | list[object] | dict[str, object] | None
        )
        type JsonObject = dict[str, JsonValue]
        type JsonArray = list[JsonValue]
        type JsonDict = dict[str, JsonValue]

        # Value types
        type Value = str | int | float | bool | object | None

        # Collection types with ordering
        OrderedDict = OrderedDict[str, object]

    # =========================================================================
    # DOMAIN TYPES - Domain-Driven Design patterns
    # =========================================================================

    class Domain:
        """Domain types for DDD patterns."""

        type EntityId = str
        type Entity = object
        type ValueObject = object
        type AggregateRoot = object
        type DomainEvent = dict[str, object]

    # =========================================================================
    # SERVICE TYPES - Service layer patterns
    # =========================================================================

    class Service:
        """Service layer types."""

        type ServiceDict = dict[str, object]
        type FactoryDict = dict[str, CollectionsCallable[[], object]]
        type ServiceType = Literal["instance", "factory"]

    # =========================================================================
    # CONFIG TYPES - Configuration and settings
    # =========================================================================

    class Config:
        """Configuration types."""

        type Environment = Literal[
            "development",
            "staging",
            "production",
            "testing",
            "test",
            "local",
        ]
        type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        type ConfigSerializer = CollectionsCallable[
            [
                dict[
                    str,
                    str | int | float | bool | list[object] | dict[str, object] | None,
                ],
            ],
            str,
        ]

    # =========================================================================
    # VALIDATION TYPES - Validation patterns
    # =========================================================================

    class Validation:
        """Validation types."""

        type Validator = CollectionsCallable[[object], bool]
        type ValidationRule = CollectionsCallable[[object], bool]
        type BusinessRule = CollectionsCallable[[object], bool]

    # =========================================================================
    # OUTPUT TYPES - Generic output formatting types
    # =========================================================================

    class Output:
        """Generic output formatting types."""

        type OutputFormat = Literal["json", "yaml", "table", "csv", "text", "xml"]
        type SerializationFormat = Literal["json", "yaml", "toml", "ini", "xml"]
        type CompressionFormat = Literal["gzip", "bzip2", "xz", "lzma"]

    # =========================================================================
    # SERVICE ORCHESTRATION TYPES - Service orchestration patterns
    # =========================================================================

    class ServiceOrchestration:
        """Service orchestration types."""

        type ServiceOrchestrator = dict[str, object]
        type AdvancedServiceOrchestrator = dict[str, object]

    # =========================================================================
    # PROJECT TYPES - Project management types
    # =========================================================================

    class Project:
        """Project management types."""

        type ProjectType = Literal[
            "library",
            "application",
            "service",
            "cli",
            "web",
            "api",
            "PYTHON",
            "GO",
            "JAVASCRIPT",
        ]
        type ProjectStatus = Literal["active", "inactive", "deprecated", "archived"]
        type ProjectConfig = dict[str, object]

    # =========================================================================
    # PROCESSING TYPES - Generic processing patterns
    # =========================================================================

    class Processing:
        """Generic processing types for ecosystem patterns."""

        type ProcessingStatus = Literal[
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        ]
        type ProcessingMode = Literal["batch", "stream", "parallel", "sequential"]
        type ValidationLevel = Literal["strict", "lenient", "standard"]
        type ProcessingPhase = Literal["prepare", "execute", "validate", "complete"]
        type HandlerType = Literal["command", "query", "event", "processor"]
        type WorkflowStatus = Literal[
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        ]
        type StepStatus = Literal[
            "pending",
            "running",
            "completed",
            "failed",
            "skipped",
        ]

    # =========================================================================
    # IDENTIFIER TYPES - Common identifier patterns
    # =========================================================================

    class Identifiers:
        """Common identifier and path types used across FLEXT modules."""

        type Id = str
        type Name = str
        type Path = str
        type Uri = str
        type Token = str

    # =========================================================================
    # PROTOCOL TYPES - Generic protocol patterns
    # =========================================================================

    class Protocol:
        """Generic protocol type definitions."""

        type ProtocolVersion = str
        type ConnectionString = str
        type AuthCredentials = dict[str, str]
        type ProtocolConfig = dict[str, object]
        type MessageFormat = Literal["json", "xml", "binary", "text"]

    # =========================================================================
    # CONVENIENCE ALIASES - Direct access to commonly used types
    # =========================================================================

    # Direct access to Core types for convenience
    ConfigValue = Core.ConfigValue
    JsonValue = Core.JsonValue
    ConfigDict = Core.ConfigDict
    JsonDict = Core.JsonDict
    type Headers = Core.Headers
    type Metadata = Core.Metadata
    type Parameters = Core.Parameters


# =========================================================================
# PUBLIC API EXPORTS - Essential TypeVars and types only
# =========================================================================

__all__: list[str] = [
    # Core TypeVars
    "T1",
    "T2",
    "T3",
    "Command",
    "E",
    "Event",
    "F",
    # Main classes
    "FlextTypes",
    "K",
    # Domain TypeVars
    "Message",
    "MessageT",
    "MessageT_contra",
    "P",
    "Query",
    "R",
    "RateLimiterState",
    "ResultT",
    "T",
    "T1_co",
    "T2_co",
    "T3_co",
    "TAccumulate",
    "TAggregate",
    "TAggregate_co",
    "TCacheKey",
    "TCacheKey_contra",
    "TCacheValue",
    "TCacheValue_co",
    "TCommand",
    "TCommand_contra",
    "TConcurrent",
    "TConfigKey",
    "TConfigKey_contra",
    "TConfigValue",
    "TConfigValue_co",
    "TDomainEvent",
    "TDomainEvent_co",
    "TEntity",
    "TEntity_co",
    "TEvent",
    "TEvent_contra",
    "TInput_contra",
    "TItem",
    "TItem_contra",
    "TKey",
    "TKey_contra",
    "TMessage",
    "TParallel",
    "TQuery",
    "TQuery_contra",
    "TResource",
    "TResult",
    "TResult_co",
    "TResult_contra",
    "TService",
    "TState",
    "TState_co",
    "TTimeout",
    "TUtil",
    "TUtil_contra",
    "TValue",
    "TValueObject_co",
    "TValue_co",
    "T_contra",
    "U",
    "UParallel",
    "UResource",
    "V",
    "W",
    "WorkspaceStatus",
]
