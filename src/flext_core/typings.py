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

from collections.abc import Callable
from typing import Literal, ParamSpec, TypeVar

# =============================================================================
# CENTRALIZED TYPE VARIABLES - All TypeVars for the entire FLEXT ecosystem
# =============================================================================

# Core generic type variables
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
R = TypeVar("R")
E = TypeVar("E")
F = TypeVar("F")
K = TypeVar("K")
P = ParamSpec("P")

# Covariant type variables (read-only)
T_co = TypeVar("T_co", covariant=True)
TResult_co = TypeVar("TResult_co", covariant=True)
TAggregate_co = TypeVar("TAggregate_co", covariant=True)
TEntity_co = TypeVar("TEntity_co", covariant=True)
TValueObject_co = TypeVar("TValueObject_co", covariant=True)
TDomainEvent_co = TypeVar("TDomainEvent_co", covariant=True)
TState_co = TypeVar("TState_co", covariant=True)
TValue_co = TypeVar("TValue_co", covariant=True)
TCacheValue_co = TypeVar("TCacheValue_co", covariant=True)
ResultT = TypeVar("ResultT")
TConfigValue_co = TypeVar("TConfigValue_co", covariant=True)

# Contravariant type variables (write-only)
T_contra = TypeVar("T_contra", contravariant=True)
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TCommand_contra = TypeVar("TCommand_contra", contravariant=True)
TQuery_contra = TypeVar("TQuery_contra", contravariant=True)
TEvent_contra = TypeVar("TEvent_contra", contravariant=True)
TKey_contra = TypeVar("TKey_contra", contravariant=True)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
TCacheKey_contra = TypeVar("TCacheKey_contra", contravariant=True)
TConfigKey_contra = TypeVar("TConfigKey_contra", contravariant=True)

# Domain-specific type variables
TCommand = TypeVar("TCommand")
TQuery = TypeVar("TQuery")
TEvent = TypeVar("TEvent")
TState = TypeVar("TState")
TKey = TypeVar("TKey")
TValue = TypeVar("TValue")
TEntity = TypeVar("TEntity")
TAggregate = TypeVar("TAggregate")
TDomainEvent = TypeVar("TDomainEvent")
TMessage = TypeVar("TMessage")
TResult = TypeVar("TResult")
MessageT = TypeVar("MessageT")

# Service and infrastructure type variables
TService = TypeVar("TService")
TCacheKey = TypeVar("TCacheKey")
TCacheValue = TypeVar("TCacheValue")
TConfigKey = TypeVar("TConfigKey")
TConfigValue = TypeVar("TConfigValue")
TResource = TypeVar("TResource")
TTimeout = TypeVar("TTimeout")

# Additional generic type variables
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
TConcurrent = TypeVar("TConcurrent")
TAccumulate = TypeVar("TAccumulate")
TParallel = TypeVar("TParallel")
UParallel = TypeVar("UParallel")
UResource = TypeVar("UResource")
TItem = TypeVar("TItem")
TUtil = TypeVar("TUtil")

# Additional generic type variables for ecosystem patterns
TProcessor = TypeVar("TProcessor")
THandler = TypeVar("THandler")
TBuilder = TypeVar("TBuilder")
TFactory = TypeVar("TFactory")


class FlextTypes:
    """Centralized type system namespace for the FLEXT ecosystem.

    Provides a comprehensive, organized type system following strict Flext standards:
    - Single source of truth for all type definitions
    - Python 3.13+ syntax with modern union types
    - Clean Architecture principles
    - No wrappers, aliases, or legacy patterns
    - Complete type safety and validation
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

        # Note: Use standard Optional[T] or T | None syntax directly

        # Basic collection types
        type Dict = dict[str, object]
        type List = list[object]
        type StringList = list[str]
        type IntList = list[int]
        type FloatList = list[float]
        type BoolList = list[bool]

        # Advanced collection types
        type NestedDict = dict[str, dict[str, object]]
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
        # Note: Using object for recursive types due to Python limitation
        type JsonValue = (
            str | int | float | bool | list[object] | dict[str, object] | None
        )
        type JsonObject = dict[str, JsonValue]
        type JsonArray = list[JsonValue]
        type JsonDict = dict[str, JsonValue]

        # Value types
        type Value = str | int | float | bool | object | None

        # Function types
        type Operation = Callable[[object], object]
        type Serializer = Callable[[object], dict[str, object]]
        type Validator = Callable[[object], bool]

        # gRPC target types (for flext-grpc domain usage)
        type GrpcTarget = str
        type GrpcStreamType = str
        type GrpcChannelState = str
        type GrpcServerState = str

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
        type FactoryDict = dict[str, Callable[[], object]]
        type ServiceType = Literal["instance", "factory"]

    # =========================================================================
    # CONFIG TYPES - Configuration and settings
    # =========================================================================

    class Config:
        """Configuration types."""

        type Environment = Literal[
            "development", "staging", "production", "testing", "test", "local"
        ]
        type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        type ConfigSerializer = Callable[[FlextTypes.Core.ConfigDict], str]

    # =========================================================================
    # VALIDATION TYPES - Validation patterns
    # =========================================================================

    class Validation:
        """Validation types."""

        type Validator = Callable[[object], bool]
        type ValidationRule = Callable[[object], bool]
        type BusinessRule = Callable[[object], bool]

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
            "library", "application", "service", "cli", "web", "api"
        ]
        type ProjectStatus = Literal["active", "inactive", "deprecated", "archived"]
        type ProjectConfig = dict[str, object]
        type WorkspaceStatus = Literal["initializing", "ready", "error", "maintenance"]

    # =========================================================================
    # PROCESSING TYPES - Generic processing patterns
    # =========================================================================

    class Processing:
        """Generic processing types for ecosystem patterns."""

        type ProcessingStatus = Literal[
            "pending", "running", "completed", "failed", "cancelled"
        ]
        type ProcessingMode = Literal["batch", "stream", "parallel", "sequential"]
        type ValidationLevel = Literal["strict", "lenient", "standard"]
        type ProcessingPhase = Literal["prepare", "execute", "validate", "complete"]
        type HandlerType = Literal["command", "query", "event", "processor"]
        type WorkflowStatus = Literal[
            "pending", "running", "completed", "failed", "cancelled"
        ]
        type StepStatus = Literal[
            "pending", "running", "completed", "failed", "skipped"
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
    # client-a TYPES - client-a Telecom specific types
    # =========================================================================

    class client-a:
        """client-a Telecom specific types."""

        type EnvironmentType = Literal[
            "development", "staging", "production", "testing"
        ]
        type ProfileType = Literal["user", "group", "organizational", "other"]
        type MigrationStatus = Literal["pending", "running", "completed", "failed"]
        type SyncStatus = Literal["pending", "running", "completed", "failed"]
        type OperationMode = Literal["migrate", "validate", "sync", "status"]
        type EntryCategory = Literal["user", "group", "organizational", "other"]
        type ProcessingPhase = Literal["01", "02", "03", "04", "05", "all"]
        type WorkflowStatus = Literal["pending", "running", "completed", "failed"]
        type StepStatus = Literal["pending", "running", "completed", "failed"]
        type LdapOperationType = Literal["search", "add", "modify", "delete"]
        type LdapScope = Literal["BASE", "LEVEL", "SUBTREE"]
        type EntryType = Literal["user", "group", "organizational", "other"]
        type ProcessingMode = Literal["batch", "stream", "parallel"]
        type ConversionStrategy = Literal["direct", "transform", "validate"]
        type ValidationLevel = Literal["strict", "lenient", "standard"]
        type SyncMode = Literal["incremental", "full", "validate"]
        type ConflictResolution = Literal["skip", "overwrite", "merge", "manual"]
        type ReportFormat = Literal["json", "yaml", "csv", "html", "text"]
        type Environment = Literal["development", "staging", "production", "testing"]
        type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        type ServiceOperationType = Literal[
            "create", "read", "update", "delete", "search"
        ]
        type ServiceStatus = Literal["active", "inactive", "maintenance", "error"]
        type ServiceMode = Literal["sync", "async", "batch", "stream"]
        type HandlerType = Literal["command", "query", "event", "notification"]
        type EventType = Literal["created", "updated", "deleted", "migrated"]
        type ProcessorType = Literal["parser", "validator", "transformer", "writer"]
        type OudConfig = dict[str, object]

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
# PUBLIC API EXPORTS - Essential TypeVars and types only
# =========================================================================

__all__: list[str] = [
    "T1",
    "T2",
    "T3",
    "E",
    "F",
    "FlextTypes",
    "K",
    "MessageT",
    "MessageT_contra",
    "P",
    "R",
    "T",
    "TAccumulate",
    "TAggregate",
    "TAggregate_co",
    "TBuilder",
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
    "TFactory",
    "THandler",
    "TItem",
    "TKey",
    "TKey_contra",
    "TMessage",
    "TParallel",
    "TProcessor",
    "TQuery",
    "TQuery_contra",
    "TResource",
    "TResult",
    "TResult_co",
    "TService",
    "TState",
    "TState_co",
    "TTimeout",
    "TUtil",
    "TValue",
    "TValueObject_co",
    "TValue_co",
    "T_co",
    "T_contra",
    "U",
    "UParallel",
    "UResource",
    "V",
    "W",
]
