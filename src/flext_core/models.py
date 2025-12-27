"""DDD base models with Pydantic v2 validation and dispatcher-first CQRS.

Expose ``FlextModels`` as the façade for entities, value objects, aggregates,
commands, queries, and domain events that integrate directly with the
dispatcher-driven CQRS layer. Concrete implementations live in the
``_models`` subpackage and are organized for clear validation, serialization,
and event collection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Discriminator

from flext_core._models.base import FlextModelsBase
from flext_core._models.collections import FlextModelsCollections
from flext_core._models.container import FlextModelsContainer
from flext_core._models.context import FlextModelsContext
from flext_core._models.cqrs import FlextModelsCqrs
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.handler import FlextModelsHandler
from flext_core._models.settings import FlextModelsConfig
from flext_core._models.validation import FlextModelsValidation
from flext_core.protocols import p


class FlextModels:
    """Facade that groups DDD building blocks for CQRS-ready domains.

    Architecture: Domain layer helper
    Provides strongly typed base classes for entities, aggregates, commands,
    queries, and domain events so dispatcher handlers can enforce invariants,
    collect domain events, and validate inputs through Pydantic v2.

    Core concepts
    - Entity: Domain object with identity and lifecycle controls.
    - Value: Immutable value objects for pure operations.
    - AggregateRoot: Consistency boundary that aggregates events.
    - Command/Query: Message shapes consumed by dispatcher handlers.
    - DomainEvent: Stored and published through dispatcher pipelines.

    Pydantic v2 integration supplies BaseModel validation, computed fields,
    and JSON-ready serialization for all exported types.
    """

    # =========================================================================
    # CORE DOMAIN ENTITIES - Inheritable base classes
    # =========================================================================

    class Entity(FlextModelsEntity.Entry):
        """Entity base class - domain objects with identity."""

    class ValueObject(FlextModelsEntity.Value):
        """Value object base class - immutable, compared by value."""

    class AggregateRoot(FlextModelsEntity.AggregateRoot):
        """Aggregate root base class - consistency boundary."""

    class DomainEvent(FlextModelsEntity.DomainEvent):
        """Domain event base class - published through dispatcher."""

    # =========================================================================
    # GENERIC MODELS BY BUSINESS FUNCTION - Inheritable base classes
    # =========================================================================

    class Value(FlextModelsEntity.Value):
        """Value objects - immutable data compared by value.

        Inherits frozen=True, extra="forbid" from FlextModelsBase.FrozenStrictModel.
        """

    class Snapshot(FlextModelsBase.FrozenStrictModel):
        """Snapshots - state captured at a specific moment.

        Inherits frozen=True, extra="forbid" from FlextModelsBase.FrozenStrictModel.
        """

    class Progress(FlextModelsBase.ArbitraryTypesModel):
        """Progress trackers - mutable accumulators during operations.

        Inherits validate_assignment=True from FlextModelsBase.ArbitraryTypesModel.
        """

    # =========================================================================
    # NAMESPACE CLASSES - Direct access for internal model classes
    # =========================================================================

    Base = FlextModelsBase
    Cqrs = FlextModelsCqrs
    EntityModels = FlextModelsEntity
    Entity_ns = FlextModelsEntity
    ContextModels = FlextModelsContext
    Context = FlextModelsContext
    HandlerModels = FlextModelsHandler
    Handler_ns = FlextModelsHandler
    ValidationModels = FlextModelsValidation
    Validation = FlextModelsValidation

    # =========================================================================
    # CQRS MESSAGING - Direct access for common usage
    # =========================================================================

    Command: TypeAlias = FlextModelsCqrs.Command
    Query: TypeAlias = FlextModelsCqrs.Query
    Bus: TypeAlias = FlextModelsCqrs.Bus
    Pagination: TypeAlias = FlextModelsCqrs.Pagination

    # =========================================================================
    # AUTH DOMAIN MODELS
    # =========================================================================

    class Identity(FlextModelsEntity.AggregateRoot):
        """User identity aggregate root for authentication domain.

        Represents a user account with authentication credentials,
        roles, and session management capabilities.
        """

        name: str
        contact: str
        credential: str
        roles: list[str] | None = None
        metadata: dict[str, str] | None = None

    class IdentityRequest(FlextModelsCqrs.Command):
        """Command for identity operations in auth domain."""

        name: str
        contact: str
        credential: str
        roles: list[str] | None = None
        metadata: dict[str, str] | None = None

    # =========================================================================
    # CONFIGURATION MODELS - Direct access for common usage
    # =========================================================================

    Config = FlextModelsConfig
    ProcessingRequest: TypeAlias = FlextModelsConfig.ProcessingRequest
    ProcessingConfig: TypeAlias = FlextModelsConfig.ProcessingRequest
    BatchProcessingConfig: TypeAlias = FlextModelsConfig.BatchProcessingConfig
    ValidationConfiguration: TypeAlias = FlextModelsConfig.ValidationConfiguration
    HandlerRegistration: TypeAlias = FlextModelsHandler.Registration
    HandlerExecutionConfig: TypeAlias = FlextModelsConfig.HandlerExecutionConfig

    # =========================================================================
    # SERVICE MODELS
    # =========================================================================

    class ServiceRuntime(FlextModelsBase.ArbitraryTypesModel):
        """Runtime triple (config, context, container) for services.

        Represents the core service runtime with configuration, context,
        and dependency injection container. CQRS components (dispatcher,
        registry) should be used directly - not through FlextService.
        """

        config: p.Config
        context: p.Ctx
        container: p.DI

    # =========================================================================
    # CONTEXT MODELS - Direct access for common usage
    # =========================================================================

    ContextData: TypeAlias = FlextModelsContext.ContextData
    ContextDomainData: TypeAlias = FlextModelsContext.ContextDomainData
    ContextExport: TypeAlias = FlextModelsContext.ContextExport
    ContextScopeData: TypeAlias = FlextModelsContext.ContextScopeData
    ContextStatistics: TypeAlias = FlextModelsContext.ContextStatistics
    ContextMetadata: TypeAlias = FlextModelsContext.ContextMetadata

    # =========================================================================
    # COLLECTIONS MODELS - Direct access for common usage
    # =========================================================================

    Collections = FlextModelsCollections
    CollectionsCategories: TypeAlias = FlextModelsCollections.Categories
    CollectionsConfig: TypeAlias = FlextModelsCollections.Config
    CollectionsResults: TypeAlias = FlextModelsCollections.Results
    CollectionsOptions: TypeAlias = FlextModelsCollections.Options
    CollectionsStatistics: TypeAlias = FlextModelsCollections.Statistics
    Options: TypeAlias = FlextModelsCollections.Options
    CollectionsParseOptions: TypeAlias = FlextModelsCollections.ParseOptions
    Categories: TypeAlias = FlextModelsCollections.Categories

    # =========================================================================
    # CONTAINER MODELS - DI registry and service registration
    # =========================================================================

    class Container(FlextModelsContainer):
        """Container models namespace for DI and service registry.

        Re-exports FlextModelsContainer as a proper class for mypy compatibility.
        """

    # =========================================================================
    # CONFIG CLASSES - Direct access for common usage
    # =========================================================================

    RetryConfiguration: TypeAlias = FlextModelsConfig.RetryConfiguration
    DispatchConfig: TypeAlias = FlextModelsConfig.DispatchConfig

    class ExecuteDispatchAttemptOptions(
        FlextModelsConfig.ExecuteDispatchAttemptOptions,
    ):
        """Execute dispatch attempt options - direct class for mypy compatibility."""

    RuntimeScopeOptions: TypeAlias = FlextModelsConfig.RuntimeScopeOptions
    NestedExecutionOptions: TypeAlias = FlextModelsConfig.NestedExecutionOptions
    ExceptionConfig: TypeAlias = FlextModelsConfig.ExceptionConfig
    ValidationErrorConfig: TypeAlias = FlextModelsConfig.ValidationErrorConfig
    ConfigurationErrorConfig: TypeAlias = FlextModelsConfig.ConfigurationErrorConfig
    ConnectionErrorConfig: TypeAlias = FlextModelsConfig.ConnectionErrorConfig
    TimeoutErrorConfig: TypeAlias = FlextModelsConfig.TimeoutErrorConfig
    AuthenticationErrorConfig: TypeAlias = FlextModelsConfig.AuthenticationErrorConfig
    AuthorizationErrorConfig: TypeAlias = FlextModelsConfig.AuthorizationErrorConfig
    NotFoundErrorConfig: TypeAlias = FlextModelsConfig.NotFoundErrorConfig
    ConflictErrorConfig: TypeAlias = FlextModelsConfig.ConflictErrorConfig
    RateLimitErrorConfig: TypeAlias = FlextModelsConfig.RateLimitErrorConfig
    InternalErrorConfig: TypeAlias = FlextModelsConfig.InternalErrorConfig
    TypeErrorOptions: TypeAlias = FlextModelsConfig.TypeErrorOptions
    TypeErrorConfig: TypeAlias = FlextModelsConfig.TypeErrorConfig
    ValueErrorConfig: TypeAlias = FlextModelsConfig.ValueErrorConfig
    CircuitBreakerErrorConfig: TypeAlias = FlextModelsConfig.CircuitBreakerErrorConfig
    OperationErrorConfig: TypeAlias = FlextModelsConfig.OperationErrorConfig
    AttributeAccessErrorConfig: TypeAlias = FlextModelsConfig.AttributeAccessErrorConfig
    MiddlewareConfig: TypeAlias = FlextModelsConfig.MiddlewareConfig
    RateLimiterState: TypeAlias = FlextModelsConfig.RateLimiterState

    # =========================================================================
    # BASE CLASSES - Direct access for common usage
    # =========================================================================

    ArbitraryTypesModel: TypeAlias = FlextModelsBase.ArbitraryTypesModel
    FrozenStrictModel: TypeAlias = FlextModelsBase.FrozenStrictModel
    IdentifiableMixin: TypeAlias = FlextModelsBase.IdentifiableMixin
    TimestampableMixin: TypeAlias = FlextModelsBase.TimestampableMixin
    TimestampedModel: TypeAlias = FlextModelsBase.TimestampedModel
    VersionableMixin: TypeAlias = FlextModelsBase.VersionableMixin
    Metadata: TypeAlias = FlextModelsBase.Metadata

    # =========================================================================
    # HANDLER MODELS - Direct access for common usage
    # =========================================================================

    class Handler(FlextModelsCqrs.Handler):
        """Handler base class - real inheritance."""

        class RegistrationDetails(FlextModelsHandler.RegistrationDetails):
            """Handler registration details - real inheritance."""

        class ExecutionContext(FlextModelsHandler.ExecutionContext):
            """Handler execution context - real inheritance."""

        class DecoratorConfig(FlextModelsHandler.DecoratorConfig):
            """Handler decorator configuration - direct class for mypy compatibility."""

        class FactoryDecoratorConfig(FlextModelsContainer.FactoryDecoratorConfig):
            """Handler factory decorator configuration - direct class for mypy compatibility."""

    # Direct aliases for top-level access
    HandlerDecoratorConfig: TypeAlias = Handler.DecoratorConfig
    HandlerFactoryDecoratorConfig: TypeAlias = Handler.FactoryDecoratorConfig
    HandlerRegistrationDetails: TypeAlias = Handler.RegistrationDetails
    HandlerExecutionContext: TypeAlias = Handler.ExecutionContext
    CqrsHandler: TypeAlias = Handler

    # =========================================================================
    # UNIONS - Pydantic discriminated unions
    # =========================================================================

    MessageUnion = Annotated[
        FlextModelsCqrs.Command | FlextModelsCqrs.Query | FlextModelsEntity.DomainEvent,
        Discriminator("message_type"),
    ]


# =========================================================================
# MODULE ALIASES - Runtime access patterns
# =========================================================================

# Main alias for direct access
m = FlextModels
m_core = FlextModels

__all__ = ["FlextModels", "m", "m_core"]
