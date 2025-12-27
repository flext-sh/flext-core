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

from typing import Annotated

from pydantic import Discriminator

from flext_core._models.base import FlextModelsBase
from flext_core._models.collections import FlextModelsCollections
from flext_core._models.container import FlextModelsContainer
from flext_core._models.context import FlextModelsContext
from flext_core._models.cqrs import FlextModelsCqrs
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.generic import FlextGenericModels
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
    # CORE DOMAIN ENTITIES - Direct access for common usage
    # =========================================================================

    Entity = FlextModelsEntity.Entry
    ValueObject = FlextModelsEntity.Value
    AggregateRoot = FlextModelsEntity.AggregateRoot
    DomainEvent = FlextModelsEntity.DomainEvent

    # =========================================================================
    # GENERIC MODELS BY BUSINESS FUNCTION (from flext_core._models.generic)
    # =========================================================================

    Value = FlextGenericModels.Value
    """Value objects - immutable data compared by value."""

    Snapshot = FlextGenericModels.Snapshot
    """Snapshots - state captured at a specific moment."""

    Progress = FlextGenericModels.Progress
    """Progress trackers - mutable accumulators during operations."""

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

    Command = FlextModelsCqrs.Command
    Query = FlextModelsCqrs.Query
    Bus = FlextModelsCqrs.Bus
    Pagination = FlextModelsCqrs.Pagination

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
    ProcessingRequest = FlextModelsConfig.ProcessingRequest
    ProcessingConfig = ProcessingRequest
    BatchProcessingConfig = FlextModelsConfig.BatchProcessingConfig
    ValidationConfiguration = FlextModelsConfig.ValidationConfiguration
    HandlerRegistration = FlextModelsHandler.Registration
    HandlerExecutionConfig = FlextModelsConfig.HandlerExecutionConfig

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

    ContextData = FlextModelsContext.ContextData
    ContextDomainData = FlextModelsContext.ContextDomainData
    ContextExport = FlextModelsContext.ContextExport
    ContextScopeData = FlextModelsContext.ContextScopeData
    ContextStatistics = FlextModelsContext.ContextStatistics
    ContextMetadata = FlextModelsContext.ContextMetadata

    # =========================================================================
    # COLLECTIONS MODELS - Direct access for common usage
    # =========================================================================

    Collections = FlextModelsCollections
    CollectionsCategories = FlextModelsCollections.Categories
    CollectionsConfig = FlextModelsCollections.Config
    CollectionsResults = FlextModelsCollections.Results
    CollectionsOptions = FlextModelsCollections.Options
    CollectionsStatistics = FlextModelsCollections.Statistics
    Options = FlextModelsCollections.Options
    CollectionsParseOptions = FlextModelsCollections.ParseOptions
    Categories = CollectionsCategories

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

    RetryConfiguration = FlextModelsConfig.RetryConfiguration
    DispatchConfig = FlextModelsConfig.DispatchConfig

    class ExecuteDispatchAttemptOptions(
        FlextModelsConfig.ExecuteDispatchAttemptOptions,
    ):
        """Execute dispatch attempt options - direct class for mypy compatibility."""

    RuntimeScopeOptions = FlextModelsConfig.RuntimeScopeOptions
    NestedExecutionOptions = FlextModelsConfig.NestedExecutionOptions
    ExceptionConfig = FlextModelsConfig.ExceptionConfig
    ValidationErrorConfig = FlextModelsConfig.ValidationErrorConfig
    ConfigurationErrorConfig = FlextModelsConfig.ConfigurationErrorConfig
    ConnectionErrorConfig = FlextModelsConfig.ConnectionErrorConfig
    TimeoutErrorConfig = FlextModelsConfig.TimeoutErrorConfig
    AuthenticationErrorConfig = FlextModelsConfig.AuthenticationErrorConfig
    AuthorizationErrorConfig = FlextModelsConfig.AuthorizationErrorConfig
    NotFoundErrorConfig = FlextModelsConfig.NotFoundErrorConfig
    ConflictErrorConfig = FlextModelsConfig.ConflictErrorConfig
    RateLimitErrorConfig = FlextModelsConfig.RateLimitErrorConfig
    InternalErrorConfig = FlextModelsConfig.InternalErrorConfig
    TypeErrorOptions = FlextModelsConfig.TypeErrorOptions
    TypeErrorConfig = FlextModelsConfig.TypeErrorConfig
    ValueErrorConfig = FlextModelsConfig.ValueErrorConfig
    CircuitBreakerErrorConfig = FlextModelsConfig.CircuitBreakerErrorConfig
    OperationErrorConfig = FlextModelsConfig.OperationErrorConfig
    AttributeAccessErrorConfig = FlextModelsConfig.AttributeAccessErrorConfig
    MiddlewareConfig = FlextModelsConfig.MiddlewareConfig
    RateLimiterState = FlextModelsConfig.RateLimiterState

    # =========================================================================
    # BASE CLASSES - Direct access for common usage
    # =========================================================================

    ArbitraryTypesModel = FlextModelsBase.ArbitraryTypesModel
    FrozenStrictModel = FlextModelsBase.FrozenStrictModel
    IdentifiableMixin = FlextModelsBase.IdentifiableMixin
    TimestampableMixin = FlextModelsBase.TimestampableMixin
    TimestampedModel = FlextModelsBase.TimestampedModel
    VersionableMixin = FlextModelsBase.VersionableMixin
    Metadata = FlextModelsBase.Metadata

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
    HandlerDecoratorConfig = Handler.DecoratorConfig
    HandlerFactoryDecoratorConfig = Handler.FactoryDecoratorConfig
    HandlerRegistrationDetails = Handler.RegistrationDetails
    HandlerExecutionContext = Handler.ExecutionContext
    CqrsHandler = Handler

    # =========================================================================
    # UNIONS - Pydantic discriminated unions
    # =========================================================================

    MessageUnion = Annotated[
        Command | Query | DomainEvent,
        Discriminator("message_type"),
    ]


# =========================================================================
# MODULE ALIASES - Runtime access patterns
# =========================================================================

# Main alias for direct access
m = FlextModels
m_core = FlextModels

__all__ = ["FlextModels", "m", "m_core"]
