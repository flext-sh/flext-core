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

from typing import TYPE_CHECKING, Annotated

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

if TYPE_CHECKING:
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

    type Entity = FlextModelsEntity.Entry
    type ValueObject = FlextModelsEntity.Value
    type AggregateRoot = FlextModelsEntity.AggregateRoot
    type DomainEvent = FlextModelsEntity.DomainEvent

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

    type Command = FlextModelsCqrs.Command
    type Query = FlextModelsCqrs.Query
    type Bus = FlextModelsCqrs.Bus
    type Pagination = FlextModelsCqrs.Pagination

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

    type Config = FlextModelsConfig
    type ProcessingRequest = FlextModelsConfig.ProcessingRequest
    type ProcessingConfig = ProcessingRequest
    type BatchProcessingConfig = FlextModelsConfig.BatchProcessingConfig
    type ValidationConfiguration = FlextModelsConfig.ValidationConfiguration
    type HandlerRegistration = FlextModelsHandler.Registration
    type HandlerExecutionConfig = FlextModelsConfig.HandlerExecutionConfig

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

    type ContextData = FlextModelsContext.ContextData
    type ContextDomainData = FlextModelsContext.ContextDomainData
    type ContextExport = FlextModelsContext.ContextExport
    type ContextScopeData = FlextModelsContext.ContextScopeData
    type ContextStatistics = FlextModelsContext.ContextStatistics
    type ContextMetadata = FlextModelsContext.ContextMetadata

    # =========================================================================
    # COLLECTIONS MODELS - Direct access for common usage
    # =========================================================================

    Collections = FlextModelsCollections
    type CollectionsCategories = FlextModelsCollections.Categories
    type CollectionsConfig = FlextModelsCollections.Config
    type CollectionsResults = FlextModelsCollections.Results
    type CollectionsOptions = FlextModelsCollections.Options
    type CollectionsStatistics = FlextModelsCollections.Statistics
    type Options = FlextModelsCollections.Options
    type CollectionsParseOptions = FlextModelsCollections.ParseOptions
    type Categories = CollectionsCategories

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

    type RetryConfiguration = FlextModelsConfig.RetryConfiguration
    type DispatchConfig = FlextModelsConfig.DispatchConfig

    class ExecuteDispatchAttemptOptions(
        FlextModelsConfig.ExecuteDispatchAttemptOptions,
    ):
        """Execute dispatch attempt options - direct class for mypy compatibility."""

    type RuntimeScopeOptions = FlextModelsConfig.RuntimeScopeOptions
    type NestedExecutionOptions = FlextModelsConfig.NestedExecutionOptions
    type ExceptionConfig = FlextModelsConfig.ExceptionConfig
    type ValidationErrorConfig = FlextModelsConfig.ValidationErrorConfig
    type ConfigurationErrorConfig = FlextModelsConfig.ConfigurationErrorConfig
    type ConnectionErrorConfig = FlextModelsConfig.ConnectionErrorConfig
    type TimeoutErrorConfig = FlextModelsConfig.TimeoutErrorConfig
    type AuthenticationErrorConfig = FlextModelsConfig.AuthenticationErrorConfig
    type AuthorizationErrorConfig = FlextModelsConfig.AuthorizationErrorConfig
    type NotFoundErrorConfig = FlextModelsConfig.NotFoundErrorConfig
    type ConflictErrorConfig = FlextModelsConfig.ConflictErrorConfig
    type RateLimitErrorConfig = FlextModelsConfig.RateLimitErrorConfig
    type InternalErrorConfig = FlextModelsConfig.InternalErrorConfig
    type TypeErrorOptions = FlextModelsConfig.TypeErrorOptions
    type TypeErrorConfig = FlextModelsConfig.TypeErrorConfig
    type ValueErrorConfig = FlextModelsConfig.ValueErrorConfig
    type CircuitBreakerErrorConfig = FlextModelsConfig.CircuitBreakerErrorConfig
    type OperationErrorConfig = FlextModelsConfig.OperationErrorConfig
    type AttributeAccessErrorConfig = FlextModelsConfig.AttributeAccessErrorConfig
    type MiddlewareConfig = FlextModelsConfig.MiddlewareConfig
    type RateLimiterState = FlextModelsConfig.RateLimiterState

    # =========================================================================
    # BASE CLASSES - Direct access for common usage
    # =========================================================================

    type ArbitraryTypesModel = FlextModelsBase.ArbitraryTypesModel
    type FrozenStrictModel = FlextModelsBase.FrozenStrictModel
    type IdentifiableMixin = FlextModelsBase.IdentifiableMixin
    type TimestampableMixin = FlextModelsBase.TimestampableMixin
    type TimestampedModel = FlextModelsBase.TimestampedModel
    type VersionableMixin = FlextModelsBase.VersionableMixin
    type Metadata = FlextModelsBase.Metadata

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
    type HandlerDecoratorConfig = Handler.DecoratorConfig
    type HandlerFactoryDecoratorConfig = Handler.FactoryDecoratorConfig
    type HandlerRegistrationDetails = Handler.RegistrationDetails
    type HandlerExecutionContext = Handler.ExecutionContext
    type CqrsHandler = Handler

    # =========================================================================
    # TYPE UNIONS - Pydantic discriminated unions
    # =========================================================================

    type MessageUnion = Annotated[
        FlextModels.Command | FlextModels.Query | FlextModels.DomainEvent,
        Discriminator("message_type"),
    ]


# =========================================================================
# MODULE ALIASES - Runtime access patterns
# =========================================================================

# Main alias for direct access
m = FlextModels
m_core = FlextModels

__all__ = ["FlextModels", "m", "m_core"]
