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
    # CORE DOMAIN ENTITIES - Direct access for common usage
    # =========================================================================

    Entity: TypeAlias = FlextModelsEntity.Entry
    Value: TypeAlias = FlextModelsEntity.Value
    AggregateRoot: TypeAlias = FlextModelsEntity.AggregateRoot
    DomainEvent: TypeAlias = FlextModelsEntity.DomainEvent

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

    Config: TypeAlias = FlextModelsConfig
    ProcessingRequest: TypeAlias = FlextModelsConfig.ProcessingRequest
    ProcessingConfig: TypeAlias = ProcessingRequest
    BatchProcessingConfig: TypeAlias = FlextModelsConfig.BatchProcessingConfig
    ValidationConfiguration: TypeAlias = FlextModelsConfig.ValidationConfiguration
    HandlerRegistration: TypeAlias = FlextModelsHandler.Registration
    HandlerExecutionConfig: TypeAlias = FlextModelsConfig.HandlerExecutionConfig

    # =========================================================================
    # SERVICE MODELS
    # =========================================================================

    class ServiceRuntime(FlextModelsBase.ArbitraryTypesModel):
        """Runtime quintuple (config, context, container, dispatcher, registry) for services.

        Represents the complete application runtime with configuration, context,
        dependency injection container, command bus dispatcher, and handler registry.
        Supports zero-config initialization via auto-discovery factories in
        FlextDispatcher.create() and FlextRegistry.create().
        """

        config: p.Config
        context: p.Ctx
        container: p.DI
        dispatcher: p.CommandBus
        registry: p.Registry

    # =========================================================================
    # CONTEXT MODELS - Direct access for common usage
    # =========================================================================

    ContextDomainData: TypeAlias = FlextModelsContext.ContextDomainData
    ContextExport: TypeAlias = FlextModelsContext.ContextExport
    ContextScopeData: TypeAlias = FlextModelsContext.ContextScopeData
    ContextStatistics: TypeAlias = FlextModelsContext.ContextStatistics
    ContextMetadata: TypeAlias = FlextModelsContext.ContextMetadata

    # =========================================================================
    # COLLECTIONS MODELS - Direct access for common usage
    # =========================================================================

    CollectionsCategories: TypeAlias = FlextModelsCollections.Categories
    CollectionsConfig: TypeAlias = FlextModelsCollections.Config
    CollectionsResults: TypeAlias = FlextModelsCollections.Results
    CollectionsOptions: TypeAlias = FlextModelsCollections.Options
    Options: TypeAlias = FlextModelsCollections.Options
    CollectionsParseOptions: TypeAlias = FlextModelsCollections.ParseOptions
    Categories: TypeAlias = CollectionsCategories

    # =========================================================================
    # VALIDATION FUNCTIONS - Direct access for common usage
    # =========================================================================

    validate_business_rules = FlextModelsValidation.validate_business_rules
    validate_cross_fields = FlextModelsValidation.validate_cross_fields
    validate_performance = FlextModelsValidation.validate_performance
    validate_batch = FlextModelsValidation.validate_batch
    validate_domain_invariants = FlextModelsValidation.validate_domain_invariants
    validate_aggregate_consistency_with_rules = (
        FlextModelsValidation.validate_aggregate_consistency_with_rules
    )
    validate_event_sourcing = FlextModelsValidation.validate_event_sourcing
    validate_cqrs_patterns = FlextModelsValidation.validate_cqrs_patterns
    validate_domain_event = FlextModelsValidation.validate_domain_event
    validate_aggregate_consistency = (
        FlextModelsValidation.validate_aggregate_consistency
    )
    validate_entity_relationships = FlextModelsValidation.validate_entity_relationships
    validate_uri = FlextModelsValidation.validate_uri
    validate_port_number = FlextModelsValidation.validate_port_number

    # =========================================================================
    # CONFIG CLASSES - Direct access for common usage
    # =========================================================================

    RetryConfiguration: TypeAlias = FlextModelsConfig.RetryConfiguration
    DispatchConfig: TypeAlias = FlextModelsConfig.DispatchConfig

    class ExecuteDispatchAttemptOptions(
        FlextModelsConfig.ExecuteDispatchAttemptOptions
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
    # TYPE UNIONS - Pydantic discriminated unions
    # =========================================================================

    type MessageUnion = Annotated[
        Command | Query | DomainEvent,
        Discriminator("message_type"),
    ]

    # =========================================================================
    # NAMESPACE ACCESS - Backward compatibility
    # =========================================================================

    # Context namespace - aggregates all context-related models
    class Context:
        """Context-related models aggregated for convenient access."""

        StructlogProxyContextVar: TypeAlias = (
            FlextModelsContext.StructlogProxyContextVar
        )
        StructlogProxyToken: TypeAlias = FlextModelsContext.StructlogProxyToken
        Token: TypeAlias = FlextModelsContext.Token
        ContextData: TypeAlias = FlextModelsContext.ContextData
        ContextDomainData: TypeAlias = FlextModelsContext.ContextDomainData
        ContextExport: TypeAlias = FlextModelsContext.ContextExport
        ContextScopeData: TypeAlias = FlextModelsContext.ContextScopeData
        ContextStatistics: TypeAlias = FlextModelsContext.ContextStatistics
        ContextMetadata: TypeAlias = FlextModelsContext.ContextMetadata

    # CQRS and Collections aliases
    Cqrs: TypeAlias = FlextModelsCqrs
    Collections: TypeAlias = FlextModelsCollections
    CollectionsStatistics: TypeAlias = FlextModelsCollections.Statistics

    # Container namespace - aggregates FlextModelsContainer
    class Container:
        """Container-related models aggregated for convenient access."""

        Container: TypeAlias = FlextModelsContainer
        ServiceRegistration: TypeAlias = FlextModelsContainer.ServiceRegistration
        FactoryRegistration: TypeAlias = FlextModelsContainer.FactoryRegistration
        ResourceRegistration: TypeAlias = FlextModelsContainer.ResourceRegistration
        ContainerConfig: TypeAlias = FlextModelsContainer.ContainerConfig

    # Validation namespace - aggregates FlextModelsValidation
    Validation: TypeAlias = FlextModelsValidation

    # Direct aliases for backward compatibility
    ContextData: TypeAlias = FlextModelsContext.ContextData


# =========================================================================
# MODULE ALIASES - Runtime access patterns
# =========================================================================

# Main alias for direct access
m = FlextModels
m_core = FlextModels

# Resolve forward references for Pydantic v2 compatibility
FlextModels.Identity.model_rebuild()
FlextModels.IdentityRequest.model_rebuild()

__all__ = ["FlextModels", "m", "m_core"]
