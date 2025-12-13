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

from collections.abc import Sequence
from typing import Annotated, TypeAlias

from pydantic import Discriminator, Field

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

    # CQRS Handler with subclasses (keep for compatibility)
    class Handler(FlextModelsCqrs.Handler):
        """Handler base class - real inheritance."""

        class RegistrationDetails(FlextModelsHandler.RegistrationDetails):
            """Handler registration details - real inheritance."""

        class ExecutionContext(FlextModelsHandler.ExecutionContext):
            """Handler execution context - real inheritance."""

        class DecoratorConfig(FlextModelsBase.ArbitraryTypesModel):
            """Handler decorator configuration."""

            command: type | None = None
            priority: int = 100
            timeout: float | None = None
            middleware: Sequence[type[object]] = Field(default_factory=list)

        class FactoryDecoratorConfig(FlextModelsContainer.FactoryDecoratorConfig):
            """Factory decorator config - real inheritance."""

    # Aliases for direct access
    HandlerRegistrationDetails: TypeAlias = Handler.RegistrationDetails
    HandlerExecutionContext = Handler.ExecutionContext
    Config = FlextModelsConfig
    ProcessingRequest = FlextModelsConfig.ProcessingRequest
    ProcessingConfig = ProcessingRequest  # Simple access alias
    BatchProcessingConfig = FlextModelsConfig.BatchProcessingConfig
    ValidationConfiguration = FlextModelsConfig.ValidationConfiguration
    HandlerRegistration = FlextModelsHandler.Registration
    HandlerExecutionConfig = FlextModelsConfig.HandlerExecutionConfig

    # Direct class definitions for type safety
    class HandlerDecoratorConfig(FlextModelsHandler.DecoratorConfig):
        """Handler decorator configuration - direct class for mypy compatibility."""

    class HandlerFactoryDecoratorConfig(FlextModelsContainer.FactoryDecoratorConfig):
        """Handler factory decorator configuration - direct class for mypy compatibility."""

    # Direct alias for CQRS handler
    CqrsHandler = Handler

    # Type aliases for mypy compatibility
    Entity: TypeAlias = FlextModelsEntity.Entry
    Value: TypeAlias = FlextModelsEntity.Value
    AggregateRoot: TypeAlias = FlextModelsEntity.AggregateRoot
    DomainEvent: TypeAlias = FlextModelsEntity.DomainEvent
    ArbitraryTypesModel: TypeAlias = FlextModelsBase.ArbitraryTypesModel
    FrozenStrictModel: TypeAlias = FlextModelsBase.FrozenStrictModel
    IdentifiableMixin: TypeAlias = FlextModelsBase.IdentifiableMixin
    TimestampableMixin: TypeAlias = FlextModelsBase.TimestampableMixin
    TimestampedModel: TypeAlias = FlextModelsBase.TimestampedModel
    VersionableMixin: TypeAlias = FlextModelsBase.VersionableMixin
    CollectionsCategories: TypeAlias = FlextModelsCollections.Categories

    # Direct aliases for simple access (remove subnamespaces)
    Command = FlextModelsCqrs.Command
    Pagination = FlextModelsCqrs.Pagination
    Query = FlextModelsCqrs.Query
    Bus = FlextModelsCqrs.Bus

    # NOTE: Use FlextSettings.get_global_instance() directly in model defaults
    # No wrapper methods needed - access config directly per FLEXT standards

    # Type aliases for mypy compatibility
    Metadata: TypeAlias = FlextModelsBase.Metadata
    StructlogProxyToken: TypeAlias = FlextModelsContext.StructlogProxyToken
    StructlogProxyContextVar: TypeAlias = FlextModelsContext.StructlogProxyContextVar
    Token: TypeAlias = FlextModelsContext.Token
    ContextData: TypeAlias = FlextModelsContext.ContextData

    # Configuration Models - All classes use real inheritance (no type aliases)
    # Access directly: FlextModels.ProcessingConfig, etc.

    # Service Models - Real inheritance classes
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

    # Container Models - Real inheritance classes (flattened from Container namespace)

    # Type aliases for mypy compatibility
    ServiceRegistration: TypeAlias = FlextModelsContainer.ServiceRegistration
    FactoryRegistration: TypeAlias = FlextModelsContainer.FactoryRegistration
    ResourceRegistration: TypeAlias = FlextModelsContainer.ResourceRegistration
    ContainerConfig: TypeAlias = FlextModelsContainer.ContainerConfig

    # Pydantic v2 discriminated union using modern typing (PEP 695)
    # Use direct references to base classes, not aliases
    type MessageUnion = Annotated[
        FlextModelsCqrs.Command | FlextModelsCqrs.Query | FlextModelsEntity.DomainEvent,
        Discriminator("message_type"),
    ]

    # =========================================================================
    # FLAT NAMESPACE ACCESS - NEW STANDARD (December 2025)
    # =========================================================================
    # Classes are now accessible via flat namespace for common usage patterns.
    # Sub-namespaces (Config, Cqrs, Context, etc.) are DEPRECATED but maintained
    # for backward compatibility. New code should use flat access.
    #
    # CORRECT (new): m.DispatchConfig, m.Command, m.ContextData
    # DEPRECATED:    m.DispatchConfig, m.Command, m.ContextData
    #
    # Project-specific namespaces (m.Cli.*, m.Ldif.*) remain for extensions.
    # =========================================================================

    # Direct class definitions for type safety
    class ContextDomainData(FlextModelsContext.ContextDomainData):
        """Context domain data - direct class for mypy compatibility."""

    class ContextExport(FlextModelsContext.ContextExport):
        """Context export data - direct class for mypy compatibility."""

    class ContextScopeData(FlextModelsContext.ContextScopeData):
        """Context scope data - direct class for mypy compatibility."""

    class ContextStatistics(FlextModelsContext.ContextStatistics):
        """Context statistics - direct class for mypy compatibility."""

    class ContextMetadata(FlextModelsContext.ContextMetadata):
        """Context metadata - direct class for mypy compatibility."""

    # CQRS and Collections aliases
    Cqrs = FlextModelsCqrs
    Collections = FlextModelsCollections
    CollectionsStatistics = FlextModelsCollections.Statistics
    # Type aliases for mypy compatibility
    CollectionsConfig: TypeAlias = FlextModelsBase.ArbitraryTypesModel
    CollectionsResults: TypeAlias = FlextModelsCollections.Results
    CollectionsOptions: TypeAlias = FlextModelsCollections.Options
    Options: TypeAlias = FlextModelsCollections.Options  # Simple access
    CollectionsParseOptions: TypeAlias = FlextModelsCollections.ParseOptions
    Categories: TypeAlias = CollectionsCategories

    # NOTE: m.Metadata is already available via class Metadata(FlextModelsBase.Metadata)
    # defined in the Entity namespace section above (line 156)

    # Validation namespace - aggregates FlextModelsValidation
    Validation = FlextModelsValidation

    # Direct aliases for validation functions (remove subnamespace)
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
    # ESSENTIAL CONFIG CLASSES - Direct access for common usage
    # =========================================================================
    # Type aliases for mypy compatibility
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


m = FlextModels
m_core = FlextModels

__all__ = ["FlextModels", "m", "m_core"]
