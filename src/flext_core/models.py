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
from flext_core._models.config import FlextModelsConfig
from flext_core._models.container import FlextModelsContainer
from flext_core._models.context import FlextModelsContext
from flext_core._models.cqrs import FlextModelsCqrs
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.handler import FlextModelsHandler
from flext_core._models.service import FlextModelsService
from flext_core._models.validation import FlextModelsValidation
from flext_core.protocols import p
from flext_core.typings import t


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

    # Entity & DDD Patterns - Real inheritance classes
    class Entity(FlextModelsEntity.Entry):
        """Entity base class with real inheritance."""

    class Value(FlextModelsEntity.Value):
        """Value object base class with real inheritance."""

    class AggregateRoot(Entity, FlextModelsEntity.AggregateRoot):
        """Aggregate root base class with real inheritance.

        Inherits from both Entity (for hierarchy) and FlextModelsEntity.AggregateRoot
        (for functionality) to maintain correct inheritance chain.

        Note: type: ignore[misc] is required because mypy cannot determine type of
        "model_config" in base classes with multiple inheritance (Pydantic v2 limitation).
        """

    class DomainEvent(FlextModelsEntity.DomainEvent):
        """Domain event base class with real inheritance."""

    # Direct base class - real inheritance
    class ArbitraryTypesModel(FlextModelsBase.ArbitraryTypesModel):
        """Base model with arbitrary types support - real inheritance."""

    # Base namespace - Real inheritance classes
    class Base:
        """Base namespace with real inheritance classes."""

        class Metadata(FlextModelsBase.Metadata):
            """Standard metadata model - real inheritance."""

    # Base Models - Real inheritance classes
    class FrozenStrictModel(FlextModelsBase.FrozenStrictModel):
        """Immutable base model with strict validation - real inheritance."""

    class IdentifiableMixin(FlextModelsBase.IdentifiableMixin):
        """Mixin for unique identifiers - real inheritance."""

    class TimestampableMixin(FlextModelsBase.TimestampableMixin):
        """Mixin for timestamps - real inheritance."""

    class TimestampedModel(FlextModelsBase.TimestampedModel):
        """Timestamped model - real inheritance."""

    class VersionableMixin(FlextModelsBase.VersionableMixin):
        """Mixin for versioning - real inheritance."""

    # Collections - Real inheritance classes
    class Collections:
        """Collections namespace with real inheritance classes."""

        class Config(FlextModelsCollections.Config):
            """Collections config - real inheritance."""

        class Rules(FlextModelsCollections.Rules):
            """Collections rules - real inheritance."""

        class Statistics(FlextModelsCollections.Statistics):
            """Collections statistics - real inheritance."""

        class Results(FlextModelsCollections.Results):
            """Collections results - real inheritance."""

        class Options(FlextModelsCollections.Options):
            """Collections options - real inheritance."""

        class ParseOptions(FlextModelsCollections.ParseOptions):
            """Collections parse options - real inheritance."""

        # Categories is generic - expose as class with real inheritance
        class Categories[T](FlextModelsCollections.Categories[T]):
            """Categories collection with real inheritance - generic class."""

    # Categories with default type - expose at class level with real inheritance
    class Categories(FlextModelsCollections.Categories[t.GeneralValueType]):
        """Categories collection with default GeneralValueType - real inheritance."""

    # Collections - Real inheritance classes (no type aliases)
    # CollectionsConfig removed - use Collections.Config directly

    # CQRS Patterns - Real inheritance classes
    class Cqrs:
        """CQRS namespace with real inheritance classes."""

        class Command(FlextModelsCqrs.Command):
            """Command base class - real inheritance."""

        class Pagination(FlextModelsCqrs.Pagination):
            """Pagination base class - real inheritance."""

        class Query(FlextModelsCqrs.Query):
            """Query base class - real inheritance."""

        class Bus(FlextModelsCqrs.Bus):
            """Bus base class - real inheritance."""

        class Handler(FlextModelsCqrs.Handler):
            """Handler base class - real inheritance."""

        # NOTE: Use FlextConfig.get_global_instance() directly in model defaults
        # No wrapper methods needed - access config directly per FLEXT standards

    # Base Utility Models - Real inheritance classes
    class Metadata(FlextModelsBase.Metadata):
        """Metadata model with real inheritance."""

    # Configuration Models - Real inheritance classes
    class Config:
        """Configuration namespace with real inheritance classes."""

        # Domain model configuration - moved from constants.py
        # constants.py cannot import ConfigDict, so this belongs here
        DOMAIN_MODEL_CONFIG = FlextModelsConfig.DOMAIN_MODEL_CONFIG
        """Domain model configuration defaults.
        
        Moved from FlextConstants.Domain.DOMAIN_MODEL_CONFIG because
        constants.py cannot import ConfigDict from pydantic.
        
        Use m.Config.DOMAIN_MODEL_CONFIG instead of c.Domain.DOMAIN_MODEL_CONFIG.
        """

        class ProcessingRequest(FlextModelsConfig.ProcessingRequest):
            """Processing request config - real inheritance."""

        class RetryConfiguration(FlextModelsConfig.RetryConfiguration):
            """Retry configuration - real inheritance."""

        class ValidationConfiguration(FlextModelsConfig.ValidationConfiguration):
            """Validation configuration - real inheritance."""

        class BatchProcessingConfig(FlextModelsConfig.BatchProcessingConfig):
            """Batch processing config - real inheritance."""

        class HandlerExecutionConfig(FlextModelsConfig.HandlerExecutionConfig):
            """Handler execution config - real inheritance."""

        class OperationExtraConfig(FlextModelsConfig.OperationExtraConfig):
            """Operation extra config - real inheritance."""

        class LogOperationFailureConfig(FlextModelsConfig.LogOperationFailureConfig):
            """Log operation failure config - real inheritance."""

        class RetryLoopConfig(FlextModelsConfig.RetryLoopConfig):
            """Retry loop config - real inheritance."""

        class DispatchConfig(FlextModelsConfig.DispatchConfig):
            """Dispatch config - real inheritance."""

        class ExecuteDispatchAttemptOptions(
            FlextModelsConfig.ExecuteDispatchAttemptOptions,
        ):
            """Execute dispatch attempt options - real inheritance."""

        class RuntimeScopeOptions(FlextModelsConfig.RuntimeScopeOptions):
            """Runtime scope options - real inheritance."""

        class NestedExecutionOptions(FlextModelsConfig.NestedExecutionOptions):
            """Nested execution options - real inheritance."""

        class ExceptionConfig(FlextModelsConfig.ExceptionConfig):
            """Exception config - real inheritance."""

        class ValidationErrorConfig(FlextModelsConfig.ValidationErrorConfig):
            """Validation error config - real inheritance."""

        class ConfigurationErrorConfig(FlextModelsConfig.ConfigurationErrorConfig):
            """Configuration error config - real inheritance."""

        class ConnectionErrorConfig(FlextModelsConfig.ConnectionErrorConfig):
            """Connection error config - real inheritance."""

        class TimeoutErrorConfig(FlextModelsConfig.TimeoutErrorConfig):
            """Timeout error config - real inheritance."""

        class AuthenticationErrorConfig(FlextModelsConfig.AuthenticationErrorConfig):
            """Authentication error config - real inheritance."""

        class AuthorizationErrorConfig(FlextModelsConfig.AuthorizationErrorConfig):
            """Authorization error config - real inheritance."""

        class NotFoundErrorConfig(FlextModelsConfig.NotFoundErrorConfig):
            """NotFound error config - real inheritance."""

        class ConflictErrorConfig(FlextModelsConfig.ConflictErrorConfig):
            """Conflict error config - real inheritance."""

        class RateLimitErrorConfig(FlextModelsConfig.RateLimitErrorConfig):
            """Rate limit error config - real inheritance."""

        class InternalErrorConfig(FlextModelsConfig.InternalErrorConfig):
            """Internal error config - real inheritance."""

        class TypeErrorOptions(FlextModelsConfig.TypeErrorOptions):
            """Type error options - real inheritance."""

        class TypeErrorConfig(FlextModelsConfig.TypeErrorConfig):
            """Type error config - real inheritance."""

        class ValueErrorConfig(FlextModelsConfig.ValueErrorConfig):
            """Value error config - real inheritance."""

        class CircuitBreakerErrorConfig(FlextModelsConfig.CircuitBreakerErrorConfig):
            """Circuit breaker error config - real inheritance."""

        class OperationErrorConfig(FlextModelsConfig.OperationErrorConfig):
            """Operation error config - real inheritance."""

        class AttributeAccessErrorConfig(FlextModelsConfig.AttributeAccessErrorConfig):
            """Attribute access error config - real inheritance."""

        class MiddlewareConfig(FlextModelsConfig.MiddlewareConfig):
            """Middleware config - real inheritance."""

        class RateLimiterState(FlextModelsConfig.RateLimiterState):
            """Rate limiter state - real inheritance."""

    # Configuration Models - All classes use real inheritance (no type aliases)
    # Access via Config namespace: FlextModels.Config.ProcessingRequest, etc.

    class Context:
        """Context namespace with real inheritance classes."""

        class StructlogProxyToken(FlextModelsContext.StructlogProxyToken):
            """Structlog proxy token - real inheritance."""

        class StructlogProxyContextVar[T](
            FlextModelsContext.StructlogProxyContextVar[T],
        ):
            """Structlog proxy context var - real inheritance."""

        class Token(FlextModelsContext.Token):
            """Token - real inheritance."""

        class ContextData(FlextModelsContext.ContextData):
            """Context data - real inheritance."""

        class ContextExport(FlextModelsContext.ContextExport):
            """Context export - real inheritance."""

        class ContextScopeData(FlextModelsContext.ContextScopeData):
            """Context scope data - real inheritance."""

        class ContextStatistics(FlextModelsContext.ContextStatistics):
            """Context statistics - real inheritance."""

        class ContextMetadata(FlextModelsContext.ContextMetadata):
            """Context metadata - real inheritance."""

        class ContextDomainData(FlextModelsContext.ContextDomainData):
            """Context domain data - real inheritance."""

    # Handler Management Models - Real inheritance classes
    class Handler:
        """Handler namespace with real inheritance classes."""

        class Registration(FlextModelsHandler.Registration):
            """Handler registration - real inheritance."""

        class RegistrationDetails(FlextModelsHandler.RegistrationDetails):
            """Registration details - real inheritance."""

        class ExecutionContext(FlextModelsHandler.ExecutionContext):
            """Handler execution context - real inheritance."""

        class DecoratorConfig(FlextModelsHandler.DecoratorConfig):
            """Decorator configuration - real inheritance."""

    # Service Models - Real inheritance classes
    # ServiceRuntime needs to stay as class due to protocol fields
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

    class Service:
        """Service namespace with real inheritance classes."""

        class DomainServiceExecutionRequest(
            FlextModelsService.DomainServiceExecutionRequest,
        ):
            """Domain service execution request - real inheritance."""

        class DomainServiceBatchRequest(FlextModelsService.DomainServiceBatchRequest):
            """Domain service batch request - real inheritance."""

        class DomainServiceMetricsRequest(
            FlextModelsService.DomainServiceMetricsRequest,
        ):
            """Domain service metrics request - real inheritance."""

        class DomainServiceResourceRequest(
            FlextModelsService.DomainServiceResourceRequest,
        ):
            """Domain service resource request - real inheritance."""

        class AclResponse(FlextModelsService.AclResponse):
            """ACL response - real inheritance."""

        class OperationExecutionRequest(FlextModelsService.OperationExecutionRequest):
            """Operation execution request - real inheritance."""

    # Container Models - Real inheritance classes
    class Container:
        """Container namespace with real inheritance classes."""

        class ServiceRegistration(FlextModelsContainer.ServiceRegistration):
            """Service registration - real inheritance."""

        class FactoryRegistration(FlextModelsContainer.FactoryRegistration):
            """Factory registration - real inheritance."""

        class FactoryDecoratorConfig(FlextModelsContainer.FactoryDecoratorConfig):
            """Factory decorator config - real inheritance."""

        class ResourceRegistration(FlextModelsContainer.ResourceRegistration):
            """Resource registration - real inheritance."""

        class ContainerConfig(FlextModelsContainer.ContainerConfig):
            """Container config - real inheritance."""

    # Pydantic v2 discriminated union using modern typing (PEP 695)
    # Use direct references to base classes, not aliases
    type MessageUnion = Annotated[
        FlextModelsCqrs.Command | FlextModelsCqrs.Query | FlextModelsEntity.DomainEvent,
        Discriminator("message_type"),
    ]

    # Validation Patterns - Real inheritance class
    class Validation(FlextModelsValidation):
        """Validation namespace with real inheritance.

        All validation methods are available through inheritance from
        FlextModelsValidation. Methods return Result[T] for various T.
        """

    # =========================================================================
    # NAMESPACE HIERARCHY - PADRAO CORRETO
    # =========================================================================
    # Todos os projetos devem usar namespace hierárquico completo
    # SEM duplicação de declarações ou aliases de raiz
    #
    # CORRETO: m.Cqrs.Command, m.Config.ProcessingRequest, m.Context.ContextData
    # ERRADO:  m.Command, m.ProcessingRequest, m.ContextData (PROIBIDO)
    #
    # Herança real, não aliases - todas as classes herdam diretamente
    # Sem quebra de código - mantém compatibilidade backward
    # =========================================================================


m = FlextModels
m_core = FlextModels

__all__ = ["FlextModels", "m", "m_core"]
