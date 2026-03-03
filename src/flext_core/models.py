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

from collections.abc import Mapping
from typing import ClassVar, TypeAlias

from flext_core import p, t
from flext_core._models.base import FlextModelFoundation
from flext_core._models.collections import FlextModelsCollections
from flext_core._models.container import FlextModelsContainer
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.context import FlextModelsContext
from flext_core._models.cqrs import FlextModelsCqrs
from flext_core._models.decorators import FlextModelsDecorators
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.generic import FlextGenericModels
from flext_core._models.handler import FlextModelsHandler
from flext_core._models.mixin import FlextModelsMixin
from flext_core._models.settings import FlextModelsConfig


class FlextModels:
    """Facade that groups DDD building blocks for CQRS-ready domains.

    Architecture: Domain layer helper
    Provides strongly typed base classes for entities, aggregates, commands,
    queries, and domain events so dispatcher handlers can enforce invariants,
    collect domain events, and validate inputs through Pydantic v2.

    Core concepts
    - Entity: Domain instance with identity and lifecycle controls.
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

    class Value(FlextModelsEntity.Value):
        """Value object base class - immutable, compared by value."""

    class AggregateRoot(FlextModelsEntity.AggregateRoot):
        """Aggregate root base class - consistency boundary."""

    class DomainEvent(FlextModelsEntity.DomainEvent):
        """Domain event — real re-export for pydantic-mypy compatibility."""

    # =========================================================================
    # GENERIC MODELS BY BUSINESS FUNCTION - FLAT namespace (no intermediate levels)
    # =========================================================================
    # CRITICAL: Flat namespace — models on FlextModels directly (m.Configuration, m.Service, etc.).
    # Subprojects: inherit facade, add nested classes, then class-level aliases at root so usage
    # is m.Foo (e.g. ExecuteResult = TargetOracle.ExecuteResult). MRO protocol only; no subdivision.

    # VALUE OBJECTS - Immutable data compared by value
    class OperationContext(FlextGenericModels.Value.OperationContext):
        """Immutable context of an operation (from Value namespace)."""

    # SNAPSHOTS - State captured at a specific moment
    class Service(FlextGenericModels.Snapshot.Service):
        """Snapshot of service state (from Snapshot namespace)."""

    class Configuration(FlextGenericModels.Snapshot.Configuration):
        """Snapshot of configuration at a moment (from Snapshot namespace)."""

    class Health(FlextGenericModels.Snapshot.Health):
        """Result of health check (from Snapshot namespace)."""

    # PROGRESS TRACKERS - Mutable accumulators during operations
    class Operation(FlextGenericModels.Progress.Operation):
        """Progress of ongoing operation (from Progress namespace)."""

    class Conversion(FlextGenericModels.Progress.Conversion):
        """Progress of conversion with errors/warnings (from Progress namespace)."""

    # GENERIC CONTAINERS - Replace dict aliases
    ConfigMap: TypeAlias = FlextModelsContainers.ConfigMap
    """Configuration map container (replaces ConfigurationDict)."""

    ServiceMap: TypeAlias = FlextModelsContainers.ServiceMap
    """Service registry map container."""

    ErrorMap: TypeAlias = FlextModelsContainers.ErrorMap
    """Error type mapping container."""

    Dict: TypeAlias = FlextModelsContainers.Dict
    """Generic dictionary container."""

    ObjectList: TypeAlias = FlextModelsContainers.ObjectList
    """Sequence of container values for batch operations."""

    FactoryMap: TypeAlias = FlextModelsContainers.FactoryMap
    """Map of factory registration callables."""

    ResourceMap: TypeAlias = FlextModelsContainers.ResourceMap
    """Map of resource callables."""

    ValidatorCallable: TypeAlias = FlextModelsContainers.ValidatorCallable
    """Callable validator container."""

    FieldValidatorMap: TypeAlias = FlextModelsContainers.FieldValidatorMap
    """Map of field validators."""

    ConsistencyRuleMap: TypeAlias = FlextModelsContainers.ConsistencyRuleMap
    """Map of consistency rules."""

    EventValidatorMap: TypeAlias = FlextModelsContainers.EventValidatorMap
    """Map of event validators."""

    BatchResultDict: TypeAlias = FlextModelsContainers.BatchResultDict
    """Result payload model for batch operation outputs."""
    # =========================================================================
    # NAMESPACE CLASSES - Direct access for internal model classes
    # =========================================================================

    Base = FlextModelFoundation
    Validators = FlextModelFoundation.Validators
    Cqrs = FlextModelsCqrs
    EntityModels = FlextModelsEntity
    Entity_ns = FlextModelsEntity
    ContextModels = FlextModelsContext
    Context = FlextModelsContext
    HandlerModels = FlextModelsHandler
    Handler_ns = FlextModelsHandler
    ValidationModels = FlextModelFoundation.Validators
    Validation = FlextModelFoundation.Validators

    # =========================================================================
    # CQRS MESSAGING - Direct access for common usage
    # =========================================================================

    class Command(FlextModelsCqrs.Command):
        """CQRS Command base."""

    class Query(FlextModelsCqrs.Query):
        """CQRS Query base."""

    class Event(FlextModelsCqrs.Event):
        """CQRS Event base."""

    class Bus(FlextModelsCqrs.Bus):
        """CQRS Bus base."""

    class Pagination(FlextModelsCqrs.Pagination):
        """Pagination model base."""

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
        metadata: Mapping[str, str] | None = None

    class IdentityRequest(FlextModelsCqrs.Command):
        """Command for identity operations in auth domain."""

        name: str
        contact: str
        credential: str
        roles: list[str] | None = None
        metadata: Mapping[str, str] | None = None

    # =========================================================================
    # CONFIGURATION MODELS - Direct access for common usage
    # =========================================================================

    Config: TypeAlias = FlextModelsConfig

    class ProcessingRequest(FlextModelsConfig.ProcessingRequest):
        """Processing request — real re-export for pydantic-mypy compatibility."""

    class ProcessingConfig(FlextModelsConfig.ProcessingRequest):
        """Processing config alias — real re-export for pydantic-mypy compatibility."""

    class BatchProcessingConfig(FlextModelsConfig.BatchProcessingConfig):
        """Batch processing config — real re-export for pydantic-mypy compatibility."""

    class ValidationConfiguration(FlextModelsConfig.ValidationConfiguration):
        """Validation configuration — real re-export for pydantic-mypy compatibility."""

    class HandlerRegistration(FlextModelsHandler.Registration):
        """Handler registration — real re-export for pydantic-mypy compatibility."""

    class HandlerExecutionConfig(FlextModelsConfig.HandlerExecutionConfig):
        """Handler execution config — real re-export for pydantic-mypy compatibility."""

    # =========================================================================
    # SERVICE MODELS
    # =========================================================================

    class ServiceRuntime(FlextModelFoundation.ArbitraryTypesModel):
        """Runtime triple (config, context, container) for services.

        Represents the core service runtime with configuration, context,
        and dependency injection container. CQRS components (dispatcher,
        registry) should be used directly - not through FlextService.
        """

        config: p.Config
        context: p.Context
        container: p.DI

    # =========================================================================
    # CONTEXT MODELS - Direct access for common usage
    # =========================================================================

    class ContextData(FlextModelsContext.ContextData):
        """Context data — real re-export for pydantic-mypy compatibility."""

    class ContextDomainData(FlextModelsContext.ContextDomainData):
        """Context domain data — real re-export for pydantic-mypy compatibility."""

    class ContextExport(FlextModelsContext.ContextExport):
        """Context export — real re-export for pydantic-mypy compatibility."""

    class ContextScopeData(FlextModelsContext.ContextScopeData):
        """Context scope data — real re-export for pydantic-mypy compatibility."""

    class ContextStatistics(FlextModelsContext.ContextStatistics):
        """Context statistics — real re-export for pydantic-mypy compatibility."""

    class ContextMetadata(FlextModelsContext.ContextMetadata):
        """Context metadata — real re-export for pydantic-mypy compatibility."""

    # =========================================================================
    # COLLECTIONS MODELS - Direct access for common usage
    # =========================================================================

    Collections = FlextModelsCollections

    class CollectionsCategories(FlextModelsCollections.Categories):
        """Collections categories — real re-export for pydantic-mypy compatibility."""

    class CollectionsConfig(FlextModelsCollections.Config):
        """Collections config — real re-export for pydantic-mypy compatibility."""

    class CollectionsResults(FlextModelsCollections.Results):
        """Collections results — real re-export for pydantic-mypy compatibility."""

    class CollectionsOptions(FlextModelsCollections.Options):
        """Collections options — real re-export for pydantic-mypy compatibility."""

    class CollectionsStatistics(FlextModelsCollections.Statistics):
        """Collections statistics — real re-export for pydantic-mypy compatibility."""

    class Options(FlextModelsCollections.Options):
        """Options — real re-export for pydantic-mypy compatibility."""

    class CollectionsParseOptions(FlextModelsCollections.ParseOptions):
        """Collections parse options — real re-export for pydantic-mypy compatibility."""

    class Categories(FlextModelsCollections.Categories):
        """Categories — real re-export for pydantic-mypy compatibility."""

    class Rules(FlextModelsCollections.Rules):
        """Rules — real re-export for pydantic-mypy compatibility."""

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

    class RetryConfiguration(FlextModelsConfig.RetryConfiguration):
        """Retry configuration — real re-export for pydantic-mypy compatibility."""

    class DispatchConfig(FlextModelsConfig.DispatchConfig):
        """Dispatch config — real re-export for pydantic-mypy compatibility."""

    class ExecuteDispatchAttemptOptions(
        FlextModelsConfig.ExecuteDispatchAttemptOptions,
    ):
        """Execute dispatch attempt options - direct class for mypy compatibility."""

    class RuntimeScopeOptions(FlextModelsConfig.RuntimeScopeOptions):
        """Runtime scope options — real re-export for pydantic-mypy compatibility."""

    class NestedExecutionOptions(FlextModelsConfig.NestedExecutionOptions):
        """Nested execution options — real re-export for pydantic-mypy compatibility."""

    class ExceptionConfig(FlextModelsConfig.ExceptionConfig):
        """Exception config — real re-export for pydantic-mypy compatibility."""

    class ValidationErrorConfig(FlextModelsConfig.ValidationErrorConfig):
        """Validation error config — real re-export for pydantic-mypy compatibility."""

    class ConfigurationErrorConfig(FlextModelsConfig.ConfigurationErrorConfig):
        """Configuration error config — real re-export for pydantic-mypy compatibility."""

    class ConnectionErrorConfig(FlextModelsConfig.ConnectionErrorConfig):
        """Connection error config — real re-export for pydantic-mypy compatibility."""

    class TimeoutErrorConfig(FlextModelsConfig.TimeoutErrorConfig):
        """Timeout error config — real re-export for pydantic-mypy compatibility."""

    class AuthenticationErrorConfig(FlextModelsConfig.AuthenticationErrorConfig):
        """Authentication error config — real re-export for pydantic-mypy compatibility."""

    class AuthorizationErrorConfig(FlextModelsConfig.AuthorizationErrorConfig):
        """Authorization error config — real re-export for pydantic-mypy compatibility."""

    class NotFoundErrorConfig(FlextModelsConfig.NotFoundErrorConfig):
        """Not found error config — real re-export for pydantic-mypy compatibility."""

    class ConflictErrorConfig(FlextModelsConfig.ConflictErrorConfig):
        """Conflict error config — real re-export for pydantic-mypy compatibility."""

    class RateLimitErrorConfig(FlextModelsConfig.RateLimitErrorConfig):
        """Rate limit error config — real re-export for pydantic-mypy compatibility."""

    class InternalErrorConfig(FlextModelsConfig.InternalErrorConfig):
        """Internal error config — real re-export for pydantic-mypy compatibility."""

    class TypeErrorOptions(FlextModelsConfig.TypeErrorOptions):
        """Type error options — real re-export for pydantic-mypy compatibility."""

    class TypeErrorConfig(FlextModelsConfig.TypeErrorConfig):
        """Type error config — real re-export for pydantic-mypy compatibility."""

    class ValueErrorConfig(FlextModelsConfig.ValueErrorConfig):
        """Value error config — real re-export for pydantic-mypy compatibility."""

    class CircuitBreakerErrorConfig(FlextModelsConfig.CircuitBreakerErrorConfig):
        """Circuit breaker error config — real re-export for pydantic-mypy compatibility."""

    class OperationErrorConfig(FlextModelsConfig.OperationErrorConfig):
        """Operation error config — real re-export for pydantic-mypy compatibility."""

    class AttributeAccessErrorConfig(FlextModelsConfig.AttributeAccessErrorConfig):
        """Attribute access error config — real re-export for pydantic-mypy compatibility."""

    class MiddlewareConfig(FlextModelsConfig.MiddlewareConfig):
        """Middleware config — real re-export for pydantic-mypy compatibility."""

    class RateLimiterState(FlextModelsConfig.RateLimiterState):
        """Rate limiter state — real re-export for pydantic-mypy compatibility."""

    # =========================================================================
    # BASE CLASSES - Direct access for common usage
    # =========================================================================

    ArbitraryTypesModel = FlextModelFoundation.ArbitraryTypesModel
    StrictBoundaryModel = FlextModelFoundation.StrictBoundaryModel
    FrozenStrictModel = FlextModelFoundation.FrozenStrictModel
    TaggedModel = FlextModelFoundation.TaggedModel
    IdentifiableMixin = FlextModelFoundation.IdentifiableMixin
    TimestampableMixin = FlextModelFoundation.TimestampableMixin
    TimestampedModel = FlextModelFoundation.TimestampedModel
    VersionableMixin = FlextModelFoundation.VersionableMixin
    Metadata = FlextModelFoundation.Metadata

    # =========================================================================
    # HANDLER MODELS - Direct access for common usage
    # =========================================================================

    class Handler(FlextModelsCqrs.Handler):
        """Handler base class - real inheritance."""

        RegistrationDetails: ClassVar[type[FlextModelsHandler.RegistrationDetails]] = (
            FlextModelsHandler.RegistrationDetails
        )
        RegistrationResult: ClassVar[type[FlextModelsHandler.RegistrationResult]] = (
            FlextModelsHandler.RegistrationResult
        )
        RegistrationRequest: ClassVar[type[FlextModelsHandler.RegistrationRequest]] = (
            FlextModelsHandler.RegistrationRequest
        )
        ExecutionContext: ClassVar[type[FlextModelsHandler.ExecutionContext]] = (
            FlextModelsHandler.ExecutionContext
        )
        DecoratorConfig: ClassVar[type[FlextModelsHandler.DecoratorConfig]] = (
            FlextModelsHandler.DecoratorConfig
        )
        FactoryDecoratorConfig: ClassVar[
            type[FlextModelsContainer.FactoryDecoratorConfig]
        ] = FlextModelsContainer.FactoryDecoratorConfig

    # Direct aliases for top-level access
    HandlerDecoratorConfig = Handler.DecoratorConfig
    HandlerFactoryDecoratorConfig = Handler.FactoryDecoratorConfig
    HandlerRegistrationDetails = Handler.RegistrationDetails
    HandlerRegistrationResult = Handler.RegistrationResult
    HandlerExecutionContext = Handler.ExecutionContext
    HandlerRegistrationRequest = Handler.RegistrationRequest

    class CqrsHandler(Handler):
        """CQRS handler — real re-export for pydantic-mypy compatibility."""

    # =========================================================================
    # MIXIN MODELS - State models for FlextMixins infrastructure
    # =========================================================================

    class Mixin(FlextModelsMixin):
        """Mixin state models namespace for FlextMixins infrastructure.

        The runtime triple (config, context, container) is ``m.ServiceRuntime``.
        This namespace adds mixin-specific state models not covered by it.

        Models:
            OperationStats: Accumulated metrics from ``x.track()`` calls.
        """

    # =========================================================================
    # DECORATOR MODELS - Direct access for common usage
    # =========================================================================

    class Decorator(FlextModelsDecorators):
        """Decorator configuration models namespace.

        Re-exports FlextModelsDecorators as a proper class for mypy compatibility.
        """

        TimeoutConfig: type[FlextModelsDecorators.TimeoutConfig] = (
            FlextModelsDecorators.TimeoutConfig
        )

    # =========================================================================
    # UNIONS - Pydantic discriminated unions
    # =========================================================================

    Message: TypeAlias = FlextModelsCqrs.FlextMessage


# =========================================================================
# MODULE ALIASES - Runtime access patterns
# =========================================================================

# Main alias for direct access
m = FlextModels
_typing_namespace_t = t

__all__ = ["FlextModels", "m"]
