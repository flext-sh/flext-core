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

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Annotated

from pydantic import Discriminator, Field

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
    class Entity(FlextModelsEntity.Core):
        """Entity base class with real inheritance."""

    class Value(FlextModelsEntity.Value):
        """Value object base class with real inheritance."""

    class AggregateRoot(Entity, FlextModelsEntity.AggregateRoot):
        """Aggregate root base class with real inheritance.

        Inherits from both Entity (for hierarchy) and FlextModelsEntity.AggregateRoot
        (for functionality) to maintain correct inheritance chain.
        """

    class DomainEvent(FlextModelsEntity.DomainEvent):
        """Domain event base class with real inheritance."""

    # Direct base class - real inheritance
    class ArbitraryTypesModel(FlextModelsBase.ArbitraryTypesModel):
        """Base model with arbitrary types support - real inheritance."""

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

    # Domain Service Models - Real inheritance classes
    # ServiceRuntime needs to stay as class due to protocol fields
    class ServiceRuntime(FlextModelsBase.ArbitraryTypesModel):
        """Runtime triple (config, context, container) for services."""

        config: p.Configuration.Config
        context: p.Context.Ctx
        container: p.Container.DI

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
        FlextModelsValidation. Methods return Foundation.Result[T] for various T.
        """

    # =========================================================================
    # ROOT-LEVEL ALIASES (Minimize nesting for common models)
    # Usage: m.Command instead of m.Cqrs.Command
    # Both access patterns work - aliases for convenience, namespaces for clarity
    # =========================================================================

    # CQRS aliases (most common)
    Command = Cqrs.Command
    Query = Cqrs.Query
    Pagination = Cqrs.Pagination
    CqrsBus = Cqrs.Bus
    CqrsHandler = Cqrs.Handler

    # Config aliases (frequently used)
    ProcessingRequest = Config.ProcessingRequest
    RetryConfiguration = Config.RetryConfiguration
    BatchProcessingConfig = Config.BatchProcessingConfig
    DispatchConfig = Config.DispatchConfig
    MiddlewareConfig = Config.MiddlewareConfig
    ExceptionConfig = Config.ExceptionConfig
    ValidationConfiguration = Config.ValidationConfiguration
    HandlerExecutionConfig = Config.HandlerExecutionConfig
    NestedExecutionOptions = Config.NestedExecutionOptions
    RuntimeScopeOptions = Config.RuntimeScopeOptions

    # Context aliases
    ContextData = Context.ContextData
    ContextExport = Context.ContextExport
    ContextScopeData = Context.ContextScopeData
    ContextStatistics = Context.ContextStatistics
    ContextMetadata = Context.ContextMetadata
    Token = Context.Token

    # Handler aliases
    HandlerRegistration = Handler.Registration
    HandlerRegistrationDetails = Handler.RegistrationDetails
    HandlerExecutionContext = Handler.ExecutionContext

    # Service aliases
    DomainServiceExecutionRequest = Service.DomainServiceExecutionRequest
    DomainServiceBatchRequest = Service.DomainServiceBatchRequest
    OperationExecutionRequest = Service.OperationExecutionRequest
    AclResponse = Service.AclResponse

    # Container aliases
    ServiceRegistration = Container.ServiceRegistration
    FactoryRegistration = Container.FactoryRegistration
    ResourceRegistration = Container.ResourceRegistration
    ContainerConfig = Container.ContainerConfig

    # Collections aliases
    CollectionsConfig = Collections.Config
    CollectionsRules = Collections.Rules
    CollectionsStatistics = Collections.Statistics
    CollectionsResults = Collections.Results
    CollectionsOptions = Collections.Options

    # =========================================================================
    # PROJECT-SPECIFIC NAMESPACES (Python 3.13+ PEP 695 type organization)
    # =========================================================================
    # These namespaces are populated by their respective projects:
    # - Ldif: Populated by flext-ldif (FlextLdifModels)
    # - Ldap: Populated by flext-ldap (FlextLdapModels)
    # - Cli: Populated by flext-cli (FlextCliModels)
    # - Api: Populated by flext-api (FlextApiModels)
    #
    # Architecture:
    # - Each namespace is a class that can be extended by its project
    # - Projects populate their namespace by extending these classes
    # - Allows cross-project access: flext-ldap can access flext-ldif via m.Ldif.*
    # - Maintains backward compatibility with project-specific Models classes
    # =========================================================================

    class Ldif:
        """LDIF project namespace - populated by flext-ldif.

        This namespace contains all LDIF-specific models from flext-ldif.
        Access via: m.Ldif.Entry, m.Ldif.SchemaAttribute, etc.

        Populated by: flext-ldif/src/flext_ldif/models.py
        Models are populated dynamically after FlextLdifModels is defined.
        """

    class Ldap:
        """LDAP project namespace - populated by flext-ldap.

        This namespace contains all LDAP-specific models from flext-ldap.
        Access via: m.Ldap.ConnectionConfig, m.Ldap.SearchOptions, etc.

        Populated by: flext-ldap/src/flext_ldap/models.py
        Can access flext-ldif models via: m.Ldif.*
        """

    class Cli:
        """CLI project namespace - populated by flext-cli.

        This namespace contains all CLI-specific models from flext-cli.
        Access via: m.Cli.CliCommand, m.Cli.CommandResult, etc.

        Populated by: flext-cli/src/flext_cli/models.py
        """

    class Api:
        """API project namespace - populated by flext-api.

        This namespace contains all HTTP/API-specific models from flext-api.
        Access via: m.Api.HttpRequest, m.Api.HttpResponse, etc.

        Populated by: flext-api/src/flext_api/models.py
        """


# =============================================================================
# Pydantic v2 Forward Reference Resolution
# =============================================================================
# When using 'from __future__ import annotations' with nested type aliases
# (like t.Types.ConfigurationDict inside nested classes), Pydantic cannot
# automatically resolve the forward references because 't' is not available
# in the nested class's evaluation namespace.
#
# We explicitly call model_rebuild() with the proper types namespace to
# ensure all models can be instantiated correctly.
# =============================================================================

_TYPES_NS = {
    "t": t,
    "p": p,
    "Callable": Callable,
    "Mapping": Mapping,
    "Sequence": Sequence,
    "datetime": datetime,
    "Field": Field,
    "FlextModelsBase": FlextModelsBase,
}

# Base models (must be rebuilt first as others depend on them)
FlextModelsBase.Metadata.model_rebuild(_types_namespace=_TYPES_NS)

# Entity models
FlextModelsEntity.Value.model_rebuild(_types_namespace=_TYPES_NS)
# Note: Entity was renamed to Core - using Core for model_rebuild
FlextModelsEntity.Core.model_rebuild(_types_namespace=_TYPES_NS)

# Context models (depend on Base.Metadata)
FlextModelsContext.ContextData.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsContext.ContextExport.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsContext.ContextScopeData.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsContext.ContextStatistics.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsContext.StructlogProxyToken.model_rebuild(_types_namespace=_TYPES_NS)

# Config models
FlextModelsConfig.ProcessingRequest.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsConfig.RetryConfiguration.model_rebuild(_types_namespace=_TYPES_NS)
# Note: ErrorMappingConfiguration and TestServiceConfiguration may not exist
# Only rebuild models that actually exist - verify before uncommenting
# FlextModelsConfig.ErrorMappingConfiguration.model_rebuild(_types_namespace=_TYPES_NS)
# FlextModelsConfig.TestServiceConfiguration.model_rebuild(_types_namespace=_TYPES_NS)

# Service models - only rebuild classes that actually exist
FlextModelsService.OperationExecutionRequest.model_rebuild(_types_namespace=_TYPES_NS)
# ServiceRuntime, DomainServiceRequest, DomainServiceResponse don't exist - commented out
# FlextModelsService.ServiceRuntime.model_rebuild(_types_namespace=_TYPES_NS)
# FlextModelsService.DomainServiceRequest.model_rebuild(_types_namespace=_TYPES_NS)

# Collections models - rebuild ParseOptions and PatternApplicationParams
FlextModelsCollections.ParseOptions.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsCollections.PatternApplicationParams.model_rebuild(
    _types_namespace=_TYPES_NS
)
# FlextModelsService.DomainServiceResponse.model_rebuild(_types_namespace=_TYPES_NS)

# Handler models - using Registration (not HandlerRegistration)
FlextModelsHandler.Registration.model_rebuild(_types_namespace=_TYPES_NS)

# Container models
FlextModelsContainer.ServiceRegistration.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsContainer.FactoryRegistration.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsContainer.ContainerConfig.model_rebuild(_types_namespace=_TYPES_NS)

# CQRS models
FlextModelsCqrs.Command.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsCqrs.Query.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsCqrs.Pagination.model_rebuild(_types_namespace=_TYPES_NS)
# Note: DomainEvent is aliased from FlextModelsEntity.DomainEvent, not FlextModelsCqrs
# FlextModelsCqrs.DomainEvent.model_rebuild(_types_namespace=_TYPES_NS)

# Validation models
# Note: Verify which Validation classes actually exist before rebuilding
# Only rebuild models that actually exist - commented out for now
# FlextModelsValidation.ValidationRule.model_rebuild(_types_namespace=_TYPES_NS)
# FlextModelsValidation.ValidationResult.model_rebuild(_types_namespace=_TYPES_NS)

# Collections models
FlextModelsCollections.Config.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsCollections.Rules.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsCollections.Statistics.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsCollections.Results.model_rebuild(_types_namespace=_TYPES_NS)
FlextModelsCollections.Options.model_rebuild(_types_namespace=_TYPES_NS)

m = FlextModels
__all__ = ["FlextModels", "FlextModelsCollections", "FlextModelsEntity", "m"]
