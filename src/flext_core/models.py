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

    # Entity & DDD Patterns - Class references for inheritance support
    # Use class attributes instead of type aliases to allow inheritance
    Entity = FlextModelsEntity.Core
    Value = FlextModelsEntity.Value
    AggregateRoot = FlextModelsEntity.AggregateRoot
    DomainEvent = FlextModelsEntity.DomainEvent

    # Direct alias to base class - no wrapper to avoid MRO duplication
    # Use FlextModelsBase.ArbitraryTypesModel directly in inheritance
    # Use class attributes for inheritance support (type aliases can't be base classes)
    ArbitraryTypesModel = FlextModelsBase.ArbitraryTypesModel

    # Base Models - Class references for inheritance support
    # Use class attributes instead of type aliases to allow inheritance
    FrozenStrictModel = FlextModelsBase.FrozenStrictModel
    IdentifiableMixin = FlextModelsBase.IdentifiableMixin
    TimestampableMixin = FlextModelsBase.TimestampableMixin
    TimestampedModel = FlextModelsBase.TimestampedModel
    VersionableMixin = FlextModelsBase.VersionableMixin

    # Collections - PEP 695 type aliases (Python 3.13+ strict)
    # Direct access - no nested namespace duplication per FLEXT standards
    Categories = FlextModelsCollections.Categories[t.GeneralValueType]
    type Statistics = FlextModelsCollections.Statistics
    type Config = FlextModelsCollections.Config
    type Results = FlextModelsCollections.Results
    type Rules = FlextModelsCollections.Rules
    type Options = FlextModelsCollections.Options
    type ParseOptions = FlextModelsCollections.ParseOptions

    # CQRS Patterns - Class references for isinstance checks
    class Cqrs:
        """CQRS namespace with nested classes."""

        Command = FlextModelsCqrs.Command
        Pagination = FlextModelsCqrs.Pagination  # Used in isinstance checks
        Query = FlextModelsCqrs.Query
        Bus = FlextModelsCqrs.Bus
        Handler = FlextModelsCqrs.Handler

        # NOTE: Use FlextConfig.get_global_instance() directly in model defaults
        # No wrapper methods needed - access config directly per FLEXT standards

    # Base Utility Models - PEP 695 type aliases (Python 3.13+ strict)
    type Metadata = FlextModelsBase.Metadata

    # Configuration Models - PEP 695 type aliases (Python 3.13+ strict)
    type ProcessingRequest = FlextModelsConfig.ProcessingRequest
    type RetryConfiguration = FlextModelsConfig.RetryConfiguration
    type ValidationConfiguration = FlextModelsConfig.ValidationConfiguration
    type BatchProcessingConfig = FlextModelsConfig.BatchProcessingConfig
    type HandlerExecutionConfig = FlextModelsConfig.HandlerExecutionConfig
    type OperationExtraConfig = FlextModelsConfig.OperationExtraConfig
    type LogOperationFailureConfig = FlextModelsConfig.LogOperationFailureConfig
    type RetryLoopConfig = FlextModelsConfig.RetryLoopConfig
    type DispatchConfig = FlextModelsConfig.DispatchConfig
    type ExecuteDispatchAttemptOptions = FlextModelsConfig.ExecuteDispatchAttemptOptions
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

    # Context Management Models - Class references for isinstance/isinstance checks
    # Direct access - no nested namespace duplication per FLEXT standards
    StructlogProxyToken = FlextModelsContext.StructlogProxyToken
    StructlogProxyContextVar = FlextModelsContext.StructlogProxyContextVar[str]
    Token = FlextModelsContext.Token
    ContextData = FlextModelsContext.ContextData
    ContextExport = FlextModelsContext.ContextExport  # Used in isinstance checks
    ContextScopeData = FlextModelsContext.ContextScopeData
    ContextStatistics = (
        FlextModelsContext.ContextStatistics
    )  # Used in isinstance checks
    ContextMetadata = FlextModelsContext.ContextMetadata
    ContextDomainData = FlextModelsContext.ContextDomainData

    # Handler Management Models - PEP 695 type aliases (Python 3.13+ strict)
    type HandlerRegistration = FlextModelsHandler.Registration
    type RegistrationDetails = FlextModelsHandler.RegistrationDetails
    type HandlerExecutionContext = FlextModelsHandler.ExecutionContext

    # Domain Service Models - PEP 695 type aliases (Python 3.13+ strict)
    # ServiceRuntime needs to stay as class due to protocol fields
    class ServiceRuntime(FlextModelsBase.ArbitraryTypesModel):
        """Runtime triple (config, context, container) for services."""

        config: p.ConfigProtocol
        context: p.ContextProtocol
        container: p.ContainerProtocol

    type DomainServiceExecutionRequest = (
        FlextModelsService.DomainServiceExecutionRequest
    )
    type DomainServiceBatchRequest = FlextModelsService.DomainServiceBatchRequest
    type DomainServiceMetricsRequest = FlextModelsService.DomainServiceMetricsRequest
    type DomainServiceResourceRequest = FlextModelsService.DomainServiceResourceRequest
    type AclResponse = FlextModelsService.AclResponse
    type OperationExecutionRequest = FlextModelsService.OperationExecutionRequest

    # Container Models - PEP 695 type aliases (Python 3.13+ strict)
    type ServiceRegistration = FlextModelsContainer.ServiceRegistration
    type FactoryRegistration = FlextModelsContainer.FactoryRegistration
    type ContainerConfig = FlextModelsContainer.ContainerConfig

    # Pydantic v2 discriminated union using modern typing (PEP 695)
    # Use direct references to base classes, not aliases
    type MessageUnion = Annotated[
        FlextModelsCqrs.Command | FlextModelsCqrs.Query | FlextModelsEntity.DomainEvent,
        Discriminator("message_type"),
    ]

    # Validation Patterns - Direct method delegation (no wrapper mapping)
    class Validation:
        """Validation namespace with direct method delegation."""

        # Direct assignment - no wrapper mapping per FLEXT standards
        # All methods return ResultProtocol[T] for various T
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
        validate_entity_relationships = (
            FlextModelsValidation.validate_entity_relationships
        )


# Pydantic v2 with 'from __future__ import annotations' automatically resolves forward references
# No manual model_rebuild() needed - annotations are stringified and resolved at runtime
m = FlextModels
__all__ = ["FlextModels", "FlextModelsCollections", "FlextModelsEntity", "m"]
