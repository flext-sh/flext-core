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

from collections.abc import Callable, Mapping
from typing import Annotated, ClassVar

from pydantic import Discriminator

from flext_core._models.base import FlextModelsBase
from flext_core._models.collections import FlextModelsCollections
from flext_core._models.config import FlextModelsConfig
from flext_core._models.container import FlextModelsContainer
from flext_core._models.context import FlextModelsContext
from flext_core._models.cqrs import FlextModelsCqrs
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.handler import FlextModelsHandler
from flext_core._models.metadata import Metadata as MetadataBase
from flext_core._models.service import FlextModelsService
from flext_core._models.validation import FlextModelsValidation
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants


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

    # Entity & DDD Patterns
    class Entity(FlextModelsEntity.Core):
        """Domain entity with identity and lifecycle."""

    class Value(FlextModelsEntity.Value):
        """Immutable value object."""

    class AggregateRoot(FlextModelsEntity.AggregateRoot, Entity):
        """Aggregate root consistency boundary."""

    class DomainEvent(FlextModelsEntity.DomainEvent):
        """Domain event for event sourcing."""

    class ArbitraryTypesModel(FlextModelsEntity.ArbitraryTypesModel):
        """Base model with arbitrary types support."""

    class FrozenStrictModel(FlextModelsEntity.FrozenStrictModel):
        """Immutable strict model."""

    class IdentifiableMixin(FlextModelsEntity.IdentifiableMixin):
        """Mixin for unique identifiers."""

    class TimestampableMixin(FlextModelsEntity.TimestampableMixin):
        """Mixin for timestamps."""

    class TimestampedModel(FlextModelsEntity.TimestampedModel):
        """Model with timestamp fields."""

    class VersionableMixin(FlextModelsEntity.VersionableMixin):
        """Mixin for versioning."""

    # Collections
    Categories = FlextModelsCollections.Categories

    class Statistics(FlextModelsCollections.Statistics):
        """Statistics model with common counters."""

    class Config(FlextModelsCollections.Config):
        """Configuration model with common fields."""

    class Results(FlextModelsCollections.Results):
        """Base for result models."""

    class Rules(FlextModelsCollections.Rules):
        """Rules model for configuration rules."""

    class Options(FlextModelsCollections.Options):
        """Options model for configuration options."""

    # CQRS Patterns
    class Cqrs:
        """CQRS namespace with nested classes."""

        class Command(FlextModelsCqrs.Command):
            """Base class for CQRS commands."""

        class Pagination(FlextModelsCqrs.Pagination):
            """Pagination model for query results."""

        class Query(FlextModelsCqrs.Query):
            """Query model for CQRS query operations."""

        class Bus(FlextModelsCqrs.Bus):
            """Dispatcher configuration model exposed via CQRS namespace."""

        class Handler(FlextModelsCqrs.Handler):
            """Handler configuration model."""

        @staticmethod
        def _get_command_timeout_default() -> int:
            """Get command timeout from config or constants."""
            config = FlextConfig.get_global_instance()
            return (
                int(config.dispatcher_timeout_seconds)
                or FlextConstants.Cqrs.DEFAULT_COMMAND_TIMEOUT
            )

        @staticmethod
        def _get_max_command_retries_default() -> int:
            """Get max retries from config or constants."""
            config = FlextConfig.get_global_instance()
            return (
                config.max_retry_attempts
                or FlextConstants.Cqrs.DEFAULT_MAX_COMMAND_RETRIES
            )

    # Base Utility Models
    class Metadata(MetadataBase):
        """Metadata model for structured information."""

    Payload = FlextModelsBase.Payload

    class Url(FlextModelsBase.Url):
        """URL model with validation."""

    class LogOperation(FlextModelsBase.LogOperation):
        """Log operation model."""

    class TimestampConfig(FlextModelsBase.TimestampConfig):
        """Timestamp configuration model."""

    class SerializationRequest(FlextModelsBase.SerializationRequest):
        """Serialization request model."""

    class ConditionalExecutionRequest(FlextModelsBase.ConditionalExecutionRequest):
        """Conditional execution request model."""

    class StateInitializationRequest(FlextModelsBase.StateInitializationRequest):
        """State initialization request model."""

    # Configuration Models
    class ProcessingRequest(FlextModelsConfig.ProcessingRequest):
        """Processing request configuration model."""

    class RetryConfiguration(FlextModelsConfig.RetryConfiguration):
        """Retry configuration model."""

    class ValidationConfiguration(FlextModelsConfig.ValidationConfiguration):
        """Validation configuration model."""

    class BatchProcessingConfig(FlextModelsConfig.BatchProcessingConfig):
        """Batch processing configuration model."""

    class HandlerExecutionConfig(FlextModelsConfig.HandlerExecutionConfig):
        """Handler execution configuration model."""

    class MiddlewareConfig(FlextModelsConfig.MiddlewareConfig):
        """Middleware configuration model."""

    class RateLimiterState(FlextModelsConfig.RateLimiterState):
        """Rate limiter state model."""

    # Context Management Models
    class StructlogProxyToken(FlextModelsContext.StructlogProxyToken):
        """Structlog proxy token model."""

    StructlogProxyContextVar = FlextModelsContext.StructlogProxyContextVar

    class Token(FlextModelsContext.Token):
        """Context token model."""

    class ContextData(FlextModelsContext.ContextData):
        """Context data model."""

    class ContextExport(FlextModelsContext.ContextExport):
        """Context export model."""

    class ContextScopeData(FlextModelsContext.ContextScopeData):
        """Context scope data model."""

    class ContextStatistics(FlextModelsContext.ContextStatistics):
        """Context statistics model."""

    class ContextMetadata(FlextModelsContext.ContextMetadata):
        """Context metadata model."""

    class ContextDomainData(FlextModelsContext.ContextDomainData):
        """Context domain data model."""

    class Context:
        """Context management facade with DRY mapping."""

        # DRY mapping for context classes
        _CONTEXT_CLASSES: ClassVar[Mapping[str, type]] = {
            "StructlogProxyToken": FlextModelsContext.StructlogProxyToken,
            "StructlogProxyContextVar": FlextModelsContext.StructlogProxyContextVar,
            "Token": FlextModelsContext.Token,
            "ContextData": FlextModelsContext.ContextData,
            "ContextExport": FlextModelsContext.ContextExport,
            "ContextScopeData": FlextModelsContext.ContextScopeData,
            "ContextStatistics": FlextModelsContext.ContextStatistics,
            "ContextMetadata": FlextModelsContext.ContextMetadata,
            "ContextDomainData": FlextModelsContext.ContextDomainData,
        }

        # DRY assignment
        StructlogProxyToken = _CONTEXT_CLASSES["StructlogProxyToken"]
        StructlogProxyContextVar = _CONTEXT_CLASSES["StructlogProxyContextVar"]
        Token = _CONTEXT_CLASSES["Token"]
        ContextData = _CONTEXT_CLASSES["ContextData"]
        ContextExport = _CONTEXT_CLASSES["ContextExport"]
        ContextScopeData = _CONTEXT_CLASSES["ContextScopeData"]
        ContextStatistics = _CONTEXT_CLASSES["ContextStatistics"]
        ContextMetadata = _CONTEXT_CLASSES["ContextMetadata"]
        ContextDomainData = _CONTEXT_CLASSES["ContextDomainData"]

    # Handler Management Models
    class HandlerRegistration(FlextModelsHandler.Registration):
        """Handler registration model."""

    class RegistrationDetails(FlextModelsHandler.RegistrationDetails):
        """Registration details model."""

    class HandlerExecutionContext(FlextModelsHandler.ExecutionContext):
        """Handler execution context model."""

    # Domain Service Models
    class DomainServiceExecutionRequest(
        FlextModelsService.DomainServiceExecutionRequest
    ):
        """Domain service execution request model."""

    class DomainServiceBatchRequest(FlextModelsService.DomainServiceBatchRequest):
        """Domain service batch request model."""

    class DomainServiceMetricsRequest(FlextModelsService.DomainServiceMetricsRequest):
        """Domain service metrics request model."""

    class DomainServiceResourceRequest(FlextModelsService.DomainServiceResourceRequest):
        """Domain service resource request model."""

    class OperationExecutionRequest(FlextModelsService.OperationExecutionRequest):
        """Operation execution request model."""

    # Container Models
    class ServiceRegistration(FlextModelsContainer.ServiceRegistration):
        """Service registration model - DI container service entry."""

    class FactoryRegistration(FlextModelsContainer.FactoryRegistration):
        """Factory registration model - DI container factory entry."""

    class ContainerConfig(FlextModelsContainer.ContainerConfig):
        """Container configuration model - DI container settings."""

    # Pydantic v2 discriminated union using modern typing (PEP 695)
    type MessageUnion = Annotated[
        Cqrs.Command | Cqrs.Query | DomainEvent, Discriminator("message_type")
    ]

    # Validation Patterns - DRY mapping for method delegation
    class Validation:
        """Validation namespace with DRY method delegation."""

        # Mapping for DRY validation method assignment
        _VALIDATION_METHODS: ClassVar[Mapping[str, Callable[..., object]]] = {
            "validate_business_rules": FlextModelsValidation.validate_business_rules,
            "validate_cross_fields": FlextModelsValidation.validate_cross_fields,
            "validate_performance": FlextModelsValidation.validate_performance,
            "validate_batch": FlextModelsValidation.validate_batch,
            "validate_domain_invariants": FlextModelsValidation.validate_domain_invariants,
            "validate_aggregate_consistency_with_rules": FlextModelsValidation.validate_aggregate_consistency_with_rules,
            "validate_event_sourcing": FlextModelsValidation.validate_event_sourcing,
            "validate_cqrs_patterns": FlextModelsValidation.validate_cqrs_patterns,
            "validate_domain_event": FlextModelsValidation.validate_domain_event,
            "validate_aggregate_consistency": FlextModelsValidation.validate_aggregate_consistency,
            "validate_entity_relationships": FlextModelsValidation.validate_entity_relationships,
        }

        # DRY assignment using mapping
        validate_business_rules = _VALIDATION_METHODS["validate_business_rules"]
        validate_cross_fields = _VALIDATION_METHODS["validate_cross_fields"]
        validate_performance = _VALIDATION_METHODS["validate_performance"]
        validate_batch = _VALIDATION_METHODS["validate_batch"]
        validate_domain_invariants = _VALIDATION_METHODS["validate_domain_invariants"]
        validate_aggregate_consistency_with_rules = _VALIDATION_METHODS[
            "validate_aggregate_consistency_with_rules"
        ]
        validate_event_sourcing = _VALIDATION_METHODS["validate_event_sourcing"]
        validate_cqrs_patterns = _VALIDATION_METHODS["validate_cqrs_patterns"]
        validate_domain_event = _VALIDATION_METHODS["validate_domain_event"]
        validate_aggregate_consistency = _VALIDATION_METHODS[
            "validate_aggregate_consistency"
        ]
        validate_entity_relationships = _VALIDATION_METHODS[
            "validate_entity_relationships"
        ]


# Pydantic v2 with 'from __future__ import annotations' automatically resolves forward references
# No manual model_rebuild() needed - annotations are stringified and resolved at runtime

__all__ = ["FlextModels", "FlextModelsCollections"]
