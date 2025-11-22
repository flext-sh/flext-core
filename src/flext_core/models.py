"""Domain models for domain-driven design patterns.

This module provides FlextModels, a comprehensive collection of base classes
and utilities for implementing domain-driven design (DDD) patterns in the
FLEXT ecosystem.

All models use Pydantic for validation and serialization, providing type-safe
domain modeling with automatic validation and error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import Annotated, TypeAlias

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
from flext_core.constants import FlextConstants


class FlextModels:
    """Domain-Driven Design (DDD) patterns with Pydantic validation.

    Architecture: Layer 2 (Domain)
    ==============================
    Provides comprehensive base classes for implementing Domain-Driven Design
    patterns with Pydantic v2 validation, event sourcing support, and CQRS
    integration across the FLEXT ecosystem.

    Structural Typing and Protocol Compliance:
    ===========================================
    FlextModels implements multiple FlextProtocols interfaces via structural
    typing (duck typing) through nested classes:

    - Entity (satisfies FlextProtocols.Entity):
      * id: Unique identifier for entity tracking
      * created_at: Creation timestamp
      * updated_at: Modification timestamp
      * is_valid(): Validate entity state
      * to_dict(): Serialize to dictionary

    - Value (satisfies FlextProtocols.Value):
      * Immutable value objects (frozen Pydantic models)
      * Compared by value, not identity
      * No mutable state after creation
      * Hashable for use in collections

    - AggregateRoot (satisfies FlextProtocols.AggregateRoot):
      * Consistency boundary enforcement
      * Transactional invariant protection
      * Event sourcing support
      * Domain event publishing

    - Command (satisfies FlextProtocols.Command):
      * Represents domain operations
      * Command validation
      * Handler mapping
      * Idempotency support

    - Query (satisfies FlextProtocols.Query):
      * Read-side operations
      * Non-mutating operations
      * Result projection
      * Caching support

    - DomainEvent (satisfies FlextProtocols.DomainEvent):
      * Event sourcing backbone
      * Immutable past events
      * Event replay support
      * Audit trail

    Core DDD Concepts:
    ==================
    1. **Entity**: Domain object with identity and lifecycle
       - Changes tracked through updated_at
       - Compared by id, not value equality
       - Supports domain logic and invariants
       - Integrated with FlextResult for operations

    2. **Value Object**: Immutable domain values
       - No identity (compared by value)
       - Immutable after creation (frozen Pydantic)
       - Composable building blocks
       - Hashable for collections

    3. **Aggregate Root**: Consistency boundary
       - Contains entities and value objects
       - Enforces transactional invariants
       - Single root for external references
       - Event sourcing support

    4. **Command**: Domain operation request
       - Represents "I want X to happen"
       - Immutable command object
       - Handler determines execution
       - Async command bus support

    5. **Query**: Domain read operation
       - Represents "I want to know X"
       - Non-mutating read projection
       - Result optimization via caching
       - Query bus support

    6. **Domain Event**: Significant domain occurrence
       - Represents "X happened"
       - Event sourcing backbone
       - Event replay for reconstruction
       - Audit trail support

    Pydantic v2 Integration:
    =======================
    - Full Pydantic BaseModel support
    - Automatic validation via field_validator
    - Model validation via model_validator
    - Computed fields for derived properties
    - Custom serializers for domain logic
    - Config inheritance via ConfigDict
    - Immutable models (frozen=True)

    Features and Components:
    ========================
    - Entity: Base domain entity with lifecycle
    - Value: Immutable value objects
    - AggregateRoot: Consistency boundary root
    - Command: CQRS command pattern
    - Query: CQRS query pattern
    - DomainEvent: Event sourcing events
    - Validation: Business rule validators
    - Timestamps: Automatic created_at/updated_at
    - Serialization: JSON and dict conversion
    - Type validation: Complete type safety

    Advanced Patterns:
    ==================
    - Event Sourcing: Replay events to reconstruct state
    - CQRS: Separate read/write models with Command/Query
    - Transactional Invariants: Enforce business rules
    - Aggregate Roots: Consistency boundary enforcement
    - Value Objects: Rich domain types
    - Domain Events: Capture domain state changes

    Error Handling:
    ===============
    - FlextResult[object] wrapping for operations
    - Validation errors caught in is_valid()
    - Business rule violations in invariants
    - Structured error information

    Usage Patterns:
    ===============
        >>> from flext_core.models import FlextModels
        >>> from flext_core.result import FlextResult
        >>>
        >>> # Value Object - immutable by design
        >>> from pydantic import EmailStr
        >>> class Email(FlextModels.Value):
        ...     address: EmailStr  # Pydantic v2 EmailStr validates format natively
        >>>
        >>> # Entity - with identity and lifecycle
        >>> class User(FlextModels.Entity):
        ...     name: str
        ...     email: Email
        ...     is_active: bool = False
        ...
        ...     def activate(self) -> FlextResult[bool]:
        ...         if self.is_active:
        ...             return FlextResult.fail("Already active")
        ...         self.is_active = True
        ...         return FlextResult[bool].ok(True)
        >>>
        >>> # Aggregate Root - consistency boundary
        >>> class Account(FlextModels.AggregateRoot):
        ...     owner: User
        ...     balance: float = 0.0
        ...
        ...     def deposit(self, amount: float) -> FlextResult[bool]:
        ...         if amount <= 0:
        ...             return FlextResult.fail("Amount must be positive")
        ...         self.balance += amount
        ...         return FlextResult[bool].ok(True)
        >>>
        >>> # Command - CQRS pattern
        >>> class CreateUserCommand(FlextModels.Cqrs.Command):
        ...     name: str
        ...     email: str
        >>>
        >>> # Query - CQRS pattern
        >>> class GetUserQuery(FlextModels.Cqrs.Query):
        ...     user_id: str
        >>>
        >>> # Domain Event - Event sourcing
        >>> class UserCreatedEvent(FlextModels.DomainEvent):
        ...     user_id: str
        ...     name: str
        ...     email: str

    Integration with FLEXT Ecosystem:
    ==================================
    - Service Layer: Services receive FlextResult[object] from models
    - Handler Layer: CQRS handlers process Commands/Queries
    - Bus Layer: Command/Event buses route through aggregates
    - Logger Integration: Automatic audit logging
    - Protocol Compliance: Structural typing satisfaction
    - Validation Layer: Business rule enforcement

    Thread Safety:
    ==============
    - Pydantic models are immutable when frozen=True
    - Value objects are always immutable
    - Entities are mutable but thread-safe for creation
    - Aggregate roots manage transactional boundaries
    - Event sourcing provides replay safety

    Performance Characteristics:
    ===========================
    - O(1) entity/value creation via Pydantic
    - O(1) identity comparison for entities
    - O(1) timestamp tracking (automatic)
    - O(n) event replay for aggregate reconstruction
    - O(1) command/query dispatch via handler registry
    """

    # =========================================================================
    # NESTED CLASSES - Entity & DDD Patterns
    # =========================================================================
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

    # Generic classes - use simple assignment (not wrapper) to avoid type parameter conflicts
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

    # =========================================================================
    # OFFICIAL NAMESPACE - CQRS Patterns
    # =========================================================================
    class Cqrs:
        """CQRS namespace with nested classes."""

        class Command(FlextModelsCqrs.Command):
            """Base class for CQRS commands."""

        class Pagination(FlextModelsCqrs.Pagination):
            """Pagination model for query results."""

        class Query(FlextModelsCqrs.Query):
            """Query model for CQRS query operations."""

        class Bus(FlextModelsCqrs.Bus):
            """Bus configuration model."""

        class Handler(FlextModelsCqrs.Handler):
            """Handler configuration model."""

        # Internal utility functions exposed for testing
        @staticmethod
        def _get_command_timeout_default() -> int:
            """Get command timeout from Constants (default).

            Internal utility function exposed for testing purposes.
            """
            return FlextConstants.Cqrs.DEFAULT_COMMAND_TIMEOUT

        @staticmethod
        def _get_max_command_retries_default() -> int:
            """Get max retry attempts from Constants (default).

            Internal utility function exposed for testing purposes.
            """
            return FlextConstants.Cqrs.DEFAULT_MAX_COMMAND_RETRIES

    # =========================================================================
    # NESTED CLASSES - Base Utility Models
    # =========================================================================
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

    # =========================================================================
    # NESTED CLASSES - Configuration Models
    # =========================================================================
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

    # =========================================================================
    # NESTED CLASSES - Context Management Models
    # =========================================================================
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

    # =========================================================================
    # NESTED CLASSES - Handler Management Models
    # =========================================================================
    class HandlerRegistration(FlextModelsHandler.Registration):
        """Handler registration model."""

    class RegistrationDetails(FlextModelsHandler.RegistrationDetails):
        """Registration details model."""

    class HandlerExecutionContext(FlextModelsHandler.ExecutionContext):
        """Handler execution context model."""

    # =========================================================================
    # NESTED CLASSES - Domain Service Models
    # =========================================================================
    class DomainServiceExecutionRequest(
        FlextModelsService.DomainServiceExecutionRequest,
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

    # =========================================================================
    # CONTAINER MODELS - Dependency Injection registry models
    # =========================================================================

    class ServiceRegistration(FlextModelsContainer.ServiceRegistration):
        """Service registration model - DI container service entry."""

    class FactoryRegistration(FlextModelsContainer.FactoryRegistration):
        """Factory registration model - DI container factory entry."""

    class ContainerConfig(FlextModelsContainer.ContainerConfig):
        """Container configuration model - DI container settings."""

    # =========================================================================
    # PYDANTIC V2 DISCRIMINATED UNION - Type-safe message routing
    # =========================================================================
    # Discriminated union for CQRS message types eliminating object types
    # Uses Pydantic v2's most innovative feature: discriminated unions with
    # Discriminator field for automatic routing based on message_type
    # Note: Uses raw assignment to support runtime introspection across ecosystem
    _MessageUnion = Annotated[
        Cqrs.Command | Cqrs.Query | DomainEvent,
        Discriminator("message_type"),
    ]
    MessageUnion = _MessageUnion
    """Pydantic v2 discriminated union for type-safe CQRS message routing.

    This union type enables automatic message type detection and routing
    based on the 'message_type' field discriminator, replacing all object
    types in message handling across the entire FLEXT ecosystem.

    Usage:
        def process_message(message: MessageUnion) -> FlextResult[object]:
            match message.message_type:
                case "command":
                    return handle_command(message)
                case "query":
                    return handle_query(message)
                case "event":
                    return handle_event(message)

    Pydantic v2 automatically validates and routes messages to the correct
    type based on the discriminator field value.
    """

    # =========================================================================
    # OFFICIAL NAMESPACE - Validation Patterns
    # =========================================================================
    class Validation:
        """Validation namespace providing official access paths."""

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


# Resolve forward references for models with complex types (required by Pydantic v2)
# This is done at module level after FlextModels class definition to ensure proper initialization
# ruff: noqa: E402 - imports must be after class definition for model_rebuild() to work
from collections.abc import Callable

# Type alias for operation callables - used in OperationExecutionRequest
OperationCallable: TypeAlias = Callable[..., object]

# Build types namespace dict for model_rebuild() - Pydantic v2 requires types to be in namespace
_types_namespace = {
    "Callable": Callable,
    "OperationCallable": OperationCallable,
}
FlextModels.ConditionalExecutionRequest.model_rebuild(_types_namespace=_types_namespace)


__all__ = [
    "FlextModels",
]
