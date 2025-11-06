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

from typing import (
    Annotated,
    TypeAlias,
)

from pydantic import (
    Discriminator,
)

from flext_core._models.base import FlextModelsBase
from flext_core._models.config import FlextModelsConfig
from flext_core._models.context import FlextModelsContext
from flext_core._models.cqrs import FlextModelsCqrs
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.handler import FlextModelsHandler
from flext_core._models.service import FlextModelsService
from flext_core._models.validation import FlextModelsValidation


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
        ...     def activate(self) -> FlextResult[None]:
        ...         if self.is_active:
        ...             return FlextResult.fail("Already active")
        ...         self.is_active = True
        ...         return FlextResult.ok(None)
        >>>
        >>> # Aggregate Root - consistency boundary
        >>> class Account(FlextModels.AggregateRoot):
        ...     owner: User
        ...     balance: float = 0.0
        ...
        ...     def deposit(self, amount: float) -> FlextResult[None]:
        ...         if amount <= 0:
        ...             return FlextResult.fail("Amount must be positive")
        ...         self.balance += amount
        ...         return FlextResult.ok(None)
        >>>
        >>> # Command - CQRS pattern
        >>> class CreateUserCommand(FlextModels.Command):
        ...     name: str
        ...     email: str
        >>>
        >>> # Query - CQRS pattern
        >>> class GetUserQuery(FlextModels.Query):
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
    # BACKWARD COMPATIBILITY ALIASES - Entity & DDD Patterns
    # =========================================================================
    AggregateRoot: TypeAlias = FlextModelsEntity.AggregateRoot
    ArbitraryTypesModel: TypeAlias = FlextModelsEntity.ArbitraryTypesModel
    DomainEvent: TypeAlias = FlextModelsEntity.DomainEvent
    Entity: TypeAlias = FlextModelsEntity.Core
    FrozenStrictModel: TypeAlias = FlextModelsEntity.FrozenStrictModel
    IdentifiableMixin: TypeAlias = FlextModelsEntity.IdentifiableMixin
    TimestampableMixin: TypeAlias = FlextModelsEntity.TimestampableMixin
    TimestampedModel: TypeAlias = FlextModelsEntity.TimestampedModel
    Value: TypeAlias = FlextModelsEntity.Value
    VersionableMixin: TypeAlias = FlextModelsEntity.VersionableMixin

    # =========================================================================
    # BACKWARD COMPATIBILITY ALIASES - CQRS Patterns
    # =========================================================================
    Command: TypeAlias = FlextModelsCqrs.Command
    Cqrs: TypeAlias = FlextModelsCqrs
    Pagination: TypeAlias = FlextModelsCqrs.Pagination
    Query: TypeAlias = FlextModelsCqrs.Query

    # =========================================================================
    # BACKWARD COMPATIBILITY ALIASES - Validation Patterns
    # =========================================================================
    Validation = FlextModelsValidation

    # =========================================================================
    # BACKWARD COMPATIBILITY ALIASES - Base Utility Models
    # =========================================================================
    Metadata = FlextModelsBase.Metadata
    Payload = FlextModelsBase.Payload
    Url = FlextModelsBase.Url
    LogOperation = FlextModelsBase.LogOperation
    TimestampConfig = FlextModelsBase.TimestampConfig
    SerializationRequest = FlextModelsBase.SerializationRequest
    ConditionalExecutionRequest = FlextModelsBase.ConditionalExecutionRequest
    StateInitializationRequest = FlextModelsBase.StateInitializationRequest

    # =========================================================================
    # BACKWARD COMPATIBILITY ALIASES - Configuration Models
    # =========================================================================
    ProcessingRequest = FlextModelsConfig.ProcessingRequest
    RetryConfiguration = FlextModelsConfig.RetryConfiguration
    ValidationConfiguration = FlextModelsConfig.ValidationConfiguration
    BatchProcessingConfig = FlextModelsConfig.BatchProcessingConfig
    HandlerExecutionConfig = FlextModelsConfig.HandlerExecutionConfig
    MiddlewareConfig = FlextModelsConfig.MiddlewareConfig
    RateLimiterState = FlextModelsConfig.RateLimiterState

    # =========================================================================
    # BACKWARD COMPATIBILITY ALIASES - Context Management Models
    # =========================================================================
    StructlogProxyToken = FlextModelsContext.StructlogProxyToken
    StructlogProxyContextVar = FlextModelsContext.StructlogProxyContextVar
    Token = FlextModelsContext.Token
    ContextData = FlextModelsContext.ContextData
    ContextExport = FlextModelsContext.ContextExport
    ContextScopeData = FlextModelsContext.ContextScopeData
    ContextStatistics = FlextModelsContext.ContextStatistics
    ContextMetadata = FlextModelsContext.ContextMetadata
    ContextDomainData = FlextModelsContext.ContextDomainData

    # =========================================================================
    # BACKWARD COMPATIBILITY ALIASES - Handler Management Models
    # =========================================================================
    HandlerRegistration = FlextModelsHandler.Registration
    RegistrationDetails = FlextModelsHandler.RegistrationDetails
    HandlerExecutionContext = FlextModelsHandler.ExecutionContext

    # =========================================================================
    # BACKWARD COMPATIBILITY ALIASES - Domain Service Models
    # =========================================================================
    DomainServiceExecutionRequest = FlextModelsService.DomainServiceExecutionRequest
    DomainServiceBatchRequest = FlextModelsService.DomainServiceBatchRequest
    DomainServiceMetricsRequest = FlextModelsService.DomainServiceMetricsRequest
    DomainServiceResourceRequest = FlextModelsService.DomainServiceResourceRequest
    OperationExecutionRequest = FlextModelsService.OperationExecutionRequest

    # =========================================================================
    # PYDANTIC V2 DISCRIMINATED UNION - Type-safe message routing
    # =========================================================================
    # Discriminated union for CQRS message types eliminating object types
    # Uses Pydantic v2's most innovative feature: discriminated unions with
    # Discriminator field for automatic routing based on message_type
    type MessageUnion = Annotated[
        FlextModels.Command | FlextModels.Query | FlextModels.DomainEvent,
        Discriminator("message_type"),
    ]
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


# Rebuild models after all classes are defined to resolve forward references
FlextModels.DomainEvent.model_rebuild()
FlextModels.Entity.model_rebuild()
FlextModels.Query.model_rebuild()
FlextModels.Pagination.model_rebuild()

__all__ = [
    "FlextModels",
]
