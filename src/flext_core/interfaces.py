"""FLEXT Core Interfaces - Configuration Layer Contract Definitions.

Clean Architecture interface definitions enabling dependency inversion, extensibility,
and consistent contracts across the 32-project FLEXT ecosystem. Foundation for
plugin architecture, service boundaries, and domain-driven design patterns.

Module Role in Architecture:
    Configuration Layer → Interface Contracts → Clean Architecture Boundaries

    This module provides interface abstractions used throughout FLEXT projects:
    - Protocol-based interfaces for structural typing and implementation flexibility
    - Abstract base classes for enforcing implementation contracts
    - Service lifecycle interfaces for start/stop/health operations
    - Plugin interfaces for runtime extensibility without core modification

Interface Architecture Patterns:
    Dependency Inversion: Abstractions independent of concrete implementations
    Protocol-Based Typing: Structural typing for maximum implementation flexibility
    Plugin Architecture: Runtime extensibility through well-defined contracts
    Service Boundaries: Clear interface definition for Clean Architecture layers

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Validation, service, handler, repository interfaces
    ✅ Implemented: Event sourcing interfaces (FlextDomainEvent, FlextEventStore, FlextEventStreamReader, FlextProjectionBuilder)
    🚧 Active Development: Plugin architecture foundation (Priority 3 - October 2025)

Interface Categories:
    Validation Interfaces: FlextValidator protocol and FlextValidationRule ABC
    Service Interfaces: FlextService lifecycle and FlextConfigurable protocol
    Handler Interfaces: FlextHandler and FlextMiddleware for CQRS patterns
    Repository Interfaces: FlextRepository and FlextUnitOfWork for data access
    Plugin Interfaces: FlextPlugin and FlextPluginContext for extensibility
    Event Interfaces: FlextEventPublisher and FlextEventSubscriber patterns

Ecosystem Usage Patterns:
    # FLEXT Service Implementation
    class ApiService(FlextService):
        def start(self) -> FlextResult[None]: ...
        def health_check(self) -> FlextResult[TAnyDict]: ...

    # Singer Tap/Target Validation
    class OracleValidator(FlextValidator):
        def validate(self, value: object) -> FlextResult[object]: ...

    # Plugin Development
    class CustomPlugin(FlextPlugin):
        def initialize(self, context: FlextPluginContext) -> FlextResult[None]: ...

    # Repository Pattern (DDD)
    class UserRepository(FlextRepository):
        def save(self, entity: FlextEntity) -> FlextResult[FlextEntity]:
            return FlextResult.ok(entity)

Repository Interfaces: Comprehensive data access contracts for domain-driven design
    FlextRepository: Generic repository interface for entity persistence
    FlextUnitOfWork: Transaction boundary for aggregate operations
    FlextQueryRepository: Read-optimized repository for query operations
        def find_by_id(self, entity_id: str) -> FlextResult[object]: ...

Clean Architecture Benefits:
    - Dependency inversion preventing tight coupling to implementations
    - Domain layer independence from infrastructure concerns
    - Plugin architecture enabling runtime extensibility
    - Testability through interface mocking and substitution

Quality Standards:
    - All interfaces must use FlextResult for consistent error handling
    - Protocols must be runtime-checkable when used for dynamic validation
    - Abstract base classes must provide comprehensive implementation guidance
    - Interface evolution must maintain backward compatibility

See Also:
    docs/TODO.md: Priority 3 - Plugin architecture foundation
    abc: Abstract base class patterns for interface definition
    typing: Protocol definitions and runtime type checking

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from structlog.stdlib import BoundLogger

    from flext_core.flext_types import TAnyDict
    from flext_core.result import FlextResult

# =============================================================================
# VALIDATION INTERFACES
# =============================================================================


@runtime_checkable
class FlextValidator(Protocol):
    """Protocol for custom validators enabling flexible validation implementation.

    Runtime-checkable protocol defining the contract for custom validation logic
    with FlextResult integration for consistent error handling. Supports structural
    typing allowing any class with matching validate method to be used as validator.

    Protocol Features:
        - Structural typing for maximum implementation flexibility
        - Runtime type checking through @runtime_checkable decorator
        - FlextResult integration for consistent error handling
        - Generic object validation supporting any data type
        - Integration with validation pipelines and chains

    Implementation Guidelines:
        - Return validated/transformed value in success case
        - Use descriptive error messages in failure case
        - Support immutable validation without side effects
        - Handle edge cases gracefully with appropriate error messages
        - Maintain performance for high-frequency validation scenarios

    Usage Patterns:
        # Protocol-compliant validator implementation
        class EmailValidator:
            def validate(self, value: object) -> FlextResult[object]:
                if not _BaseValidators.is_string(value):
                    return FlextResult.fail("Email must be a string")

                if not _BaseValidators.is_email(value):
                    return FlextResult.fail("Invalid email format")

                return FlextResult.ok(value.lower().strip())

        # Runtime type checking
        def use_validator(validator: FlextValidator, data: object):
            if isinstance(validator, FlextValidator):
                return validator.validate(data)
            raise FlextTypeError(
                "Expected FlextValidator protocol",
                expected_type="FlextValidator",
                actual_type=type(validator)
            )

        # Structural typing usage
        email_validator = EmailValidator()
        result = email_validator.validate("user@example.com")

    """

    def validate(self, value: object) -> FlextResult[object]:
        """Validate value and return result with transformation support.

        Core validation method that processes input value and returns either
        validated/transformed value on success or descriptive error on failure.

        Args:
            value: Value to validate (any type supported)

        Returns:
            FlextResult containing validated/transformed value on success
            or error message with context on validation failure

        Usage:
            validator = MyValidator()
            result = validator.validate("input_data")
            if result.success:
                validated_value = result.data
            else:
                error_message = result.error

        """
        ...


class FlextValidationRule(ABC):
    """Abstract base class for validation rules with boolean evaluation.

    Abstract base class defining the contract for validation rules that perform
    boolean evaluation with separated error message generation. Provides a
    template for implementing reusable validation logic with consistent error reporting.

    Rule Design Patterns:
        - Boolean evaluation through check method for simple pass/fail logic
        - Separated error message generation for customizable error reporting
        - Reusable rule composition for complex validation scenarios
        - Template method pattern for consistent validation rule implementation
        - Integration with validation pipelines and rule engines

    Implementation Guidelines:
        - Keep check method pure and side-effect free
        - Provide descriptive error messages that help users understand failures
        - Handle edge cases gracefully without raising exceptions
        - Optimize for performance in high-frequency validation scenarios
        - Support composability for building complex validation logic

    Usage Patterns:
        # Custom validation rule implementation
        class PositiveNumberRule(FlextValidationRule):
            def check(self, value: object) -> bool:
                return isinstance(value, (int, float)) and value > 0

            def error_message(self) -> str:
                return "Value must be a positive number"

        # Email format validation rule
        class EmailFormatRule(FlextValidationRule):
            def check(self, value: object) -> bool:
                return _BaseValidators.is_email(value)

            def error_message(self) -> str:
                return "Invalid email format"

        # Rule composition for complex validation
        class UserAgeRule(FlextValidationRule):
            def __init__(self, min_age: int = 18, max_age: int = 120):
                self.min_age = min_age
                self.max_age = max_age

            def check(self, value: object) -> bool:
                if not isinstance(value, int):
                    return False
                return self.min_age <= value <= self.max_age

            def error_message(self) -> str:
                return f"Age must be between {self.min_age} and {self.max_age}"

        # Using rules in validation logic
        def validate_with_rule(
            rule: FlextValidationRule,
            value: object
        ) -> FlextResult[object]:
            if rule.check(value):
                return FlextResult.ok(value)
            return FlextResult.fail(rule.error_message())

    """

    @abstractmethod
    def check(self, value: object) -> bool:
        """Check if value passes validation rule.

        Core validation logic that evaluates whether the provided value
        satisfies the rule's criteria. Should be implemented as a pure
        function without side effects.

        Args:
            value: Value to validate against this rule

        Returns:
            True if value passes validation, False otherwise

        Implementation Notes:
            - Should handle any input type gracefully
            - Must not raise exceptions for invalid input types
            - Should be optimized for performance if used frequently
            - Keep logic simple and focused on single validation concern

        """
        ...

    @abstractmethod
    def error_message(self) -> str:
        """Get human-readable error message for validation failure.

        Provides descriptive error message that explains why validation
        failed and what the expected criteria are. Used for user feedback
        and debugging purposes.

        Returns:
            Clear, actionable error message explaining validation failure

        Implementation Notes:
            - Should be descriptive and help users understand the requirement
            - Include specific criteria when helpful (e.g., value ranges)
            - Use consistent language and formatting across rules
            - Avoid technical jargon in user-facing messages

        """
        ...


# =============================================================================
# SERVICE INTERFACES
# =============================================================================


class FlextService(ABC):
    """Base interface for all FLEXT services."""

    @abstractmethod
    def start(self) -> FlextResult[None]:
        """Start the service.

        Returns:
            Result of startup

        """
        ...

    @abstractmethod
    def stop(self) -> FlextResult[None]:
        """Stop the service.

        Returns:
            Result of shutdown

        """
        ...

    @abstractmethod
    def health_check(self) -> FlextResult[TAnyDict]:
        """Check service health.

        Returns:
            Result with health status

        """
        ...


@runtime_checkable
class FlextConfigurable(Protocol):
    """Protocol for configurable components."""

    def configure(self, settings: Mapping[str, object]) -> FlextResult[None]:
        """Configure component with settings.

        Args:
            settings: Configuration settings

        Returns:
            Result of configuration

        """
        ...


# =============================================================================
# HANDLER INTERFACES
# =============================================================================


class FlextHandler(ABC):
    """Base interface for command/event handlers."""

    @abstractmethod
    def can_handle(self, message: object) -> bool:
        """Check if handler can process message.

        Args:
            message: Message to check

        Returns:
            True if can handle

        """
        ...

    @abstractmethod
    def handle(self, message: object) -> FlextResult[object]:
        """Handle the message.

        Args:
            message: Message to handle

        Returns:
            Result of handling

        """
        ...


class FlextMiddleware(ABC):
    """Middleware interface for processing pipeline."""

    @abstractmethod
    def process(
        self,
        message: object,
        next_handler: FlextHandler,
    ) -> FlextResult[object]:
        """Process message in pipeline.

        Args:
            message: Message to process
            next_handler: Next handler in chain

        Returns:
            Result from pipeline

        """
        ...


# =============================================================================
# REPOSITORY INTERFACES
# =============================================================================


class FlextRepository(ABC):
    """Base repository interface for data access."""

    @abstractmethod
    def find_by_id(self, entity_id: str) -> FlextResult[object]:
        """Find entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Result with entity or not found error

        """
        ...

    @abstractmethod
    def save(self, entity: object) -> FlextResult[None]:
        """Save entity.

        Args:
            entity: Entity to save

        Returns:
            Result of save operation

        """
        ...

    @abstractmethod
    def delete(self, entity_id: str) -> FlextResult[None]:
        """Delete entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Result of delete operation

        """
        ...


class FlextUnitOfWork(ABC):
    """Unit of Work pattern interface."""

    @abstractmethod
    def commit(self) -> FlextResult[None]:
        """Commit all changes.

        Returns:
            Result of commit

        """
        ...

    @abstractmethod
    def rollback(self) -> FlextResult[None]:
        """Rollback all changes.

        Returns:
            Result of rollback

        """
        ...

    @abstractmethod
    def __enter__(self) -> FlextUnitOfWork:
        """Enter context."""
        ...

    @abstractmethod
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context with automatic rollback on error.

        Args:
            exc_type: Type of exception
            exc_val: Value of exception
            exc_tb: Traceback of exception

        """
        ...


# =============================================================================
# PLUGIN INTERFACES
# =============================================================================


class FlextPlugin(ABC):
    """Base interface for plugins.

    Args:
        **kwargs: Additional keyword arguments

    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name.

        Returns:
            Unique plugin name

        """
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version.

        Returns:
            Semantic version string

        """
        ...

    @abstractmethod
    def initialize(self, context: FlextPluginContext) -> FlextResult[None]:
        """Initialize plugin with context.

        Args:
            context: Plugin context

        Returns:
            Result of initialization

        """
        ...

    @abstractmethod
    def shutdown(self) -> FlextResult[None]:
        """Shutdown plugin cleanly.

        Returns:
            Result of shutdown

        """
        ...


@runtime_checkable
class FlextPluginContext(Protocol):
    """Protocol for plugin context."""

    @property
    def logger(self) -> BoundLogger:
        """Get logger for plugin.

        Returns:
            Logger for plugin

        """
        ...

    @property
    def config(self) -> Mapping[str, object]:
        """Get plugin configuration.

        Returns:
            Plugin configuration

        """
        ...

    def get_service(self, service_name: str) -> FlextResult[object]:
        """Get service by name.

        Args:
            service_name: Name of service

        Returns:
            FlextResult with service or error

        """
        ...


# =============================================================================
# EVENT INTERFACES
# =============================================================================


class FlextEventPublisher(ABC):
    """Interface for publishing events.

    Args:
        **kwargs: Additional keyword arguments

    """

    @abstractmethod
    def publish(self, event: object) -> FlextResult[None]:
        """Publish event.

        Args:
            event: Event to publish

        Returns:
            Result of publish

        """
        ...


class FlextEventSubscriber(ABC):
    """Interface for subscribing to events.

    Args:
        **kwargs: Additional keyword arguments

    """

    @abstractmethod
    def subscribe(
        self,
        event_type: type[object],
        handler: FlextHandler,
    ) -> FlextResult[None]:
        """Subscribe to event type.

        Args:
            event_type: Type of events to receive
            handler: Handler for events

        Returns:
            Result of subscription

        """
        ...

    @abstractmethod
    def unsubscribe(
        self,
        event_type: type[object],
        handler: FlextHandler,
    ) -> FlextResult[None]:
        """Unsubscribe from event type.

        Args:
            event_type: Type of events
            handler: Handler to remove

        Returns:
            Result of unsubscription

        """
        ...


# =============================================================================
# EVENT SOURCING INTERFACES
# =============================================================================


@runtime_checkable
class FlextDomainEvent(Protocol):
    """Protocol for domain events in event sourcing architecture.

    Runtime-checkable protocol defining the contract for domain events
    that are captured and stored in an event stream. Supports structural
    typing allowing any class with matching properties to be used as domain event.

    Event Sourcing Features:
        - Immutable event data for reliable event replay
        - Aggregate identification for event stream partitioning
        - Temporal ordering through sequence numbers
        - Event metadata for debugging and monitoring
        - Version compatibility for event schema evolution

    Usage Patterns:
        # Domain event implementation
        @dataclass(frozen=True)
        class UserRegistered:
            aggregate_id: str
            event_id: str
            occurred_at: datetime
            version: int
            user_email: str
            user_name: str

        # Protocol compliance check
        def handle_event(event: FlextDomainEvent):
            if isinstance(event, FlextDomainEvent):
                store_event(event.aggregate_id, event)
    """

    @property
    def aggregate_id(self) -> str:
        """Get aggregate identifier that produced this event.

        Returns:
            Unique identifier of the aggregate root that generated this event

        """
        ...

    @property
    def event_id(self) -> str:
        """Get unique event identifier.

        Returns:
            Universally unique identifier for this specific event

        """
        ...

    @property
    def occurred_at(self) -> object:
        """Get timestamp when event occurred.

        Returns:
            Timestamp when the event was generated (datetime or timestamp)

        """
        ...

    @property
    def version(self) -> int:
        """Get event schema version.

        Returns:
            Version number for event schema compatibility

        """
        ...


class FlextEventStore(ABC):
    """Abstract interface for event store implementations.

    Defines the contract for event storage systems supporting event sourcing
    patterns with aggregate-based event streams, versioning, and replay capabilities.

    Event Store Features:
        - Aggregate-based event streams with optimistic concurrency
        - Event versioning and schema evolution support
        - Snapshot support for performance optimization
        - Event replay capabilities for aggregate reconstruction
        - Transaction support for consistency guarantees

    Implementation Guidelines:
        - Ensure append-only semantics for event immutability
        - Support optimistic concurrency control through versioning
        - Implement efficient event stream reading for replay
        - Provide snapshot capabilities for large event streams
        - Handle concurrent writes gracefully with proper error reporting

    Usage Patterns:
        # Event storage
        result = event_store.append_events(
            aggregate_id="user-123",
            events=[UserRegistered(...), UserActivated(...)],
            expected_version=0
        )

        # Event replay
        events_result = event_store.get_events("user-123")
        if events_result.success:
            for event in events_result.data:
                aggregate.apply_event(event)
    """

    @abstractmethod
    def append_events(
        self,
        aggregate_id: str,
        events: list[FlextDomainEvent],
        expected_version: int | None = None,
    ) -> FlextResult[None]:
        """Append events to aggregate stream.

        Appends new events to the event stream for the specified aggregate
        with optimistic concurrency control through version checking.

        Args:
            aggregate_id: Unique identifier of the aggregate
            events: List of domain events to append
            expected_version: Expected current version for concurrency control

        Returns:
            Result indicating success or concurrency/validation errors

        Raises:
            FlextConcurrencyError: If expected_version doesn't match current version
            FlextValidationError: If events are invalid or malformed

        """
        ...

    @abstractmethod
    def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
        to_version: int | None = None,
    ) -> FlextResult[list[FlextDomainEvent]]:
        """Get events for aggregate from event stream.

        Retrieves events for the specified aggregate within the given
        version range, supporting both full replay and partial updates.

        Args:
            aggregate_id: Unique identifier of the aggregate
            from_version: Starting version (inclusive)
            to_version: Ending version (inclusive), None for latest

        Returns:
            Result containing list of events or empty list if none found

        Implementation Notes:
            - Events must be returned in version order
            - Should handle non-existent aggregates gracefully
            - Version ranges should be inclusive on both ends

        """
        ...

    @abstractmethod
    def get_current_version(self, aggregate_id: str) -> FlextResult[int]:
        """Get current version of aggregate event stream.

        Returns the current version number of the event stream for
        the specified aggregate, used for optimistic concurrency control.

        Args:
            aggregate_id: Unique identifier of the aggregate

        Returns:
            Result containing current version number (0 if no events exist)

        """
        ...

    @abstractmethod
    def save_snapshot(
        self,
        aggregate_id: str,
        snapshot: object,
        version: int,
    ) -> FlextResult[None]:
        """Save aggregate snapshot for performance optimization.

        Stores a snapshot of the aggregate state at a specific version
        to optimize future event replay operations.

        Args:
            aggregate_id: Unique identifier of the aggregate
            snapshot: Serializable snapshot of aggregate state
            version: Version at which snapshot was taken

        Returns:
            Result indicating success or storage errors

        """
        ...

    @abstractmethod
    def get_snapshot(
        self,
        aggregate_id: str,
    ) -> FlextResult[tuple[object, int] | None]:
        """Get latest snapshot for aggregate.

        Retrieves the most recent snapshot for the specified aggregate
        along with the version at which it was taken.

        Args:
            aggregate_id: Unique identifier of the aggregate

        Returns:
            Result containing tuple of (snapshot, version) or None if no snapshot exists

        """
        ...


class FlextEventStreamReader(ABC):
    """Interface for reading event streams with advanced querying capabilities.

    Provides advanced event stream reading capabilities including filtering,
    pagination, and cross-aggregate queries for complex event sourcing scenarios.

    Stream Reading Features:
        - Cross-aggregate event streaming for projections
        - Event filtering by type, timestamp, and custom criteria
        - Pagination support for large event streams
        - Real-time event streaming capabilities
        - Event metadata querying and analysis

    Usage Patterns:
        # Stream all events from timestamp
        result = reader.stream_events(
            from_timestamp=datetime.now() - timedelta(hours=1),
            event_types=["UserRegistered", "OrderCreated"]
        )

        # Paginated event reading
        result = reader.get_events_page(
            page_size=100,
            cursor="event-cursor-123"
        )
    """

    @abstractmethod
    def stream_events(
        self,
        from_timestamp: object | None = None,
        to_timestamp: object | None = None,
        event_types: list[str] | None = None,
        aggregate_ids: list[str] | None = None,
    ) -> FlextResult[list[FlextDomainEvent]]:
        """Stream events with filtering criteria.

        Streams events across aggregates with optional filtering by
        timestamp range, event types, and specific aggregates.

        Args:
            from_timestamp: Start timestamp (inclusive)
            to_timestamp: End timestamp (inclusive)
            event_types: List of event type names to include
            aggregate_ids: List of specific aggregate IDs to include

        Returns:
            Result containing filtered list of events in temporal order

        """
        ...

    @abstractmethod
    def get_events_page(
        self,
        page_size: int = 100,
        cursor: str | None = None,
    ) -> FlextResult[tuple[list[FlextDomainEvent], str | None]]:
        """Get paginated events with cursor-based navigation.

        Retrieves a page of events with cursor-based pagination for
        efficient processing of large event streams.

        Args:
            page_size: Maximum number of events to return
            cursor: Pagination cursor for next page (None for first page)

        Returns:
            Result containing tuple of (events, next_cursor)
            next_cursor is None if no more events available

        """
        ...


class FlextProjectionBuilder(ABC):
    """Interface for building read model projections from event streams.

    Defines the contract for projection builders that create and maintain
    read models from domain events, supporting both batch and real-time
    projection updates.

    Projection Features:
        - Event-driven read model updates
        - Projection versioning and rebuilding
        - Error handling and recovery mechanisms
        - Performance optimization through batching
        - Projection state management and checkpointing

    Usage Patterns:
        # Build projection from events
        result = builder.project_events(
            events=[UserRegistered(...), UserActivated(...)],
            projection_name="user_summary"
        )

        # Rebuild projection from scratch
        result = builder.rebuild_projection("user_summary")
    """

    @abstractmethod
    def project_events(
        self,
        events: list[FlextDomainEvent],
        projection_name: str,
    ) -> FlextResult[None]:
        """Project events to update read model.

        Processes domain events to update the specified projection
        (read model) with new data from the events.

        Args:
            events: List of domain events to project
            projection_name: Name of the projection to update

        Returns:
            Result indicating success or projection errors

        """
        ...

    @abstractmethod
    def rebuild_projection(
        self,
        projection_name: str,
        from_timestamp: object | None = None,
    ) -> FlextResult[None]:
        """Rebuild projection from event stream.

        Completely rebuilds the specified projection by replaying
        all relevant events from the event store.

        Args:
            projection_name: Name of the projection to rebuild
            from_timestamp: Starting timestamp for rebuild (None for all events)

        Returns:
            Result indicating success or rebuild errors

        """
        ...

    @abstractmethod
    def get_projection_status(
        self,
        projection_name: str,
    ) -> FlextResult[Mapping[str, object]]:
        """Get status information for projection.

        Returns status information about the projection including
        last processed event, error count, and performance metrics.

        Args:
            projection_name: Name of the projection

        Returns:
            Result containing projection status dictionary

        """
        ...
