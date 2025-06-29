"""Event Bus implementation following DDD principles with lato library.

Provides a central hub for publishing and subscribing to domain events across
the FLX platform, supporting both in-memory and distributed event processing.
"""

# Strategic imports for optional dependencies

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, Protocol
from uuid import UUID, uuid4

import orjson
import structlog

# Strategic imports with type checking for optional enterprise dependencies
if TYPE_CHECKING:
    from aio_pika.abc import (
        AbstractChannel,
        AbstractExchange,
        AbstractIncomingMessage,
        AbstractQueue,
        AbstractRobustConnection,
    )

from lato import Event as LatoEvent
from pydantic import Field

# Type alias for backward compatibility and cleaner typing
Event = LatoEvent

# ZERO TOLERANCE: Dynamic import to avoid circular dependencies - domain_config is central

# Python 3.13 type aliases for event handling - with strict validation
type EventHandler = Callable[[object], Awaitable[None]]  # Generic event handler
type DomainEventHandler = Callable[[object], Awaitable[None]]  # Domain event handler

if TYPE_CHECKING:
    from asyncio import Future

    from flx_core.domain.advanced_types import DomainEventData, MetadataDict


logger = structlog.get_logger()


class EventBusProtocol(Protocol):
    """Protocol defining the unified interface for all EventBus implementations.

    This protocol provides type safety for event bus dependency injection,
    allowing any EventBus implementation (EventBus, HybridEventBus, DomainEventBus,
    InMemoryEventBus) to be used interchangeably.

    ZERO TOLERANCE: Eliminates type: ignore annotations through proper abstraction.
    """

    async def publish(
        self,
        event: Event | dict[str, object] | str,
        data: dict[str, object] | None = None,
    ) -> None:
        """Publish an event to all subscribers.

        Args:
        ----
            event: Event object, event type string, or dictionary data
            data: Optional event data when event is a string type

        Note:
        ----
            Flexible signature supporting legacy Event objects and modern patterns.

        """

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to an event type.

        Args:
        ----
            event_type: Type of event to subscribe to
            handler: Function to handle events of this type

        """


class DomainEvent(LatoEvent):
    """Base domain event following DDD principles."""

    event_id: UUID = Field(default_factory=uuid4)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    aggregate_id: UUID | None = None
    aggregate_type: str | None = None
    event_version: int = 1
    correlation_id: UUID | None = None
    causation_id: UUID | None = None
    user_id: str | None = None

    # Event type identifier (must be set by subclasses)
    event_type: ClassVar[str] = "domain_event"

    @property
    def type(self) -> str:
        """Get event type for compatibility with event bus."""
        return self.event_type

    def get_event_data(self) -> DomainEventData:
        """Get the event-specific data payload. Override in subclasses."""
        return {}

    def to_dict(self) -> MetadataDict:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type,
            "occurred_at": self.occurred_at.isoformat(),
            "aggregate_id": str(self.aggregate_id) if self.aggregate_id else None,
            "aggregate_type": self.aggregate_type,
            "event_version": self.event_version,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "user_id": self.user_id,
            "data": self.get_event_data(),
        }

    @classmethod
    def create(cls, event_type: str, data: DomainEventData) -> DomainEvent:
        """Create a DomainEvent with specific type and data.

        Convenience method for creating domain events with data payload.
        """

        class CustomDomainEvent(cls):
            event_type: ClassVar[str] = event_type

            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                self._data = data

            def get_event_data(self) -> DomainEventData:
                return self._data

        return CustomDomainEvent()


# EventHandler type alias for modern event handling
DomainEventHandler = Callable[[DomainEvent], Awaitable[None]]


class DomainEventBus:
    """Domain Event Bus using lato library with DDD principles."""

    def __init__(self) -> None:
        """Initialize the domain event bus.

        Sets up the domain event bus with handler registry and structured logging
        for async event processing across domain boundaries.

        Note:
        ----
            Uses async processing with proper error handling and logging.

        """
        self.logger = logger.bind(component="domain_event_bus")
        self._handlers: dict[str, list[Callable[[DomainEvent], Awaitable[None]]]] = {}

    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event.

        Publishes a domain event to all registered handlers for the event type,
        executing handlers concurrently with error isolation.

        Args:
        ----
            event: Domain event to publish to handlers

        Note:
        ----
            Executes all handlers concurrently using asyncio.gather with
            return_exceptions=True for error isolation.

        """
        self.logger.debug(
            "Publishing domain event",
            event_type=event.event_type,
            event_id=str(event.event_id),
        )

        # Get handlers for this event type
        handlers = self._handlers.get(event.event_type, [])

        # Execute all handlers concurrently
        if handlers:
            await asyncio.gather(
                *[handler(event) for handler in handlers],
                return_exceptions=True,
            )

    def subscribe(
        self, event_type: str, handler: Callable[[DomainEvent], Awaitable[None]]
    ) -> None:
        """Subscribe to a domain event type.

        Registers a handler function to be called when events of the specified
        type are published through the event bus.

        Args:
        ----
            event_type: Type of domain event to subscribe to
            handler: Async function to handle the event

        Note:
        ----
            Handlers are executed concurrently when events are published.

        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)
        self.logger.info("Domain event handler subscribed", event_type=event_type)


class HybridEventBus:
    """Hybrid event bus supporting both legacy and DDD patterns."""

    def __init__(self) -> None:
        """Initialize the hybrid event bus.

        Sets up hybrid event bus supporting both legacy Event patterns and
        modern DomainEvent patterns with optional AMQP distribution.

        Note:
        ----
            Uses async processing with proper error handling and logging.

        """
        self.logger = logger.bind(component="event_bus")
        self._handlers: dict[str, list[EventHandler]] = {}
        self._running = False
        self._tasks: set[Future[object]] = set()
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._exchange: AbstractExchange | None = None
        self._queues: dict[str, AbstractQueue] = {}

        # DDD Event Bus using lato
        self._domain_event_bus = DomainEventBus()

    async def publish(
        self,
        event: Event | dict[str, object] | str,
        data: dict[str, object] | None = None,
    ) -> None:
        """Publish an event - supports flexible input types.

        Unified publish method supporting multiple event formats for compatibility
        with all FLX components and legacy patterns.

        Args:
        ----
            event: Event object, event type string, or dictionary data
            data: Optional event data when event is a string type

        Note:
        ----
            ZERO TOLERANCE: Proper type conversion eliminates type: ignore annotations.

        """
        # Convert input to Event object for unified processing
        if isinstance(event, str):
            # String event type with optional data
            event_obj = Event.create(event, data or {})
        elif isinstance(event, dict):
            # Dictionary with event type and data
            event_type = event.get("type") or event.get("event_type", "unknown")
            event_data = {
                k: v for k, v in event.items() if k not in {"type", "event_type"}
            }
            event_obj = Event.create(str(event_type), event_data)
        else:
            # Already an Event object
            event_obj = event

        self.logger.debug(
            "Publishing event (hybrid implementation)",
            event_type=event_obj.type,
            event_id=str(event_obj.id),
        )

        # Dispatch to local handlers
        handlers = self._handlers.get(event_obj.type, [])
        if handlers:
            await asyncio.gather(
                *[handler(event_obj) for handler in handlers],
                return_exceptions=True,
            )

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to an event type - base implementation.

        Registers a handler for the specified event type in the hybrid event bus.

        Args:
        ----
            event_type: Type of event to subscribe to
            handler: Function to handle events of this type

        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        self.logger.debug("Handler subscribed to event type", event_type=event_type)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from an event type - base implementation.

        Removes a handler from the specified event type in the hybrid event bus.

        Args:
        ----
            event_type: Type of event to unsubscribe from
            handler: Handler function to remove

        """
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            self.logger.debug(
                "Handler unsubscribed from event type",
                event_type=event_type,
            )

    async def initialize(self) -> None:
        """Initialize the hybrid event bus.

        Prepares the hybrid event bus for operation by setting up internal
        components and establishing any necessary connections. This method
        ensures the event bus is ready to handle event publishing and
        subscription operations.

        Note:
        ----
            This method is called during application startup to prepare
            the event bus for operation.

        """
        self.logger.info("Initializing hybrid event bus")

        # Initialize internal state if needed
        try:
            self._initialized
            self.logger.debug("Hybrid event bus already initialized")
        except AttributeError:
            self._initialized = True
            self.logger.info("Hybrid event bus initialized successfully")


# Simple in-memory event bus for testing


class InMemoryEventBus:
    """Simple in-memory event bus for testing purposes."""

    def __init__(self) -> None:
        """Initialize the in-memory event bus.

        Sets up simple in-memory event bus for testing with handler registry
        and logging for debugging event flow.

        """
        self.logger = logger.bind(component="in_memory_event_bus")
        self._handlers: dict[str, list[Callable[[Event], Awaitable[None]]]] = {}

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Publishes an event to all registered handlers for the event type,
        executing them concurrently with error isolation.

        Args:
        ----
            event: Event to publish to all subscribers

        """
        self.logger.debug("Publishing event", event_type=event.type)

        handlers = self._handlers.get(event.type, [])
        if handlers:
            await asyncio.gather(
                *[handler(event) for handler in handlers],
                return_exceptions=True,
            )

    def subscribe(
        self, event_type: str, handler: Callable[[Event], Awaitable[None]]
    ) -> None:
        """Subscribe to an event type.

        Registers a handler function to be called when events of the specified
        type are published through the in-memory event bus.

        Args:
        ----
            event_type: Type of event to subscribe to
            handler: Async function to handle the event

        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)


# Legacy EventBus for backward compatibility


class EventBus(HybridEventBus):
    """Legacy EventBus - use HybridEventBus or DomainEventBus for new implementations."""

    async def start(self) -> None:
        """Start the event bus.

        Starts the event bus with optional AMQP connection for distributed
        event processing across multiple services.

        """
        self.logger.info("Starting event bus")

        # Connect to RabbitMQ if configured - aio_pika is REQUIRED
        # ZERO TOLERANCE: Dynamic import to avoid circular dependencies
        get_config = __import__(
            "flx_core.config.domain_config",
            fromlist=["get_config"],
        ).get_config
        config = get_config()
        try:
            amqp_url = config.messaging.amqp_url
            if amqp_url:
                await self._connect_amqp()
        except AttributeError:
            pass  # No AMQP configuration available

        self._running = True
        self.logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop the event bus.

        Gracefully shuts down the event bus, canceling pending tasks and
        closing AMQP connections if they exist.

        Note:
        ----
            Uses async processing with proper error handling and logging.

        """
        self.logger.info("Stopping event bus")
        self._running = False

        # Cancel all pending tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        # Close AMQP connection
        if self._connection and not self._connection.is_closed:
            await self._connection.close()

        self.logger.info("Event bus stopped")

    async def _connect_amqp(self) -> None:
        """Connect to RabbitMQ.

        Establishes connection to RabbitMQ for distributed event processing
        with retry logic and graceful degradation if AMQP is unavailable.

        Note:
        ----
            Uses async processing with proper error handling and logging.

        """
        # ZERO TOLERANCE: Dynamic import to avoid circular dependencies
        get_config = __import__(
            "flx_core.config.domain_config",
            fromlist=["get_config"],
        ).get_config
        config = get_config()
        amqp_url = getattr(config.messaging, "amqp_url", None)
        if not amqp_url:
            self.logger.warning("AMQP disabled: no URL provided")
            return
        try:
            # Dynamic import for optional aio_pika dependency
            from aio_pika import connect_robust

            self.logger.info("Connecting to RabbitMQ", url=amqp_url)

            # aio_pika is guaranteed to be available
            connection = await connect_robust(
                amqp_url,
                heartbeat=getattr(config.messaging, "amqp_heartbeat", 60),
                connection_attempts=getattr(
                    config.messaging,
                    "amqp_connection_attempts",
                    3,
                ),
                retry_delay=getattr(config.messaging, "amqp_retry_delay", 2),
            )
            self._connection = connection

            channel = await connection.channel()
            self._channel = channel
            await channel.set_qos(prefetch_count=10)

            # Declare exchange - aio_pika is guaranteed to be available
            from aio_pika import ExchangeType

            exchange = await channel.declare_exchange(
                "flx_core.events",
                ExchangeType.TOPIC,
                durable=True,
            )
            self._exchange = exchange

            self.logger.info("Connected to RabbitMQ successfully")

        except ConnectionError as e:
            self.logger.exception("Failed to connect to RabbitMQ", error=str(e))
            # Continue without AMQP (in-memory only)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to an event type.

        Registers a handler for the specified event type and creates AMQP
        consumer if distributed messaging is enabled.

        Args:
        ----
            event_type: Type of event to subscribe to
            handler: Function to handle events of this type

        Note:
        ----
            Uses async processing with proper error handling and logging.

        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)

        # If AMQP is connected, create queue for this event type
        if self._channel and event_type not in self._queues:
            task = asyncio.create_task(self._create_amqp_consumer(event_type))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self.logger.info("Handler subscribed", event_type=event_type)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from an event type.

        Removes a handler from the specified event type and cleans up AMQP
        resources if no more handlers remain for the event type.

        Args:
        ----
            event_type: Type of event to unsubscribe from
            handler: Handler function to remove

        """
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

            if not self._handlers[event_type]:
                del self._handlers[event_type]

                # Remove AMQP queue if no more handlers
                if event_type in self._queues:
                    queue = self._queues.pop(event_type)
                    task = asyncio.create_task(queue.delete())
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)

        self.logger.info("Handler unsubscribed", event_type=event_type)

    async def publish(
        self,
        event: Event | dict[str, object] | str,
        data: dict[str, object] | None = None,
    ) -> None:
        """Publish an event with flexible input types.

        Publishes an event to both AMQP (if connected) and local handlers,
        providing distributed and local event processing capabilities.

        Args:
        ----
            event: Event object, event type string, or dictionary data
            data: Optional event data when event is a string type

        Raises:
        ------
            RuntimeError: If event bus is not running

        Note:
        ----
            ZERO TOLERANCE: Proper type conversion eliminates type: ignore annotations.

        """
        if not self._running:
            msg = "Event bus is not running"
            raise RuntimeError(msg)

        # Convert input to Event object for unified processing
        if isinstance(event, str):
            # String event type with optional data
            event_obj = Event.create(event, data or {})
        elif isinstance(event, dict):
            # Dictionary with event type and data
            event_type = event.get("type") or event.get("event_type", "unknown")
            event_data = {
                k: v for k, v in event.items() if k not in {"type", "event_type"}
            }
            event_obj = Event.create(str(event_type), event_data)
        else:
            # Already an Event object
            event_obj = event

        self.logger.debug(
            "Publishing event (enterprise implementation)",
            event_type=event_obj.type,
            event_id=str(event_obj.id),
        )

        # Publish to AMQP if connected - aio_pika is guaranteed to be available
        if self._exchange:
            try:
                from aio_pika import Message

                message = Message(body=event_obj.to_json().encode())
                await self._exchange.publish(
                    message,
                    routing_key=event_obj.type,
                )
            except RuntimeError as e:
                self.logger.exception("Failed to publish to AMQP", error=str(e))

        # Always dispatch locally
        await self._dispatch_local(event_obj)

    async def _dispatch_local(self, event: Event) -> None:
        """Dispatch event to local handlers.

        Dispatches events to all local handlers including wildcard handlers,
        creating async tasks for each handler execution.

        Args:
        ----
            event: Event to dispatch to local handlers

        """
        handlers = self._handlers.get(event.type, [])
        handlers.extend(self._handlers.get("*", []))

        for handler in handlers:
            task = asyncio.create_task(self._handle_event(handler, event))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _handle_event(self, handler: EventHandler, event: Event) -> None:
        """Handle a single event with error handling.

        Executes a single event handler with comprehensive error handling
        to prevent handler failures from affecting other handlers.

        Args:
        ----
            handler: Handler function to execute
            event: Event to pass to the handler

        """
        try:
            await handler(event)
        except (
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
            TimeoutError,
            ConnectionError,
            ImportError,
        ) as e:
            # ZERO TOLERANCE - Specific exception types for event handler failures
            self.logger.exception(
                "Event handler failed",
                event_type=event.type,
                event_id=str(event.id),
                error=str(e),
            )

    async def _create_amqp_consumer(self, event_type: str) -> None:
        """Create AMQP queue and consumer for event type.

        Creates a durable queue for the event type and binds it to the exchange
        for distributed event processing across multiple service instances.

        Args:
        ----
            event_type: Type of event to create consumer for

        """
        if not self._channel or not self._exchange:
            return

        try:
            # Declare queue
            queue = await self._channel.declare_queue(
                f"flx_events_{event_type}",
                durable=True,
            )
            self._queues[event_type] = queue

            # Bind queue to exchange
            await queue.bind(self._exchange, routing_key=event_type)

            # Start consuming
            await queue.consume(self._on_amqp_message)

            self.logger.info("AMQP consumer created", event_type=event_type)

        except (
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
            TimeoutError,
            ConnectionError,
            ImportError,
        ) as e:
            # ZERO TOLERANCE - Specific exception types for AMQP consumer failures
            self.logger.exception(
                "Failed to create AMQP consumer",
                event_type=event_type,
                error=str(e),
            )

    async def _on_amqp_message(self, message: AbstractIncomingMessage) -> None:
        """Handle incoming AMQP message.

        Processes incoming messages from AMQP, deserializes them to events,
        and dispatches them to local handlers for distributed event processing.

        Args:
        ----
            message: Incoming AMQP message containing serialized event

        Note:
        ----
            Uses async processing with proper error handling and logging.

        """
        async with message.process():
            try:
                event = Event.from_json(message.body.decode())

                self.logger.debug(
                    "Received event from AMQP",
                    event_type=event.type,
                    event_id=str(event.id),
                )

                await self._dispatch_local(event)
            except (orjson.JSONDecodeError, KeyError) as e:
                self.logger.exception("Failed to parse AMQP message", error=str(e))


# =============================================================================
# SPECIFIC DOMAIN EVENTS - CONSOLIDATED FROM domain/events.py
# =============================================================================


class PipelineCreated(DomainEvent):
    """Event raised when a pipeline is created."""

    event_type: ClassVar[str] = "pipeline.created"

    pipeline_id: str = ""
    pipeline_name: str = ""
    created_by: str | None = None
    pipeline_data: dict[str, object] = Field(default_factory=dict)

    def get_event_data(self) -> DomainEventData:
        """Get event data payload."""
        return {
            "pipeline_id": str(self.pipeline_id),
            "pipeline_name": str(self.pipeline_name),
            "created_by": str(self.created_by) if self.created_by else None,
            "pipeline_data": self.pipeline_data,
        }


class PipelineUpdated(DomainEvent):
    """Event raised when a pipeline is updated."""

    event_type: ClassVar[str] = "pipeline.updated"

    pipeline_id: str = ""
    pipeline_name: str = ""
    updated_by: str | None = None
    changes: dict[str, object] = Field(default_factory=dict)
    previous_values: dict[str, object] = Field(default_factory=dict)

    def get_event_data(self) -> DomainEventData:
        """Get event data payload."""
        return {
            "pipeline_id": str(self.pipeline_id),
            "pipeline_name": str(self.pipeline_name),
            "updated_by": str(self.updated_by) if self.updated_by else None,
            "changes": self.changes,
            "previous_values": self.previous_values,
        }


class PipelineDeleted(DomainEvent):
    """Event raised when a pipeline is deleted."""

    event_type: ClassVar[str] = "pipeline.deleted"

    pipeline_id: str = ""
    pipeline_name: str = ""
    deleted_by: str | None = None

    def get_event_data(self) -> DomainEventData:
        """Get event data payload."""
        return {
            "pipeline_id": str(self.pipeline_id),
            "pipeline_name": str(self.pipeline_name),
            "deleted_by": str(self.deleted_by) if self.deleted_by else None,
        }


class StepAdded(DomainEvent):
    """Event raised when a step is added to a pipeline."""

    event_type: ClassVar[str] = "pipeline.step.added"

    pipeline_id: str = ""
    step_id: str = ""
    plugin_id: str = ""

    def get_event_data(self) -> DomainEventData:
        """Get event data payload."""
        return {
            "pipeline_id": self.pipeline_id,
            "step_id": self.step_id,
            "plugin_id": self.plugin_id,
        }


class PipelineExecutionStarted(DomainEvent):
    """Event raised when a pipeline execution starts."""

    event_type: ClassVar[str] = "pipeline.execution.started"

    execution_id: str = ""
    pipeline_id: str = ""
    pipeline_name: str = ""
    started_by: str | None = None
    execution_context: dict[str, object] = Field(default_factory=dict)

    def get_event_data(self) -> DomainEventData:
        """Get event data payload."""
        return {
            "execution_id": str(self.execution_id),
            "pipeline_id": str(self.pipeline_id),
            "pipeline_name": str(self.pipeline_name),
            "started_by": str(self.started_by) if self.started_by else None,
            "execution_context": self.execution_context,
        }


class PipelineExecutionCompleted(DomainEvent):
    """Event raised when a pipeline execution completes."""

    event_type: ClassVar[str] = "pipeline.execution.completed"

    execution_id: str = ""
    pipeline_id: str = ""
    pipeline_name: str = ""
    status: str = "success"
    duration_seconds: float = 0.0
    result_data: dict[str, object] = Field(default_factory=dict)

    def get_event_data(self) -> DomainEventData:
        """Get event data payload."""
        return {
            "execution_id": str(self.execution_id),
            "pipeline_id": str(self.pipeline_id),
            "pipeline_name": str(self.pipeline_name),
            "status": str(self.status),
            "duration_seconds": self.duration_seconds,
            "result_data": self.result_data,
        }


# Aliases for backward compatibility
ExecutionStarted = PipelineExecutionStarted
ExecutionCompleted = PipelineExecutionCompleted


# Singleton instance of the event bus
event_bus = EventBus()
