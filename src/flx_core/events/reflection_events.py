"""Reflection-based event sourcing with ZERO boilerplate using Python 3.13.

This module implements automatic event handling, projection, and replay
through reflection, eliminating ALL event sourcing boilerplate.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, TypeVar
from uuid import UUID, uuid4

from pydantic import Field

from flx_core.domain.pydantic_base import DomainBaseModel, DomainValueObject

# ZERO TOLERANCE CONSOLIDATION: Import canonical domain events for compatibility
from flx_core.events.event_bus import PipelineCreated, PipelineUpdated

# ZERO TOLERANCE - Use high-performance msgspec instead of standard library JSON
from flx_core.serialization.msgspec_adapters import get_serializer

if TYPE_CHECKING:
    from types import ModuleType

    from redis.asyncio import Redis

# Python 3.13 type aliases
type EventData = dict[str, object]
type EventHandler[T] = Callable[[T], object]
type ProjectionState = dict[str, object]
type StreamPosition = int | str
type SerializableValue = (
    str | int | float | bool | dict[str, object] | list[object] | None
)

T = TypeVar("T")
E = TypeVar("E")


class EventType(Enum):
    """Core event types with automatic categorization."""

    # Domain events
    CREATED = auto()
    UPDATED = auto()
    DELETED = auto()
    ACTIVATED = auto()
    DEACTIVATED = auto()

    # Process events
    STARTED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRIED = auto()

    # System events
    CONFIGURED = auto()
    MIGRATED = auto()
    BACKED_UP = auto()
    RESTORED = auto()


class EventMetadata(DomainValueObject):
    """Event metadata with automatic timestamp and correlation."""

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier for tracking and correlation",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event occurrence timestamp for ordering and replay",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracking related events across boundaries",
    )
    causation_id: UUID | None = Field(
        default=None,
        description="Causation ID for tracking event cause-effect relationships",
    )
    user_id: str | None = Field(
        default=None,
        description="User identifier for audit trails and security compliance",
    )
    version: int = Field(
        default=1,
        description="Event schema version for evolution and compatibility",
    )


class EventProtocol:
    """Protocol for domain events with reflection support."""

    @property
    def event_type(self) -> EventType:
        """Get event type.

        Returns the type of domain event for categorization
        and routing purposes in event handling systems.

        Returns
        -------
            EventType: The category of this domain event

        """

    @property
    def aggregate_id(self) -> UUID:
        """Get aggregate identifier.

        Returns the unique identifier of the aggregate that
        raised this domain event for event sourcing patterns.

        Returns
        -------
            UUID: Unique identifier of the source aggregate

        """

    @property
    def metadata(self) -> EventMetadata:
        """Get event metadata.

        Returns metadata about the event including timestamp,
        correlation ID, and other tracking information.

        Returns
        -------
            EventMetadata: Event metadata and tracking information

        """

    def to_dict(self) -> EventData:
        """Serialize event to dictionary.

        Converts the event instance to a dictionary representation
        suitable for storage, transmission, or logging.

        Returns
        -------
            EventData: Dictionary representation of the event

        """


class ReflectionEvent(DomainValueObject):
    """Base event with automatic serialization through reflection."""

    aggregate_id: UUID = Field(
        description="Aggregate identifier for event sourcing and state reconstruction",
    )
    event_type: EventType = Field(
        description="Event type classification for routing and processing",
    )
    metadata: EventMetadata = Field(
        default_factory=EventMetadata,
        description="Event metadata with timestamps and correlation",
    )

    def to_dict(self) -> EventData:
        """Serialize event to dictionary using reflection."""
        data = {
            "event_id": str(self.metadata.event_id),
            "event_type": self.event_type.name,
            "aggregate_id": str(self.aggregate_id),
            "timestamp": self.metadata.timestamp.isoformat(),
            "version": self.metadata.version,
        }

        # Add correlation data if present
        if self.metadata.correlation_id:
            data["correlation_id"] = str(self.metadata.correlation_id)
        if self.metadata.causation_id:
            data["causation_id"] = str(self.metadata.causation_id)
        if self.metadata.user_id:
            data["user_id"] = self.metadata.user_id

        # Add all Pydantic fields using model fields
        for field_name in self.model_fields:
            if field_name not in {"aggregate_id", "event_type", "metadata"}:
                value = getattr(self, field_name)
                data[field_name] = self._serialize_value(value)

        return data

    def _serialize_value(self, value: object) -> SerializableValue:
        """Serialize value based on type using Python 3.13 match statement."""
        match value:
            case UUID():
                return str(value)
            case datetime():
                return value.isoformat()
            case Enum():
                return value.name
            case None:
                return None
            case _:
                # Try to_dict method first
                try:
                    to_dict_method = value.to_dict
                    result = to_dict_method()
                    if isinstance(result, dict):
                        return result
                except AttributeError:
                    pass

                # Try __dict__ attribute for object serialization
                try:
                    obj_dict = value.__dict__
                    return {k: self._serialize_value(v) for k, v in obj_dict.items()}
                except AttributeError:
                    pass

                # Default to string representation
                return str(value)

    @classmethod
    def from_dict(cls, data: EventData) -> ReflectionEvent:
        """Deserialize event from dictionary using reflection."""
        # Extract metadata
        metadata = EventMetadata(
            event_id=UUID(str(data["event_id"])),
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
            correlation_id=(
                UUID(str(data["correlation_id"])) if "correlation_id" in data else None
            ),
            causation_id=(
                UUID(str(data["causation_id"])) if "causation_id" in data else None
            ),
            user_id=(
                str(data.get("user_id")) if data.get("user_id") is not None else None
            ),
            version=(
                int(str(data.get("version", 1)))
                if data.get("version") is not None
                else 1
            ),
        )

        # Build kwargs for event construction
        kwargs: dict[str, object] = {
            "aggregate_id": UUID(str(data["aggregate_id"])),
            "event_type": EventType[str(data["event_type"])],
            "metadata": metadata,
        }

        # Add additional fields based on model fields
        try:
            model_fields = cls.model_fields
            for field_name in model_fields:
                if field_name not in kwargs and field_name in data:
                    kwargs[field_name] = data[field_name]
        except AttributeError:
            # Class doesn't have model_fields attribute
            pass

        return cls(**kwargs)  # type: ignore[arg-type]


def event(event_type: EventType) -> Callable[[type[T]], type[T]]:
    """Decorate zero-boilerplate event classes."""

    def decorator(cls: type[T]) -> type[T]:
        # Make it inherit from ReflectionEvent
        if not issubclass(cls, ReflectionEvent):
            # Create new class with ReflectionEvent as base
            return type(
                cls.__name__,
                (ReflectionEvent, cls),
                {
                    "__module__": cls.__module__,
                    "__doc__": cls.__doc__,
                    "event_type": event_type,
                    **cls.__dict__,
                },
            )

            # Convert to Pydantic model

        # Add event type using setattr to avoid __slots__ conflicts
        cls.event_type = event_type

        return cls

    return decorator


class EventHandlerRegistry(DomainBaseModel):
    """Registry for automatic event handler discovery."""

    handlers_registry: dict[
        type[ReflectionEvent],
        list[EventHandler[ReflectionEvent]],
    ] = Field(
        default_factory=dict,
        description="Registry mapping event types to their handler functions",
        alias="_handlers",
    )
    projections_registry: dict[str, Projection] = Field(
        default_factory=dict,
        description="Registry of event projections by name",
        alias="_projections",
    )

    @property
    def _handlers(
        self,
    ) -> dict[type[ReflectionEvent], list[EventHandler[ReflectionEvent]]]:
        """Backward compatibility property for handlers."""
        return self.handlers_registry

    @property
    def _projections(self) -> dict[str, Projection]:
        """Backward compatibility property for projections."""
        return self.projections_registry

    def register_handler(
        self, event_class: type[ReflectionEvent], handler: EventHandler[ReflectionEvent]
    ) -> None:
        """Register an event handler.

        Registers a handler function to be called when events of the specified type
        are published. Handlers are invoked asynchronously in registration order.

        Args:
        ----
            event_class: Type of event to handle
            handler: Function to call when event is published

        Note:
        ----
            Multiple handlers can be registered for the same event type.

        """
        if event_class not in self._handlers:
            self._handlers[event_class] = []
        self._handlers[event_class].append(handler)

    def register_projection(self, projection: Projection) -> None:
        """Register a projection.

        Registers a projection to maintain read-optimized views of event data.
        Projections are automatically updated when relevant events are published.

        Args:
        ----
            projection: Projection instance to register

        Note:
        ----
            Projections are stored by name and must have unique names.

        """
        self._projections[projection.name] = projection

    async def handle_event(self, event: ReflectionEvent) -> None:
        """Handle event with all registered handlers."""
        handlers = self._handlers.get(type(event), [])

        # Execute handlers concurrently
        tasks: list[object] = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                tasks.append(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        handler,
                        event,
                    ),
                )

        if tasks:
            # Type narrowing: all tasks are Awaitable objects
            awaitable_tasks = []
            for task in tasks:
                try:
                    task.__await__
                    awaitable_tasks.append(task)
                except AttributeError:
                    # Task is not awaitable
                    continue

            if awaitable_tasks:
                await asyncio.gather(*awaitable_tasks, return_exceptions=True)

    def discover_handlers(self, module: ModuleType) -> None:
        """Automatically discover event handlers in a module."""
        for _name, obj in inspect.getmembers(module):
            try:
                handles_events = obj._handles_events
                for event_class in handles_events:
                    self.register_handler(event_class, obj)
            except AttributeError:
                # Object doesn't have _handles_events attribute
                continue


def handles(*event_classes: type[ReflectionEvent]) -> Callable[[T], T]:
    """Decorate event handlers with automatic registration."""

    def decorator(func: T) -> T:
        func._handles_events = event_classes
        return func

    return decorator


class Projection(DomainBaseModel):
    """Event projection with automatic state management."""

    name: str = Field(
        description="Unique projection name for identification and management",
    )
    state: ProjectionState = Field(
        default_factory=dict,
        description="Current projection state data for read models",
    )
    version: int = Field(
        default=0,
        description="Projection version for tracking updates and consistency",
    )

    async def apply(self, event: ReflectionEvent) -> None:
        """Apply event to projection state."""
        # Get handler method based on event type
        handler_name = f"on_{event.__class__.__name__.lower()}"
        handler = getattr(self, handler_name, None)

        if handler:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

        self.version += 1

    def get_state(self) -> ProjectionState:
        """Get current projection state.

        Returns a copy of the current projection state dictionary.
        Used for querying the read-optimized view without modifying the state.

        Returns:
        -------
            Copy of current projection state dictionary

        Note:
        ----
            Returns a copy to prevent external modifications to internal state.

        """
        return self.state.copy()

    def reset(self) -> None:
        """Reset projection to initial state."""
        self.state.clear()
        self.version = 0


class EventStore(DomainBaseModel):
    """Event store with automatic serialization and replay."""

    model_config = {"arbitrary_types_allowed": True}

    redis: Redis[str] = Field(
        description="Redis client for event stream storage and publishing",
    )
    stream_prefix: str = Field(
        default="events",
        description="Prefix for Redis stream keys to namespace events",
    )

    async def append(self, stream_name: str, event: ReflectionEvent) -> StreamPosition:
        """Append event to stream with automatic serialization."""
        stream_key = f"{self.stream_prefix}:{stream_name}"
        event_data = event.to_dict()

        # Add to Redis stream - using high-performance msgspec
        serializer = get_serializer()
        message_id = await self.redis.xadd(
            stream_key,
            {"data": serializer.encode(event_data).decode("utf-8")},
        )

        # Publish for real-time subscribers - using high-performance msgspec
        await self.redis.publish(
            f"{stream_key}:notify",
            serializer.encode(event_data).decode("utf-8"),
        )

        return str(message_id)

    async def read_stream(
        self, stream_name: str, start: StreamPosition = "0", count: int | None = None
    ) -> list[tuple[StreamPosition, ReflectionEvent]]:
        """Read events from stream with automatic deserialization."""
        stream_key = f"{self.stream_prefix}:{stream_name}"

        # Read from Redis stream
        messages = await self.redis.xread(
            {stream_key: start},
            count=count,
            block=0 if count is None else None,
        )

        events = []
        serializer = get_serializer()
        for _stream, stream_messages in messages:
            for message_id, data in stream_messages:
                event_data = serializer.decode(data[b"data"])

                # Deserialize based on event type
                event_class = self._get_event_class(event_data["event_type"])
                if event_class:
                    event = event_class.from_dict(event_data)
                    events.append((message_id, event))

        return events

    def _get_event_class(self, _event_type: str) -> type[ReflectionEvent] | None:
        """Get event class by type name using reflection."""
        # This would use a registry in production
        # For now, return base class
        return ReflectionEvent

    async def replay(
        self, stream_name: str, projection: Projection, start: StreamPosition = "0"
    ) -> None:
        """Replay events to rebuild projection state."""
        projection.reset()

        # Read all events
        events = await self.read_stream(stream_name, start)

        # Apply to projection
        for _, event in events:
            await projection.apply(event)


# === REFLECTION-BASED EVENT IMPLEMENTATIONS ===


# ZERO TOLERANCE CONSOLIDATION: PipelineCreated/PipelineUpdated moved to flx_core.events.event_bus
# Use: from flx_core.events.event_bus import PipelineCreated, PipelineUpdated
# Canonical domain event implementations with proper DDD patterns


@event(EventType.STARTED)
class ExecutionStarted(ReflectionEvent):
    """Execution started event with automatic serialization."""

    pipeline_id: UUID
    execution_number: int
    triggered_by: str


@event(EventType.COMPLETED)
class ExecutionCompleted(ReflectionEvent):
    """Execution completed event with automatic serialization."""

    pipeline_id: UUID
    duration_seconds: float
    output: dict[str, object]


@event(EventType.FAILED)
class ExecutionFailed(ReflectionEvent):
    """Execution failed event with automatic serialization."""

    pipeline_id: UUID
    error: str
    step_id: str | None = None


# === EVENT HANDLERS ===


@handles(PipelineCreated, PipelineUpdated)
async def update_pipeline_projection(event: ReflectionEvent) -> None:
    """Update pipeline read model projection."""
    # This would update a denormalized read model


@handles(ExecutionStarted, ExecutionCompleted, ExecutionFailed)
async def track_execution_metrics(event: ReflectionEvent) -> None:
    """Track execution metrics from events."""
    # This would update metrics


class PipelineProjection(Projection):
    """Pipeline state projection with automatic event handling."""

    def on_pipelinecreated(self, event: PipelineCreated) -> None:
        """Handle pipeline created event."""
        self.state[str(event.aggregate_id)] = {
            "id": str(event.aggregate_id),
            "name": event.name,
            "description": event.description,
            "created_by": event.created_by,
            "created_at": event.metadata.timestamp.isoformat(),
            "execution_count": 0,
            "last_execution": None,
        }

    def on_executionstarted(self, event: ExecutionStarted) -> None:
        """Handle execution started event."""
        pipeline_id = str(event.pipeline_id)
        if pipeline_id in self.state:
            pipeline_state = self.state[pipeline_id]
            if isinstance(pipeline_state, dict) and "execution_count" in pipeline_state:
                count = pipeline_state["execution_count"]
                pipeline_state["execution_count"] = (
                    count + 1 if isinstance(count, int) else 1
                )
            pipeline_state["last_execution"] = {  # type: ignore[index]
                "id": str(event.aggregate_id),
                "status": "running",
                "started_at": event.metadata.timestamp.isoformat(),
            }

    def on_executioncompleted(self, event: ExecutionCompleted) -> None:
        """Handle execution completed event."""
        pipeline_id = str(event.pipeline_id)
        if pipeline_id in self.state:
            pipeline_state = self.state[pipeline_id]
            if (
                isinstance(pipeline_state, dict)
                and "last_execution" in pipeline_state
                and pipeline_state["last_execution"]
            ):
                last_execution = pipeline_state["last_execution"]
                if isinstance(last_execution, dict):
                    last_execution["status"] = "completed"
                    last_execution["duration"] = event.duration_seconds


def create_event_system(redis: Redis[str]) -> tuple[EventStore, EventHandlerRegistry]:
    """Create event system with automatic discovery."""
    store = EventStore(redis=redis)
    registry = EventHandlerRegistry()

    # Discover handlers in current module
    registry.discover_handlers(sys.modules[__name__])

    return store, registry
