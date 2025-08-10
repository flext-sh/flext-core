"""Protocol definitions for FLEXT ecosystem contracts.

Centralized protocol definitions for validation, services, logging,
and observability across all FLEXT projects.

Protocols:
    FlextValidator: Validation contract with result handling.
    FlextService: Service lifecycle and management contract.
    FlextLogger: Logging interface with structured data.

"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator
    from pathlib import Path

    from flext_core.result import FlextResult

from collections.abc import Callable

T = TypeVar("T")

# =============================================================================
# TYPE ALIASES - Service factory types
# =============================================================================

type FlextServiceFactory = Callable[[], object]
type FlextHandler = FlextMessageHandler

# =============================================================================
# CORE ECOSYSTEM PROTOCOLS - Layer 1 foundational protocols
# =============================================================================


class FlextConnectionProtocol(Protocol):
    """Protocol for external system connections.

    Provides consistent connection interface across all FLEXT projects
    for databases, APIs, message queues, and other external systems.
    """

    def test_connection(self) -> FlextResult[bool]:
        """Test connection to external system."""
        ...

    def get_connection_string(self) -> str:
        """Get connection string for external system."""
        ...

    def close_connection(self) -> FlextResult[None]:
        """Close connection to external system."""
        ...


class FlextAuthProtocol(Protocol):
    """Protocol for authentication and authorization systems."""

    def authenticate(
        self,
        credentials: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Authenticate user with provided credentials."""
        ...

    def authorize(
        self,
        user_info: dict[str, object],
        resource: str,
    ) -> FlextResult[bool]:
        """Authorize user access to resource."""
        ...

    def refresh_token(self, refresh_token: str) -> FlextResult[dict[str, object]]:
        """Refresh authentication token."""
        ...


@runtime_checkable
class FlextObservabilityProtocol(Protocol):
    """Protocol for observability and monitoring systems."""

    def record_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> FlextResult[None]:
        """Record metric value."""
        ...

    def start_trace(self, operation_name: str) -> FlextResult[str]:
        """Start distributed trace."""
        ...

    def health_check(self) -> FlextResult[dict[str, object]]:
        """Perform health check."""
        ...


# =============================================================================
# VALIDATION PROTOCOLS - Centralized from interfaces.py
# =============================================================================


@runtime_checkable
class FlextValidator(Protocol):
    """Protocol for custom validators with flexible validation implementation.

    CONSOLIDATED FROM: interfaces.py FlextValidator
    Runtime-checkable protocol for structural typing validation.
    """

    def validate(self, value: object) -> FlextResult[object]:
        """Validate and potentially transform input value."""
        ...


class FlextValidationRule(Protocol):
    """Protocol for validation rules in validation chains.

    CONSOLIDATED FROM: interfaces.py (abstract base class converted to protocol)
    """

    def apply(self, value: object, field_name: str) -> FlextResult[object]:
        """Apply validation rule to field value."""
        ...

    def get_error_message(self, field_name: str, value: object) -> str:
        """Get error message for validation failure."""
        ...


# =============================================================================
# SERVICE PROTOCOLS - Centralized from interfaces.py
# =============================================================================


class FlextService(Protocol):
    """Protocol for service lifecycle management.

    CONSOLIDATED FROM: interfaces.py FlextService (ABC converted to protocol)
    """

    @abstractmethod
    def start(self) -> FlextResult[None]:
        """Start the service."""
        ...

    @abstractmethod
    def stop(self) -> FlextResult[None]:
        """Stop the service."""
        ...

    @abstractmethod
    def health_check(self) -> FlextResult[dict[str, object]]:
        """Perform health check."""
        ...


@runtime_checkable
class FlextConfigurable(Protocol):
    """Protocol for configurable components.

    CONSOLIDATED FROM: interfaces.py FlextConfigurable
    """

    def configure(self, config: dict[str, object]) -> FlextResult[None]:
        """Configure component with provided settings."""
        ...

    def get_config(self) -> dict[str, object]:
        """Get current configuration."""
        ...


# =============================================================================
# HANDLER PROTOCOLS - Centralized from handlers.py
# =============================================================================


class FlextMessageHandler(Protocol):
    """Protocol for message handling in CQRS patterns.

    CONSOLIDATED FROM: handlers.py FlextHandlerProtocols.FlextMessageHandler
    """

    def handle(self, message: object) -> FlextResult[object]:
        """Handle incoming message and return result."""
        ...

    def can_handle(self, message_type: type) -> bool:
        """Check if handler can process message type."""
        ...


class FlextValidatingHandler(Protocol):
    """Protocol for handlers with validation capabilities.

    CONSOLIDATED FROM: handlers.py FlextHandlerProtocols.ValidatingHandler
    """

    def validate(self, message: object) -> FlextResult[object]:
        """Validate message before processing."""
        ...

    def handle(self, message: object) -> FlextResult[object]:
        """Handle validated message."""
        ...


class FlextAuthorizingHandler(Protocol):
    """Protocol for handlers with authorization capabilities.

    CONSOLIDATED FROM: handlers.py FlextHandlerProtocols.AuthorizingHandler
    """

    def authorize(
        self,
        message: object,
        context: dict[str, object],
    ) -> FlextResult[bool]:
        """Check authorization for message processing."""
        ...

    def handle(self, message: object) -> FlextResult[object]:
        """Handle authorized message."""
        ...


class FlextEventProcessor(Protocol):
    """Protocol for event processing capabilities.

    CONSOLIDATED FROM: handlers.py FlextHandlerProtocols.EventProcessor
    """

    def process_event(self, event: dict[str, object]) -> FlextResult[None]:
        """Process domain event."""
        ...

    def can_process(self, event_type: str) -> bool:
        """Check if processor can handle event type."""
        ...


class FlextMetricsCollector(Protocol):
    """Protocol for metrics collection capabilities.

    CONSOLIDATED FROM: handlers.py FlextHandlerProtocols.MetricsCollector
    """

    def collect_metrics(self, operation: str, duration: float) -> FlextResult[None]:
        """Collect performance metrics."""
        ...

    def get_metrics_summary(self) -> dict[str, object]:
        """Get current metrics summary."""
        ...


# =============================================================================
# DECORATOR PROTOCOLS - Centralized from decorators.py
# =============================================================================


class FlextDecoratedFunction(Protocol):
    """Protocol for decorated function objects.

    CONSOLIDATED FROM: decorators.py FlextDecoratedFunction
    Enables type-safe decorator patterns.
    """

    __name__: str
    __doc__: str | None

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Call the decorated function."""
        ...


# =============================================================================
# OBSERVABILITY PROTOCOLS - Centralized from observability.py
# =============================================================================


@runtime_checkable
class FlextLoggerProtocol(Protocol):
    """Protocol for logger objects with standard logging methods.

    CONSOLIDATED FROM: observability.py FlextLoggerProtocol
    """

    def trace(self, message: str, **kwargs: object) -> None:
        """Log trace message."""
        ...

    def debug(self, message: str, **kwargs: object) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **kwargs: object) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **kwargs: object) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **kwargs: object) -> None:
        """Log error message."""
        ...

    def critical(self, message: str, **kwargs: object) -> None:
        """Log critical message."""
        ...

    def exception(self, message: str, **kwargs: object) -> None:
        """Log exception message."""
        ...


@runtime_checkable
class FlextSpanProtocol(Protocol):
    """Protocol for distributed tracing spans.

    CONSOLIDATED FROM: observability.py FlextSpanProtocol
    """

    def set_tag(self, key: str, value: str) -> None:
        """Set span tag."""
        ...

    def log_event(self, event_name: str, payload: dict[str, object]) -> None:
        """Log event in span."""
        ...

    def finish(self) -> None:
        """Finish the span."""
        ...


@runtime_checkable
class FlextTracerProtocol(Protocol):
    """Protocol for distributed tracers.

    CONSOLIDATED FROM: observability.py FlextTracerProtocol
    """

    def start_span(self, operation_name: str) -> FlextSpanProtocol:
        """Start new tracing span."""
        ...

    def inject_context(self, headers: dict[str, str]) -> None:
        """Inject tracing context into headers."""
        ...


@runtime_checkable
class FlextMetricsProtocol(Protocol):
    """Protocol for metrics collection systems.

    CONSOLIDATED FROM: observability.py FlextMetricsProtocol
    """

    def increment_counter(self, name: str, tags: dict[str, str] | None = None) -> None:
        """Increment counter metric."""
        ...

    def record_gauge(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record gauge metric."""
        ...

    def record_histogram(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record histogram metric."""
        ...


@runtime_checkable
class FlextAlertsProtocol(Protocol):
    """Protocol for simple alerting systems (legacy compatibility)."""

    def info(self, message: str, **kwargs: object) -> None:
        """Send info alert."""
        ...

    def warning(self, message: str, **kwargs: object) -> None:
        """Send warning alert."""
        ...

    def error(self, message: str, **kwargs: object) -> None:
        """Send error alert."""
        ...

    def critical(self, message: str, **kwargs: object) -> None:
        """Send critical alert."""
        ...


# =============================================================================
# PLUGIN PROTOCOLS - Centralized from interfaces.py
# =============================================================================


class FlextPlugin(Protocol):
    """Protocol for plugin system extensions.

    CONSOLIDATED FROM: interfaces.py FlextPlugin (ABC converted to protocol)
    """

    @abstractmethod
    def initialize(self, context: FlextPluginContext) -> FlextResult[None]:
        """Initialize plugin with context."""
        ...

    @abstractmethod
    def shutdown(self) -> FlextResult[None]:
        """Shutdown plugin and cleanup resources."""
        ...

    @abstractmethod
    def get_info(self) -> dict[str, object]:
        """Get plugin information."""
        ...


class FlextPluginContext(Protocol):
    """Protocol for plugin execution context.

    CONSOLIDATED FROM: interfaces.py FlextPluginContext
    """

    def get_service(self, service_name: str) -> FlextResult[object]:
        """Get service instance by name."""
        ...

    def get_config(self) -> dict[str, object]:
        """Get configuration for plugin."""
        ...

    def get_logger(self) -> FlextLoggerProtocol:
        """Get logger instance for plugin."""
        ...


class FlextPluginRegistry(Protocol):
    """Protocol for plugin registry management.

    CONSOLIDATED FROM: interfaces.py FlextPluginRegistry
    """

    def register_plugin(self, plugin: FlextPlugin) -> FlextResult[None]:
        """Register plugin in registry."""
        ...

    def get_plugin(self, plugin_name: str) -> FlextResult[FlextPlugin]:
        """Get plugin by name."""
        ...

    def list_plugins(self) -> list[str]:
        """List all registered plugin names."""
        ...


class FlextPluginLoader(Protocol):
    """Protocol for dynamic plugin loading.

    CONSOLIDATED FROM: interfaces.py FlextPluginLoader
    """

    def load_plugin(self, plugin_path: str | Path) -> FlextResult[FlextPlugin]:
        """Load plugin from file path."""
        ...

    def load_plugins_from_directory(self, directory: str | Path) -> list[FlextPlugin]:
        """Load all plugins from directory."""
        ...

    def unload_plugin(self, plugin: FlextPlugin) -> FlextResult[None]:
        """Unload plugin and cleanup."""
        ...


# =============================================================================
# REPOSITORY PROTOCOLS - Centralized from interfaces.py
# =============================================================================


class FlextRepository[T](Protocol):
    """Protocol for repository pattern implementations.

    CONSOLIDATED FROM: interfaces.py FlextRepository (ABC converted to protocol)
    Modern Python 3.13 generic syntax.
    """

    @abstractmethod
    def get_by_id(self, entity_id: str) -> FlextResult[T | None]:
        """Get entity by ID."""
        ...

    @abstractmethod
    def save(self, entity: T) -> FlextResult[T]:
        """Save entity."""
        ...

    @abstractmethod
    def delete(self, entity_id: str) -> FlextResult[None]:
        """Delete entity by ID."""
        ...

    @abstractmethod
    def find_all(self) -> FlextResult[list[T]]:
        """Find all entities."""
        ...


class FlextUnitOfWork(Protocol):
    """Protocol for Unit of Work pattern.

    CONSOLIDATED FROM: interfaces.py FlextUnitOfWork (ABC converted to protocol)
    """

    @abstractmethod
    def begin(self) -> FlextResult[None]:
        """Begin transaction."""
        ...

    @abstractmethod
    def commit(self) -> FlextResult[None]:
        """Commit transaction."""
        ...

    @abstractmethod
    def rollback(self) -> FlextResult[None]:
        """Rollback transaction."""
        ...


# =============================================================================
# EVENT SOURCING PROTOCOLS - Centralized from interfaces.py
# =============================================================================


class FlextDomainEvent(Protocol):
    """Protocol for domain events in event sourcing.

    CONSOLIDATED FROM: interfaces.py FlextDomainEvent
    """

    event_id: str
    event_type: str
    aggregate_id: str
    event_version: int
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        """Convert event to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FlextDomainEvent:
        """Create event from dictionary."""
        ...


class FlextEventStore(Protocol):
    """Protocol for event store implementations.

    CONSOLIDATED FROM: interfaces.py FlextEventStore (ABC converted to protocol)
    """

    @abstractmethod
    def save_events(
        self,
        aggregate_id: str,
        events: list[FlextDomainEvent],
        expected_version: int,
    ) -> FlextResult[None]:
        """Save events for aggregate."""
        ...

    @abstractmethod
    def get_events(self, aggregate_id: str) -> FlextResult[list[FlextDomainEvent]]:
        """Get events for aggregate."""
        ...

    @abstractmethod
    def get_events_from_version(
        self,
        aggregate_id: str,
        from_version: int,
    ) -> FlextResult[list[FlextDomainEvent]]:
        """Get events from specific version."""
        ...


class FlextEventPublisher(Protocol):
    """Protocol for event publishing.

    CONSOLIDATED FROM: interfaces.py FlextEventPublisher (ABC converted to protocol)
    """

    @abstractmethod
    def publish(self, event: FlextDomainEvent) -> FlextResult[None]:
        """Publish domain event."""
        ...

    @abstractmethod
    def publish_batch(self, events: list[FlextDomainEvent]) -> FlextResult[None]:
        """Publish batch of events."""
        ...


class FlextEventSubscriber(Protocol):
    """Protocol for event subscription.

    CONSOLIDATED FROM: interfaces.py FlextEventSubscriber (ABC converted to protocol)
    """

    @abstractmethod
    def handle_event(self, event: FlextDomainEvent) -> FlextResult[None]:
        """Handle received event."""
        ...

    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if subscriber can handle event type."""
        ...


class FlextEventStreamReader(Protocol):
    """Protocol for reading event streams.

    CONSOLIDATED FROM: interfaces.py FlextEventStreamReader (ABC converted to protocol)
    """

    @abstractmethod
    def read_stream(
        self,
        stream_name: str,
        from_position: int = 0,
    ) -> FlextResult[Iterator[FlextDomainEvent]]:
        """Read events from stream."""
        ...

    @abstractmethod
    def subscribe_to_stream(
        self,
        stream_name: str,
        handler: Callable[[FlextDomainEvent], None],
    ) -> FlextResult[None]:
        """Subscribe to stream events."""
        ...


class FlextProjectionBuilder(Protocol):
    """Protocol for building projections from events.

    CONSOLIDATED FROM: interfaces.py FlextProjectionBuilder (ABC converted to protocol)
    """

    @abstractmethod
    def build_projection(
        self,
        events: list[FlextDomainEvent],
    ) -> FlextResult[dict[str, object]]:
        """Build projection from events."""
        ...

    @abstractmethod
    def update_projection(
        self,
        projection: dict[str, object],
        event: FlextDomainEvent,
    ) -> FlextResult[dict[str, object]]:
        """Update projection with new event."""
        ...


# =============================================================================
# ASYNC PROTOCOLS - Advanced async patterns
# =============================================================================


class FlextAsyncHandler(Protocol):
    """Protocol for async message handlers."""

    async def handle_async(self, message: object) -> FlextResult[object]:
        """Handle message asynchronously."""
        ...

    def can_handle(self, message_type: type) -> bool:
        """Check if handler can process message type."""
        ...


class FlextAsyncService(Protocol):
    """Protocol for async service lifecycle."""

    async def start_async(self) -> FlextResult[None]:
        """Start service asynchronously."""
        ...

    async def stop_async(self) -> FlextResult[None]:
        """Stop service asynchronously."""
        ...

    async def health_check_async(self) -> FlextResult[dict[str, object]]:
        """Perform async health check."""
        ...


# =============================================================================
# FACTORY PROTOCOLS - Type-safe factory patterns
# =============================================================================


class FlextFactory[T](Protocol):
    """Protocol for type-safe factory implementations.

    Modern Python 3.13 generic syntax for factory pattern.
    """

    def create(self, **kwargs: object) -> FlextResult[T]:
        """Create instance of type T."""
        ...


class FlextAsyncFactory[T](Protocol):
    """Protocol for async factory implementations."""

    async def create_async(self, **kwargs: object) -> FlextResult[T]:
        """Create instance asynchronously."""
        ...


# =============================================================================
# MIDDLEWARE PROTOCOLS - Pipeline and middleware patterns
# =============================================================================


class FlextMiddleware(Protocol):
    """Protocol for middleware pipeline components."""

    def process(
        self,
        request: object,
        next_handler: Callable[[object], FlextResult[object]],
    ) -> FlextResult[object]:
        """Process request with middleware logic."""
        ...


class FlextAsyncMiddleware(Protocol):
    """Protocol for async middleware components."""

    async def process_async(
        self,
        request: object,
        next_handler: Callable[[object], Awaitable[FlextResult[object]]],
    ) -> FlextResult[object]:
        """Process request asynchronously."""
        ...


# =============================================================================
# EXPORTS - All centralized protocols
# =============================================================================

__all__: list[str] = [
    "FlextAlertsProtocol",
    "FlextAsyncFactory",
    "FlextAsyncHandler",
    "FlextAsyncMiddleware",
    "FlextAsyncService",
    "FlextAuthProtocol",
    "FlextAuthorizingHandler",
    "FlextConfigurable",
    "FlextConnectionProtocol",
    "FlextDecoratedFunction",
    "FlextDomainEvent",
    "FlextEventProcessor",
    "FlextEventPublisher",
    "FlextEventStore",
    "FlextEventStreamReader",
    "FlextEventSubscriber",
    "FlextFactory",
    "FlextHandler",
    "FlextLoggerProtocol",
    "FlextMessageHandler",
    "FlextMetricsCollector",
    "FlextMetricsProtocol",
    "FlextMiddleware",
    "FlextObservabilityProtocol",
    "FlextPlugin",
    "FlextPluginContext",
    "FlextPluginLoader",
    "FlextPluginRegistry",
    "FlextProjectionBuilder",
    "FlextRepository",
    "FlextService",
    "FlextServiceFactory",
    "FlextSpanProtocol",
    "FlextTracerProtocol",
    "FlextUnitOfWork",
    "FlextValidatingHandler",
    "FlextValidationRule",
    "FlextValidator",
]
