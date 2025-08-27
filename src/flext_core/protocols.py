"""Protocol definitions for FLEXT ecosystem contracts.

Core protocol definitions that avoid ALL circular imports by using string annotations
and minimal external dependencies. This is a foundation module that should not
import from other flext_core modules at runtime.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import ParamSpec, Protocol, TypeVar, runtime_checkable

# Import only essential types to avoid circular dependencies
# Currently no TYPE_CHECKING imports needed

# ParamSpec and TypeVar for generic callable protocols
P = ParamSpec("P")
T = TypeVar("T")

# =============================================================================
# HIERARCHICAL PROTOCOL ARCHITECTURE - Optimized with composition
# =============================================================================


class FlextProtocols:
    """Hierarchical protocol architecture with optimized composition patterns.

    This class implements a structured hierarchy where subclasses compose with each
    other using Python 3.13+ syntax. The architecture follows Clean Architecture
    principles with clear separation of concerns and dependency inversion.

    Hierarchy Structure:
        Foundation -> Domain -> Application -> Infrastructure -> Extensions

    Composition Features:
        - Type-safe protocol inheritance with Python 3.13+ generics
        - Mixin composition for cross-cutting concerns
        - Runtime checkable protocols for dynamic validation
        - FlextResult integration for railway-oriented programming

    Examples:
        Foundation layer usage::

            processor: FlextProtocols.Foundation.Callable[str] = lambda x: x.upper()
            result = processor("test")  # Returns "TEST"

        Domain composition::

            class UserService(
                FlextProtocols.Domain.Service, FlextProtocols.Foundation.Validator[User]
            ):
                def validate(self, user: User) -> object: ...
                def start(self) -> object: ...

        Application layer patterns::

            handler: FlextProtocols.Application.Handler[CreateUser, str] = (
                CreateUserHandler()
            )
            result = handler(CreateUser(name="John"))  # Returns Any

    """

    # =========================================================================
    # FOUNDATION LAYER - Core building blocks
    # =========================================================================

    class Foundation:
        """Foundation layer protocols - core building blocks for the ecosystem."""

        class Callable[T](Protocol):
            """Generic callable protocol with parameter and return type safety."""

            def __call__(self, *args: object, **kwargs: object) -> T:
                """Execute the callable with given arguments."""
                ...

        @runtime_checkable
        class DecoratedCallable[T](Protocol):
            """Callable protocol with function attributes for decorators.

            This protocol represents a function that has been decorated and
            includes all standard function attributes.
            """

            def __call__(self, *args: object, **kwargs: object) -> T:
                """Execute the callable with given arguments."""
                ...

            __name__: str
            __module__: str
            __doc__: str | None
            __qualname__: str
            __annotations__: dict[str, object]
            __dict__: dict[str, object]
            __wrapped__: object | None  # Can be any callable

        class Validator[T](Protocol):
            """Generic validator protocol for type-safe validation."""

            def validate(self, data: T) -> object:
                """Validate input data and return success/failure status."""
                ...

        class ErrorHandler(Protocol):
            """Error handler protocol for exception transformation."""

            def handle_error(self, error: Exception) -> str:
                """Transform an exception into an error message string."""
                ...

        class Factory[T](Protocol):
            """Type-safe factory protocol for object creation."""

            def create(self, **kwargs: object) -> object:
                """Create instance of type T."""
                ...

        class AsyncFactory[T](Protocol):
            """Async factory protocol for asynchronous object creation."""

            async def create_async(self, **kwargs: object) -> object:
                """Create instance asynchronously."""
                ...

    # =========================================================================
    # DOMAIN LAYER - Business logic protocols
    # =========================================================================

    class Domain:
        """Domain layer protocols - business logic and domain services."""

        class Service(Protocol):
            """Domain service protocol with lifecycle management and callable interface."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for service invocation."""
                ...

            @abstractmethod
            def start(self) -> object:
                """Start the service."""
                ...

            @abstractmethod
            def stop(self) -> object:
                """Stop the service."""
                ...

            @abstractmethod
            def health_check(self) -> object:
                """Perform health check."""
                ...

        class Repository[T](Protocol):
            """Repository protocol for data access patterns."""

            @abstractmethod
            def get_by_id(self, entity_id: str) -> object:
                """Get entity by ID."""
                ...

            @abstractmethod
            def save(self, entity: T) -> object:
                """Save entity."""
                ...

            @abstractmethod
            def delete(self, entity_id: str) -> object:
                """Delete entity by ID."""
                ...

            @abstractmethod
            def find_all(self) -> object:
                """Find all entities."""
                ...

        class DomainEvent(Protocol):
            """Domain event protocol for event sourcing."""

            event_id: str
            event_type: str
            aggregate_id: str
            event_version: int
            timestamp: str

            def to_dict(self) -> dict[str, object]:
                """Convert event to dictionary."""
                ...

            @classmethod
            def from_dict(
                cls, data: dict[str, object]
            ) -> FlextProtocols.Domain.DomainEvent:
                """Create event from dictionary."""
                ...

        class EventStore(Protocol):
            """Event store protocol for domain event persistence."""

            @abstractmethod
            def save_events(
                self,
                aggregate_id: str,
                events: list[FlextProtocols.Domain.DomainEvent],
                expected_version: int,
            ) -> object:
                """Save events for aggregate."""
                ...

            @abstractmethod
            def get_events(self, aggregate_id: str) -> object:
                """Get events for aggregate."""
                ...

    # =========================================================================
    # APPLICATION LAYER - Use cases and handlers
    # =========================================================================

    class Application:
        """Application layer protocols - use cases, handlers, and orchestration."""

        class Handler[TInput, TOutput](Protocol):
            """Application handler with validation and processing."""

            def __call__(self, input_data: TInput) -> object:
                """Process input data and return transformed output."""
                ...

            def validate(self, data: TInput) -> object:
                """Validate input before processing (Foundation.Validator composition)."""
                ...

        class MessageHandler(Protocol):
            """Message handler for CQRS patterns."""

            def handle(self, message: object) -> object:
                """Handle incoming message and return result."""
                ...

            def can_handle(self, message_type: type) -> bool:
                """Check if handler can process a message type."""
                ...

        class ValidatingHandler(MessageHandler, Protocol):
            """Handler with built-in validation capabilities."""

            def validate(self, message: object) -> object:
                """Validate message before processing (Foundation.Validator composition)."""
                ...

        class AuthorizingHandler(MessageHandler, Protocol):
            """Handler with authorization capabilities."""

            def authorize(
                self,
                message: object,
                context: dict[str, object],
            ) -> object:
                """Check authorization for message processing."""
                ...

        class EventProcessor(Protocol):
            """Event processor for domain event handling."""

            def process_event(self, event: dict[str, object]) -> object:
                """Process domain event."""
                ...

            def can_process(self, event_type: str) -> bool:
                """Check if the processor can handle an event type."""
                ...

        class UnitOfWork(Protocol):
            """Unit of Work pattern for transaction management."""

            @abstractmethod
            def begin(self) -> object:
                """Begin transaction."""
                ...

            @abstractmethod
            def commit(self) -> object:
                """Commit transaction."""
                ...

            @abstractmethod
            def rollback(self) -> object:
                """Rollback transaction."""
                ...

    # =========================================================================
    # INFRASTRUCTURE LAYER - External concerns and integrations
    # =========================================================================

    class Infrastructure:
        """Infrastructure layer protocols - external systems and cross-cutting concerns."""

        class Connection(Protocol):
            """Connection protocol for external systems with callable interface."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for connection operations."""
                ...

            def test_connection(self) -> object:
                """Test connection to an external system."""
                ...

            def get_connection_string(self) -> str:
                """Get connection string for an external system."""
                ...

            def close_connection(self) -> object:
                """Close connection to an external system."""
                ...

        class LdapConnection(Connection, Protocol):
            """LDAP-specific connection protocol extending base Connection."""

            def connect(self, uri: str, bind_dn: str, password: str) -> object:
                """Connect to LDAP server with authentication."""
                ...

            def bind(self, bind_dn: str, password: str) -> object:
                """Bind with specific credentials."""
                ...

            def unbind(self) -> object:
                """Unbind from LDAP server."""
                ...

            def search(
                self, base_dn: str, search_filter: str, scope: str = "subtree"
            ) -> object:
                """Perform LDAP search operation."""
                ...

            def add(self, dn: str, attributes: dict[str, object]) -> object:
                """Add new LDAP entry."""
                ...

            def modify(self, dn: str, modifications: dict[str, object]) -> object:
                """Modify existing LDAP entry."""
                ...

            def delete(self, dn: str) -> object:
                """Delete LDAP entry."""
                ...

            def is_connected(self) -> bool:
                """Check if connection is active."""
                ...

        class Auth(Protocol):
            """Authentication and authorization protocol."""

            def authenticate(
                self,
                credentials: dict[str, object],
            ) -> object:
                """Authenticate user with provided credentials."""
                ...

            def authorize(
                self,
                user_info: dict[str, object],
                resource: str,
            ) -> object:
                """Authorize user access to resource."""
                ...

            def refresh_token(self, refresh_token: str) -> object:
                """Refresh authentication token."""
                ...

        @runtime_checkable
        class Configurable(Protocol):
            """Configurable component protocol."""

            def configure(self, config: dict[str, object]) -> object:
                """Configure component with provided settings."""
                ...

            def get_config(self) -> dict[str, object]:
                """Get current configuration."""
                ...

        @runtime_checkable
        class LoggerProtocol(Protocol):
            """Logger protocol with standard logging methods."""

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

            def exception(
                self,
                message: str,
                *,
                exc_info: bool = True,
                **kwargs: object,
            ) -> None:
                """Log exception message."""
                ...

    # =========================================================================
    # EXTENSIONS LAYER - Advanced patterns and plugins
    # =========================================================================

    class Extensions:
        """Extensions layer protocols - plugins, middleware, and advanced patterns."""

        class Plugin(Protocol):
            """Plugin protocol with configuration support."""

            def configure(self, config: dict[str, object]) -> object:
                """Configure component with provided settings (Infrastructure.Configurable composition)."""
                ...

            def get_config(self) -> dict[str, object]:
                """Get current configuration (Infrastructure.Configurable composition)."""
                ...

            @abstractmethod
            def initialize(
                self, context: FlextProtocols.Extensions.PluginContext
            ) -> object:
                """Initialize plugin with context."""
                ...

            @abstractmethod
            def shutdown(self) -> object:
                """Shutdown plugin and cleanup resources."""
                ...

            @abstractmethod
            def get_info(self) -> dict[str, object]:
                """Get plugin information."""
                ...

        class PluginContext(Protocol):
            """Plugin execution context protocol."""

            def get_service(self, service_name: str) -> object:
                """Get service instance by name."""
                ...

            def get_config(self) -> dict[str, object]:
                """Get configuration for plugin."""
                ...

            def get_logger(self) -> FlextProtocols.Infrastructure.LoggerProtocol:
                """Get logger instance for plugin."""
                ...

        class Middleware(Protocol):
            """Middleware pipeline component protocol."""

            def process(
                self,
                request: object,
                next_handler: Callable[[object], object],
            ) -> object:
                """Process request with middleware logic."""
                ...

        class AsyncMiddleware(Protocol):
            """Async middleware component protocol."""

            async def process_async(
                self,
                request: object,
                next_handler: Callable[[object], Awaitable[object]],
            ) -> object:
                """Process request asynchronously."""
                ...

        @runtime_checkable
        class Observability(Protocol):
            """Observability and monitoring protocol."""

            def record_metric(
                self,
                name: str,
                value: float,
                tags: dict[str, str] | None = None,
            ) -> object:
                """Record metric value."""
                ...

            def start_trace(self, operation_name: str) -> object:
                """Start distributed trace."""
                ...

            def health_check(self) -> object:
                """Perform health check."""
                ...

    # =========================================================================
    # BACKWARD COMPATIBILITY - Legacy protocol definitions
    # =========================================================================

    # =========================================================================
    # COMPATIBILITY LAYER - Optimized aliases for hierarchical access
    # =========================================================================

    # Direct access to hierarchical protocols
    Callable = Foundation.Callable
    Validator = Foundation.Validator
    ErrorHandler = Foundation.ErrorHandler
    Factory = Foundation.Factory
    AsyncFactory = Foundation.AsyncFactory

    # Domain layer access
    Service = Domain.Service
    Repository = Domain.Repository
    DomainEvent = Domain.DomainEvent
    EventStore = Domain.EventStore

    # Application layer access
    Handler = Application.Handler
    MessageHandler = Application.MessageHandler
    ValidatingHandler = Application.ValidatingHandler
    AuthorizingHandler = Application.AuthorizingHandler
    EventProcessor = Application.EventProcessor
    UnitOfWork = Application.UnitOfWork

    # Infrastructure layer access
    Connection = Infrastructure.Connection
    Auth = Infrastructure.Auth
    Configurable = Infrastructure.Configurable
    LoggerProtocol = Infrastructure.LoggerProtocol

    # Extensions layer access
    Plugin = Extensions.Plugin
    PluginContext = Extensions.PluginContext
    Middleware = Extensions.Middleware
    AsyncMiddleware = Extensions.AsyncMiddleware
    Observability = Extensions.Observability


# =============================================================================
# DECORATOR PROTOCOLS - Special function patterns
# =============================================================================

# =============================================================================
# DECORATOR PROTOCOLS - Special function patterns
# =============================================================================


class DecoratedFunction[T](Protocol):
    """Decorated function protocol returning FlextResult for railway-oriented programming."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute the decorated function returning FlextResult."""
        ...


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Core protocols aliases for backward compatibility
FlextProtocol = FlextProtocols  # Legacy name

# Foundation layer aliases - commented to avoid conflicts with typings.py
# FlextCallable = FlextProtocols.Foundation.Callable - moved to typings.py
FlextValidator = FlextProtocols.Foundation.Validator
# FlextErrorHandler = FlextProtocols.Foundation.ErrorHandler - moved to typings.py
# FlextFactory = FlextProtocols.Foundation.Factory - moved to typings.py
# FlextAsyncFactory = FlextProtocols.Foundation.AsyncFactory - moved to typings.py

# Domain layer aliases
FlextService = FlextProtocols.Domain.Service
FlextRepository = FlextProtocols.Domain.Repository
FlextDomainEvent = FlextProtocols.Domain.DomainEvent
FlextEventStore = FlextProtocols.Domain.EventStore

# Application layer aliases - commented to avoid conflicts with typings.py
# FlextHandler = FlextProtocols.Application.Handler - moved to typings.py
# FlextMessageHandler = FlextProtocols.Application.MessageHandler - moved to typings.py
# FlextValidatingHandler = FlextProtocols.Application.ValidatingHandler - moved to typings.py
# FlextAuthorizingHandler = FlextProtocols.Application.AuthorizingHandler - moved to typings.py
# FlextEventProcessor = FlextProtocols.Application.EventProcessor - moved to typings.py
# FlextUnitOfWork = FlextProtocols.Application.UnitOfWork - moved to typings.py

# Infrastructure layer aliases - commented to avoid conflicts with typings.py
# FlextConnection = FlextProtocols.Infrastructure.Connection - moved to typings.py
FlextAuthProtocol = FlextProtocols.Infrastructure.Auth  # Keep this one as it's specific
FlextConfigurable = FlextProtocols.Infrastructure.Configurable
# FlextLoggerProtocol = FlextProtocols.Infrastructure.LoggerProtocol - moved to typings.py

# Extensions layer aliases - commented to avoid conflicts with typings.py
# FlextPlugin = FlextProtocols.Extensions.Plugin - moved to typings.py
# FlextPluginContext = FlextProtocols.Extensions.PluginContext - moved to typings.py
# FlextMiddleware = FlextProtocols.Extensions.Middleware - moved to typings.py
# FlextAsyncMiddleware = FlextProtocols.Extensions.AsyncMiddleware - moved to typings.py
FlextObservabilityProtocol = FlextProtocols.Extensions.Observability  # Keep this one

# Decorator patterns - commented to avoid conflicts with typings.py
# FlextDecoratedFunction = DecoratedFunction - moved to typings.py

# Legacy aliases for removed protocols (redirected to new hierarchy)
FlextValidationRule = (
    FlextProtocols.Foundation.Validator
)  # Simplified to base validator
FlextMetricsCollector = (
    FlextProtocols.Extensions.Observability
)  # Metrics are part of observability
FlextAsyncHandler = FlextProtocols.Application.Handler  # Unified with regular handler
FlextAsyncService = FlextProtocols.Domain.Service  # Unified with regular service

# Typo fixes
FlextAuthProtocols = FlextProtocols.Infrastructure.Auth  # Fix legacy typo

# Additional legacy support
FlextEventPublisher = (
    FlextProtocols.Domain.EventStore
)  # Publisher is part of event store
FlextEventSubscriber = (
    FlextProtocols.Application.EventProcessor
)  # Subscriber is event processor
FlextEventStreamReader = (
    FlextProtocols.Domain.EventStore
)  # Stream reader is part of event store
FlextProjectionBuilder = (
    FlextProtocols.Application.EventProcessor
)  # Projection builder is event processor

# Observability sub-protocols (consolidated into main observability)
FlextSpanProtocol = FlextProtocols.Extensions.Observability
FlextTracerProtocol = FlextProtocols.Extensions.Observability
FlextMetricsProtocol = FlextProtocols.Extensions.Observability
FlextAlertsProtocol = FlextProtocols.Extensions.Observability

# Plugin system legacy aliases
FlextPluginRegistry = (
    FlextProtocols.Extensions.PluginContext
)  # Registry is part of context
FlextPluginLoader = FlextProtocols.Extensions.PluginContext  # Loader is part of context


# =============================================================================
# EXPORTS - Hierarchical and legacy protocols
# =============================================================================

__all__: list[str] = [
    "FlextProtocols",  # ONLY main class exported
]
