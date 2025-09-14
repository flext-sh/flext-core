"""Protocol definitions and interface contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import Generic, Protocol, TypeVar, runtime_checkable

from flext_core.typings import FlextTypes, T_co, T_contra

# Type variables for generic protocols with correct variance
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TOutput_co = TypeVar("TOutput_co", covariant=True)


class FlextProtocols:
    """Hierarchical protocol architecture with composition patterns."""

    # Protocol system with 618 lines across 5 hierarchical layers and 40+ protocols
    # Protocol definitions with hierarchical organization
    # Provides protocol coverage for various use cases

    # =========================================================================
    # FOUNDATION LAYER - Core building blocks
    # =========================================================================

    class Foundation:
        """Foundation layer protocols - core building blocks."""

        # Foundation protocols providing 15+ core building blocks

        class Callable(Protocol, Generic[T_co]):
            """Generic callable protocol with type safety."""

            # Generic callable protocol with enhanced type safety

            def __call__(self, *args: object, **kwargs: object) -> T_co:
                """Execute callable with arguments."""
                ...

        @runtime_checkable
        class DecoratedCallable(Protocol, Generic[T_co]):
            """Callable protocol with function attributes for decorators."""

            def __call__(self, *args: object, **kwargs: object) -> T_co:
                """Execute callable with arguments."""
                ...

            __name__: str
            __module__: str
            __doc__: str | None
            __qualname__: str
            __annotations__: FlextTypes.Core.Dict
            __dict__: FlextTypes.Core.Dict
            __wrapped__: object | None  # Can be any callable

        class SupportsRichComparison(Protocol):
            """Protocol for objects supporting rich comparison."""

            # Comparison protocol with enhanced ordering capabilities
            # This defines 7 methods for what Python provides automatically

            def __lt__(self, other: object) -> bool:
                """Less than comparison."""
                ...

            def __le__(self, other: object) -> bool:
                """Less than or equal comparison."""
                ...

            def __gt__(self, other: object) -> bool:
                """Greater than comparison."""
                ...

            def __ge__(self, other: object) -> bool:
                """Greater than or equal comparison."""
                ...

            def __eq__(self, other: object) -> bool:
                """Equality comparison."""
                ...

            def __ne__(self, other: object) -> bool:
                """Not equal comparison."""
                ...

            def __hash__(self) -> int:
                """Hash support for rich comparison objects."""
                ...

        class Validator(Protocol, Generic[T_contra]):
            """Generic validator protocol."""

            def validate(self, data: T_contra) -> object:
                """Validate input data and return status."""
                ...

        class ErrorHandler(Protocol):
            """Error handler protocol."""

            def handle_error(self, error: Exception) -> str:
                """Transform exception to error message."""
                ...

        class Factory(Protocol, Generic[T_co]):
            """Type-safe factory protocol."""

            def create(self, **kwargs: object) -> T_co:
                """Create instance of type T."""
                ...

        class AsyncFactory(Protocol, Generic[T_co]):
            """Async factory protocol."""

            async def create_async(self, **kwargs: object) -> T_co:
                """Create instance asynchronously."""
                ...

        @runtime_checkable
        class HasToDictBasic(Protocol):
            """Protocol for objects exposing to_dict_basic."""

            def to_dict_basic(
                self,
            ) -> FlextTypes.Core.Dict:  # pragma: no cover - typing helper
                """Convert object to basic dictionary."""
                ...

        @runtime_checkable
        class HasToDict(Protocol):
            """Protocol for objects exposing to_dict."""

            def to_dict(
                self,
            ) -> FlextTypes.Core.Dict:  # pragma: no cover - typing helper
                """Convert object to dictionary."""
                ...

        @runtime_checkable
        class SupportsDynamicAttributes(Protocol):
            """Protocol for objects that support dynamic attribute setting.

            This protocol allows mixins to set arbitrary attributes on objects
            without triggering MyPy errors for missing attributes.
            """

            def __setattr__(self, name: str, value: object, /) -> None:
                """Set attribute on object."""
                ...

            def __getattribute__(self, name: str, /) -> object:
                """Get attribute from object."""
                ...

        @runtime_checkable
        class HasModelDump(Protocol):
            """Protocol for Pydantic v2 models with model_dump method."""

            def model_dump(self) -> FlextTypes.Core.Dict:
                """Convert model to dictionary (Pydantic v2 style)."""
                ...

        @runtime_checkable
        class HasDict(Protocol):
            """Protocol for Pydantic v1 models with dict method."""

            def dict(self) -> FlextTypes.Core.Dict:
                """Convert model to dictionary (Pydantic v1 style)."""
                ...

        @runtime_checkable
        class HasModelValidate(Protocol):
            """Protocol for Pydantic v2 models with model_validate class method."""

            @classmethod
            def model_validate(cls, obj: object) -> object:
                """Validate and create model instance from object data."""
                ...

        @runtime_checkable
        class DataConstructor(Protocol):
            """Protocol for classes that can be constructed from data."""

            def __call__(self, data: object) -> object:
                """Construct instance from data object."""
                ...

        @runtime_checkable
        class SizedDict(Protocol):
            """Protocol for dict-like objects that support len()."""

            def __len__(self) -> int:
                """Return length of dict."""
                ...

        @runtime_checkable
        class SizedList(Protocol):
            """Protocol for list-like objects that support len()."""

            def __len__(self) -> int:
                """Return length of list."""
                ...

    # =========================================================================
    # DOMAIN LAYER - Business logic protocols
    # =========================================================================

    class Domain:
        """Domain layer protocols - business logic."""

        # Domain protocols providing 6 specialized patterns for business logic

        class Service(Protocol):
            """Domain service protocol with lifecycle management."""

            # Service lifecycle management with start/stop/health_check capabilities
            # Provides service management patterns

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for service."""
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

        class Repository(Protocol, Generic[T_contra]):
            """Repository protocol for data access."""

            @abstractmethod
            def get_by_id(self, entity_id: str) -> object:
                """Get entity by ID."""
                ...

            @abstractmethod
            def save(self, entity: T_contra) -> object:
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
            """Domain event protocol."""

            event_id: str
            event_type: str
            aggregate_id: str
            event_version: int
            timestamp: str

            def to_dict(self) -> FlextTypes.Core.Dict:
                """Convert event to dictionary."""
                ...

            @classmethod
            def from_dict(
                cls,
                data: FlextTypes.Core.Dict,
            ) -> FlextProtocols.Domain.DomainEvent:
                """Create event from dictionary."""
                ...

        class EventStore(Protocol):
            """Event store protocol."""

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
        """Application layer protocols - use cases and handlers."""

        class Handler(Protocol, Generic[TInput_contra, TOutput_co]):
            """Application handler with validation."""

            def __call__(self, input_data: TInput_contra) -> object:
                """Process input and return output."""
                ...

            def validate(self, data: TInput_contra) -> object:
                """Validate input before processing."""
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
                context: FlextTypes.Core.Dict,
            ) -> object:
                """Check authorization for message processing."""
                ...

        class EventProcessor(Protocol):
            """Event processor for domain event handling."""

            def process_event(self, event: FlextTypes.Core.Dict) -> object:
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
        """Infrastructure layer protocols - external systems."""

        class Connection(Protocol):
            """Connection protocol for external systems."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for connection."""
                ...

            def test_connection(self) -> object:
                """Test connection to external system."""
                ...

            def get_connection_string(self) -> str:
                """Get connection string for external system."""
                ...

            def close_connection(self) -> object:
                """Close connection to external system."""
                ...

        @runtime_checkable
        class Configurable(Protocol):
            """Configurable component protocol."""

            def configure(self, config: FlextTypes.Core.Dict) -> object:
                """Configure component with provided settings."""
                ...

            def get_config(self) -> FlextTypes.Core.Dict:
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
        """Extensions layer protocols - plugins and extension patterns."""

        # Plugin architecture and middleware system for extensible applications
        # Provides plugin ecosystem support for applications

        class Plugin(Protocol):
            """Plugin protocol with configuration."""

            # Plugin lifecycle management with configuration and initialization
            # Supports complex plugin ecosystems with full lifecycle control

            def configure(self, config: FlextTypes.Core.Dict) -> object:
                """Configure component with settings."""
                ...

            def get_config(self) -> FlextTypes.Core.Dict:
                """Get current configuration."""
                ...

            @abstractmethod
            def initialize(
                self,
                context: FlextProtocols.Extensions.PluginContext,
            ) -> object:
                """Initialize plugin."""
                ...

            @abstractmethod
            def shutdown(self) -> object:
                """Shutdown plugin and cleanup."""
                ...

            @abstractmethod
            def get_info(self) -> FlextTypes.Core.Dict:
                """Get plugin information."""
                ...

        class PluginContext(Protocol):
            """Plugin execution context."""

            def get_service(self, service_name: str) -> object:
                """Get service by name."""
                ...

            def get_config(self) -> FlextTypes.Core.Dict:
                """Get plugin configuration."""
                ...

            def flext_logger(self) -> FlextProtocols.Infrastructure.LoggerProtocol:
                """Get logger instance for plugin."""
                ...

        class Middleware(Protocol):
            """Middleware pipeline component protocol."""

            def process(
                self,
                request: object,
                _next_handler: Callable[[object], object],
            ) -> object:
                """Process request with middleware logic."""
                ...

        class AsyncMiddleware(Protocol):
            """Async middleware component protocol."""

            async def process_async(
                self,
                request: object,
                _next_handler: Callable[[object], Awaitable[object]],
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
                _tags: FlextTypes.Core.Headers | None = None,
            ) -> object:
                """Record metric value."""
                ...

            def start_trace(self, operation_name: str) -> object:
                """Start distributed trace."""
                ...

            def health_check(self) -> object:
                """Perform health check."""
                ...

    # =============================================================================
    # DECORATOR PROTOCOLS - Special function patterns
    # =============================================================================

    class DecoratedFunction(Protocol, Generic[T_co]):
        """Decorated function protocol returning FlextResult for railway-oriented programming."""

        def __call__(self, *args: object, **kwargs: object) -> T_co:
            """Execute the decorated function returning FlextResult."""
            ...


__all__: FlextTypes.Core.StringList = [
    "FlextProtocols",  # Main hierarchical protocol architecture with Config
]
