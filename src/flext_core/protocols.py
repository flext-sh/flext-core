"""Protocol definitions and interface contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, runtime_checkable

from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T_contra, TInput_contra, TOutput_co

if TYPE_CHECKING:
    from flext_core.config import FlextConfig


class FlextProtocols:
    """Hierarchical protocol architecture with composition patterns."""

    # =========================================================================
    # FOUNDATION LAYER - Core building blocks
    # =========================================================================

    class Foundation:
        """Foundation layer protocols - core building blocks."""

        class Validator(Protocol, Generic[T_contra]):
            """Generic validator protocol."""

            def validate(self, data: T_contra) -> object:
                """Validate input data and return status."""
                ...

    # =========================================================================
    # DOMAIN LAYER - Business logic protocols
    # =========================================================================

    class Domain:
        """Domain layer protocols - business logic."""

        # Domain protocols providing service and repository patterns

        class Service(Protocol):
            """Domain service protocol with lifecycle management."""

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

        @runtime_checkable
        class ConfigValidator(Protocol):
            """Protocol for configuration validation strategies."""

            def validate_runtime_requirements(self) -> FlextResult[None]:
                """Validate configuration meets runtime requirements."""
                ...

            def validate_business_rules(self) -> FlextResult[None]:
                """Validate business rules for configuration consistency."""
                ...

        @runtime_checkable
        class ConfigPersistence(Protocol):
            """Protocol for configuration persistence operations.

            Follows Single Responsibility Principle - only handles persistence.
            """

            def save_to_file(
                self, file_path: str | Path, **kwargs: object
            ) -> FlextResult[None]:
                """Save configuration to file."""
                ...

            @classmethod
            def load_from_file(cls, file_path: str | Path) -> FlextResult[FlextConfig]:
                """Load configuration from file."""
                ...

        @runtime_checkable
        class ConfigFactory(Protocol):
            """Protocol for configuration factory methods.

            Follows Open/Closed Principle - extensible for new configuration types.
            """

            @classmethod
            def create_web_service_config(
                cls, **kwargs: object
            ) -> FlextResult[FlextConfig]:
                """Create web service configuration."""
                ...

            @classmethod
            def create_microservice_config(
                cls, **kwargs: object
            ) -> FlextResult[FlextConfig]:
                """Create microservice configuration."""
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

    class Commands:
        """CQRS Command and Query protocols for Flext CQRS components."""

        class CommandHandler[CommandT, ResultT](Protocol):
            """Protocol for command handlers in CQRS pattern."""

            def handle(self, command: CommandT) -> ResultT:
                """Handle a command and return result.

                Args:
                    command: The command to handle

                Returns:
                    The result of handling the command

                """
                ...

            def can_handle(self, command_type: type) -> bool:
                """Check if this handler can process the given command type.

                Args:
                    command_type: The type of command to check

                Returns:
                    True if this handler can process the command type

                """
                ...

        class QueryHandler[QueryT, ResultT](Protocol):
            """Protocol for query handlers in CQRS pattern."""

            def handle(self, query: QueryT) -> ResultT:
                """Handle a query and return result.

                Args:
                    query: The query to handle

                Returns:
                    The result of handling the query

                """
                ...

        class CommandBus(Protocol):
            """Protocol for command bus routing and execution."""

            def register_handler(self, handler: object) -> None:
                """Register a command handler.

                Args:
                    handler: The handler to register

                """
                ...

            def execute(self, command: object) -> object:
                """Execute a command through registered handlers.

                Args:
                    command: The command to execute

                Returns:
                    The result of command execution

                """
                ...

        class Middleware(Protocol):
            """Protocol for command bus middleware."""

            def process(self, command: object, handler: object) -> object:
                """Process command through middleware.

                Args:
                    command: The command being processed
                    handler: The handler that will process the command

                Returns:
                    The result of middleware processing

                """
                ...


__all__ = [
    "FlextProtocols",  # Main hierarchical protocol architecture with Config
]
