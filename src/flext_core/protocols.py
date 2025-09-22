"""Protocol definitions codifying the FLEXT-Core 1.0.0 contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, overload, runtime_checkable

from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T_contra, TInput_contra, TOutput_co

if TYPE_CHECKING:
    from flext_core.config import FlextConfig


class FlextProtocols:
    """Grouped protocol interfaces underpinning the modernization contracts.

    They clarify the callable semantics, configuration hooks, and extension
    points relied upon during the 1.0.0 rollout.
    """

    # =========================================================================
    # FOUNDATION LAYER - Core building blocks
    # =========================================================================

    class Foundation:
        """Foundation layer protocols cementing the 1.0.0 contracts."""

        class Validator(Protocol, Generic[T_contra]):
            """Generic validator protocol reused by modernization guardrails."""

            def validate(self, data: T_contra) -> object:
                """Validate input data according to the shared release policy."""
                ...

    # =========================================================================
    # DOMAIN LAYER - Business logic protocols
    # =========================================================================

    class Domain:
        """Domain layer protocols reflecting FLEXT's modernization DDD usage."""

        # Domain protocols providing service and repository patterns

        @runtime_checkable
        class Service(Protocol):
            """Domain service contract aligned with FlextDomainService implementation."""

            @abstractmethod
            def execute(self) -> FlextResult[object]:
                """Execute the main domain operation.

                Returns:
                    FlextResult[object]: Success with domain result or failure with error

                """
                ...

            def is_valid(self) -> bool:
                """Check if the domain service is in a valid state.

                Returns:
                    bool: True if valid, False otherwise

                """
                ...

            def validate_business_rules(self) -> FlextResult[None]:
                """Validate business rules for the domain service.

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                ...

            def validate_config(self) -> FlextResult[None]:
                """Validate service configuration.

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                ...

            def execute_operation(self, operation: object) -> FlextResult[object]:
                """Execute operation using OperationExecutionRequest model.

                Args:
                    operation: OperationExecutionRequest containing operation settings

                Returns:
                    FlextResult[object]: Success with result or failure with error

                """
                ...

            def get_service_info(self) -> FlextTypes.Core.Dict:
                """Get service information and metadata.

                Returns:
                    FlextTypes.Core.Dict: Service information dictionary

                """
                ...

        class Repository(Protocol, Generic[T_contra]):
            """Repository protocol shaping modernization data access patterns."""

            @abstractmethod
            def get_by_id(self, entity_id: str) -> object:
                """Retrieve an aggregate using the standardized identity lookup."""
                ...

            @abstractmethod
            def save(self, entity: T_contra) -> object:
                """Persist an entity following modernization consistency rules."""
                ...

            @abstractmethod
            def delete(self, entity_id: str) -> object:
                """Delete an entity while respecting modernization invariants."""
                ...

            @abstractmethod
            def find_all(self) -> object:
                """Enumerate entities for modernization-aligned queries."""
                ...

    # =========================================================================
    # APPLICATION LAYER - Use cases and handlers
    # =========================================================================

    class Application:
        """Application layer protocols - use cases and handlers."""

        @runtime_checkable
        class Handler(Protocol, Generic[TInput_contra, TOutput_co]):
            """Application handler protocol aligned with FlextHandlers implementation."""

            @abstractmethod
            def handle(self, message: TInput_contra) -> FlextResult[TOutput_co]:
                """Handle the message and return result.

                Args:
                    message: The input message to process

                Returns:
                    FlextResult[TOutput_co]: Success with result or failure with error

                """
                ...

            def can_handle(self, message_type: object) -> bool:
                """Check if handler can process this message type.

                Args:
                    message_type: The message type to check

                Returns:
                    bool: True if handler can process the message type, False otherwise

                """
                ...

            def execute(self, message: TInput_contra) -> FlextResult[TOutput_co]:
                """Execute the handler with the given message.

                Args:
                    message: The input message to execute

                Returns:
                    FlextResult[TOutput_co]: Execution result

                """
                ...

            def validate_command(self, command: TInput_contra) -> FlextResult[None]:
                """Validate a command message.

                Args:
                    command: The command to validate

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                ...

            def validate_query(self, query: TInput_contra) -> FlextResult[None]:
                """Validate a query message.

                Args:
                    query: The query to validate

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                ...

            @property
            def handler_name(self) -> str:
                """Get the handler name.

                Returns:
                    str: Handler name

                """
                ...

            @property
            def mode(self) -> str:
                """Get the handler mode (command/query).

                Returns:
                    str: Handler mode

                """
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

            def configure(self, config: FlextTypes.Core.Dict) -> FlextResult[None]:
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
                self,
                file_path: str | Path,
                **kwargs: object,
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
                cls,
                **kwargs: object,
            ) -> FlextResult[FlextConfig]:
                """Create web service configuration."""
                ...

            @classmethod
            def create_microservice_config(
                cls,
                **kwargs: object,
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
            """Plugin protocol with configuration.

            Plugin lifecycle management with configuration and initialization
            Supports complex plugin ecosystems with full lifecycle control
            """

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

            def handle(self, command: CommandT) -> FlextResult[ResultT]:
                """Handle a command and return a :class:`FlextResult` wrapper.

                Args:
                    command: The command to handle

                Returns:
                    FlextResult containing the command handling outcome

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

            def handle(self, query: QueryT) -> FlextResult[ResultT]:
                """Handle a query and return a :class:`FlextResult` wrapper.

                Args:
                    query: The query to handle

                Returns:
                    FlextResult containing the query handling outcome

                """
                ...

        @runtime_checkable
        class CommandBus(Protocol):
            """Protocol for command bus routing and execution."""

            @overload
            def register_handler(self, handler: Callable, /) -> FlextResult[None]: ...

            @overload
            def register_handler(
                self,
                command_type: type,
                handler: Callable,
                /,
            ) -> FlextResult[None]: ...

            def register_handler(self, *args: object) -> FlextResult[None]:
                """Register a command handler using one of two supported signatures.

                The command bus accepts both ``register_handler(handler)`` for
                auto-discoverable handlers and
                ``register_handler(command_type, handler)`` when explicitly
                binding a handler to a message type.

                Args:
                    *args: Positional arguments matching one of the supported
                        registration signatures.

                Returns:
                    FlextResult[None]: Outcome of the registration attempt.

                """
                ...

            def unregister_handler(self, command_type: type | str) -> bool:
                """Remove a handler registration by type or name.

                Args:
                    command_type: The command type or name to unregister

                Returns:
                    bool: True if handler was removed, False otherwise

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
