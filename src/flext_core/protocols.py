"""Protocol definitions codifying the FLEXT-Core 1.0.0 contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    overload,
    runtime_checkable,
)

from flext_core.typings import (
    FlextTypes,
    T,
    T_contra,
    TInput_contra,
    TResult,
)

if TYPE_CHECKING:
    from flext_core.models import FlextModels
    from flext_core.result import FlextResult


class FlextProtocols:
    """Grouped protocol interfaces underpinning modernization contracts.

    They clarify the callable semantics, configuration hooks, and
    extension points relied upon during the 1.0.0 rollout. Provides
    type-safe protocol definitions for the entire FLEXT ecosystem.

    # Re-export commonly used foundation protocols for convenience
    HasModelDump = FlextProtocols.Foundation.HasModelDump

    **Function**: Protocol interface definitions for type safety
    "ExceptionProtocol",  # Protocol for exception types (breaks circular dependency)
    "FlextProtocols",  # Main hierarchical protocol architecture with Config
    "HasModelDump",  # Convenience alias for foundation dump protocol
    "ResultProtocol",  # Protocol for result types (breaks circular dependency)
        - Extension protocols for plugin architecture
        - Protocol registration and validation system
        - Circuit breaker and rate limiting for protocols
        - Middleware support for protocol processing
        - Performance metrics and audit logging
        - Batch and parallel protocol validation

    **Uses**: Core FLEXT infrastructure for protocols
        - FlextResult[T] for all operation results (lazy loaded)
        - FlextTypes for type definitions and aliases
        - FlextConfig for configuration management (lazy loaded)
        - typing.Protocol for runtime-checkable protocols
        - typing.Generic for generic protocol types
        - abc.abstractmethod for abstract protocol methods
        - time module for rate limiting and metrics
        - datetime for timestamp operations
        - pathlib for file operations
        - collections.abc for callable protocols

    **How to use**: Protocol definition and validation
        ```python
        from flext_core import FlextProtocols, FlextResult


        # Example 1: Use Foundation validator protocol
        class EmailValidator(FlextProtocols.Foundation.Validator[str]):
            def validate(self, data: str) -> object:
                if "@" not in data:
                    return FlextResult[None].fail("Invalid email")
                return FlextResult[None].ok(None)


        # Example 2: Use Domain service protocol
        class UserService(FlextProtocols.Domain.Service):
            def execute(self) -> FlextResult[object]:
                return FlextResult[object].ok({"status": "success"})

            def is_valid(self) -> bool:
                return True


        # Example 3: Use Application handler protocol
        class CreateUserHandler(FlextProtocols.Application.Handler[dict, str]):
            def handle(self, message: dict) -> FlextResult[str]:
                return FlextResult[str].ok("user_created")


        # Example 4: Register protocol for validation
        protocols = FlextProtocols()
        reg_result = protocols.register("user_service", UserService)
        if reg_result.is_success:
            # Validate implementation
            validation = protocols.validate_implementation("user_service", UserService)


        # Example 5: Use Infrastructure logger protocol
        class CustomLogger(FlextProtocols.Infrastructure.LoggerProtocol):
            def info(self, message: str, **kwargs: object) -> None:
                print(f"INFO: {message}")


        # Example 6: Use Extension plugin protocol
        class CustomPlugin(FlextProtocols.Extensions.Plugin):
            def initialize(self, context) -> object:
                return FlextResult[None].ok(None)


        # Example 7: Batch protocol validation
        implementations = [UserService, CreateUserHandler]
        batch_result = protocols.validate_batch("services", implementations)
        ```

        - [ ] Add distributed protocol validation across services
        - [ ] Implement protocol versioning for evolution
        - [ ] Support protocol composition and inheritance
        - [ ] Add protocol discovery and introspection
        - [ ] Implement protocol compatibility checking
        - [ ] Support protocol migration tools
        - [ ] Add protocol documentation generation
        - [ ] Implement protocol testing framework
        - [ ] Support protocol performance profiling
        - [ ] Add protocol security validation

    Attributes:
        Foundation: Foundation layer protocol definitions.
        Domain: Domain layer protocol definitions for DDD.
        Application: Application layer protocol definitions.
        Infrastructure: Infrastructure protocol definitions.
        Extensions: Extension and plugin protocol definitions.
        Commands: CQRS command and query protocol definitions.

    Note:
        All protocols use @runtime_checkable for isinstance checks.
        Protocol registration enables validation and type checking.
        Circuit breaker and rate limiting protect validation.
        Middleware can transform protocol implementations.
        Metrics and audit logs track protocol usage patterns.

    Warning:
        Protocol validation has rate limiting (10 per 60 seconds).
        Circuit breaker opens after 5 consecutive failures.
        Cache TTL defaults to 300 seconds for validation.
        Batch validation fails if any implementation fails.

    Example:
        Complete protocol definition and validation workflow:

        >>> protocols = FlextProtocols()
        >>> protocols.register("validator", EmailValidator)
        >>> result = protocols.validate_implementation("validator", EmailValidator)
        >>> print(result.is_success)
        True

    See Also:
        FlextHandlers: For handler implementation patterns.
        FlextBus: For command/query bus implementation.
        FlextConfig: For configuration management.
        FlextResult: For result type definitions.

    """

    # =========================================================================
    # FOUNDATION LAYER - Core building blocks
    # =========================================================================

    class Foundation:
        """Foundation layer protocols cementing the 1.0.0 contracts."""

        @runtime_checkable
        class OperationCallable(Protocol):
            """Protocol for callable operations in the FLEXT ecosystem.

            This protocol defines the interface for operations that can be executed
            within the FLEXT framework, ensuring type safety and consistent behavior.
            """

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Execute the operation with given arguments.

                Args:
                    *args: Positional arguments for the operation
                    **kwargs: Keyword arguments for the operation

                Returns:
                    The result of the operation execution

                """
                ...

        @runtime_checkable
        class Validator(Protocol, Generic[T_contra]):
            """Generic validator protocol reused by modernization guardrails."""

            def validate(self, data: T_contra) -> object:
                """Validate input data according to the shared release policy."""
                ...

        @runtime_checkable
        class HasModelDump(Protocol):
            """Protocol for objects that have model_dump method.

            Supports Pydantic's model_dump signature with optional mode parameter.
            """

            def model_dump(self, mode: str = "python") -> FlextTypes.Dict:
                """Dump the model to a dictionary.

                Args:
                    mode: Serialization mode ('python' or 'json')

                Returns:
                    Dictionary representation of the model

                """
                ...

        @runtime_checkable
        class HasModelFields(Protocol):
            """Protocol for objects that have model_fields attribute.

            Consolidated from mixins.py for centralized protocol management.
            """

            model_fields: FlextTypes.Dict

            def model_dump(self, **kwargs: object) -> FlextTypes.Dict:
                """Dump model to dictionary (Pydantic compatibility)."""
                ...

        @runtime_checkable
        class HasValue(Protocol):
            """Protocol for enum-like objects with a value attribute.

            Consolidated from loggings.py for centralized protocol management.
            """

            value: object

        @runtime_checkable
        class HasResultValue(Protocol):
            """Protocol for FlextResult-like objects with value and is_success attributes.

            Consolidated from processors.py for centralized protocol management.
            """

            value: object
            is_success: bool

        @runtime_checkable
        class HasTimestamps(Protocol):
            """Protocol for objects with created_at and updated_at timestamps.

            Consolidated from service.py for centralized protocol management.
            """

            created_at: object
            updated_at: object

        @runtime_checkable
        class HasHandlerType(Protocol):
            """Protocol for config objects with handler_type attribute.

            Consolidated from config.py for centralized protocol management.
            """

            handler_type: str | None

        @runtime_checkable
        class HasValidateCommand(Protocol):
            """Protocol for commands with validate_command method.

            Consolidated from bus.py for centralized protocol management.
            """

            def validate_command(self) -> FlextResult[None]:
                """Validate command and return FlextResult."""
                ...

    # =========================================================================
    # DOMAIN LAYER - Business logic protocols
    # =========================================================================

    class Domain:
        """Domain layer protocols reflecting FLEXT's modernization DDD usage."""

        # Domain protocols providing service and repository patterns

        @runtime_checkable
        class Service(Protocol[T]):
            """Domain service contract aligned with FlextService implementation."""

            @abstractmethod
            def execute(self) -> FlextResult[T]:
                """Execute the main domain operation.

                Returns:
                    FlextResult[T_co]: Success with domain result or failure with error

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

            def execute_operation(
                self, operation: FlextModels.OperationExecutionRequest
            ) -> FlextResult[object]:
                """Execute operation using OperationExecutionRequest model.

                Args:
                    operation: OperationExecutionRequest containing operation settings

                Returns:
                    FlextResult[T_co]: Success with result or failure with error

                """
                ...

            def get_service_info(self: object) -> FlextTypes.Dict:
                """Get service information and metadata.

                Returns:
                    FlextTypes.Dict: Service information dictionary

                """
                ...

        @runtime_checkable
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
            def find_all(self: object) -> object:
                """Enumerate entities for modernization-aligned queries."""
                ...

    # =========================================================================
    # APPLICATION LAYER - Use cases and handlers
    # =========================================================================

    class Application:
        """Application layer protocols - use cases and handlers."""

        @runtime_checkable
        class Handler(Protocol, Generic[TInput_contra, TResult]):
            """Application handler protocol aligned with FlextHandlers implementation."""

            @abstractmethod
            def handle(self, message: TInput_contra) -> FlextResult[TResult]:
                """Handle the message and return result.

                Args:
                    message: The input message to process

                Returns:
                    FlextResult[TResult]: Success with result or failure with error

                """
                ...

            def __call__(self, input_data: TInput_contra) -> FlextResult[TResult]:
                """Process input and return a ``FlextResult`` containing the output."""
                ...

            def can_handle(self, message_type: object) -> bool:
                """Check if handler can process this message type.

                Args:
                    message_type: The message type to check

                Returns:
                    bool: True if handler can process the message type, False otherwise

                """
                ...

            def execute(self, message: TInput_contra) -> FlextResult[TResult]:
                """Execute the handler with the given message.

                Args:
                    message: The input message to execute

                Returns:
                    FlextResult[TResult]: Execution result

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

            def validate(self, _data: TInput_contra) -> FlextResult[None]:
                """Validate input before processing and wrap the outcome in ``FlextResult``."""
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
            def handler_name(self: object) -> str:
                """Get the handler name.

                Returns:
                    str: Handler name

                """
                ...

            @property
            def mode(self: object) -> str:
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

        @runtime_checkable
        class Connection(Protocol):
            """Connection protocol for external systems."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for connection."""
                ...

            def test_connection(self: object) -> object:
                """Test connection to external system."""
                ...

            def get_connection_string(self: object) -> str:
                """Get connection string for external system."""
                ...

            def close_connection(self: object) -> object:
                """Close connection to external system."""
                ...

        @runtime_checkable
        class Configurable(Protocol):
            """Configurable component protocol."""

            def configure(self, config: FlextTypes.Dict) -> FlextResult[None]:
                """Configure component with provided settings."""
                ...

        @runtime_checkable
        class LoggerProtocol(Protocol):
            """Logger protocol with standard logging methods returning FlextResult."""

            def trace(
                self, message: str, *args: object, **kwargs: object
            ) -> FlextResult[None]:
                """Log trace message."""
                ...

            def debug(
                self, message: str, *args: object, **context: object
            ) -> FlextResult[None]:
                """Log debug message."""
                ...

            def info(
                self, message: str, *args: object, **context: object
            ) -> FlextResult[None]:
                """Log info message."""
                ...

            def warning(
                self, message: str, *args: object, **context: object
            ) -> FlextResult[None]:
                """Log warning message."""
                ...

            def error(
                self, message: str, *args: object, **kwargs: object
            ) -> FlextResult[None]:
                """Log error message."""
                ...

            def critical(
                self, message: str, *args: object, **kwargs: object
            ) -> FlextResult[None]:
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

        @runtime_checkable
        class Plugin(Protocol):
            """Plugin protocol with configuration.

            Plugin lifecycle management with configuration and initialization
            Supports complex plugin ecosystems with full lifecycle control
            """

            def configure(self, config: FlextTypes.Dict) -> object:
                """Configure component with settings."""
                ...

            def get_config(self: object) -> FlextTypes.Dict:
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
            def shutdown(self: object) -> object:
                """Shutdown plugin and cleanup."""
                ...

            @abstractmethod
            def get_info(self: object) -> FlextTypes.Dict:
                """Get plugin information."""
                ...

        @runtime_checkable
        class PluginContext(Protocol):
            """Plugin execution context."""

            def get_service(self, service_name: str) -> object:
                """Get service by name."""
                ...

            def get_config(self: object) -> FlextTypes.Dict:
                """Get plugin configuration."""
                ...

            def flext_logger(
                self: object,
            ) -> FlextProtocols.Infrastructure.LoggerProtocol:
                """Get logger instance for plugin."""
                ...

        @runtime_checkable
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
                _tags: FlextTypes.StringDict | None = None,
            ) -> object:
                """Record metric value."""
                ...

            def start_trace(self, operation_name: str) -> object:
                """Start distributed trace."""
                ...

            def health_check(self: object) -> object:
                """Perform health check."""
                ...

    class Commands:
        """CQRS Command and Query protocols for Flext CQRS components."""

        @runtime_checkable
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

        @runtime_checkable
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
            def register_handler(
                self,
                handler: Callable[[object], object],
                /,
            ) -> FlextResult[None]: ...

            @overload
            def register_handler(
                self,
                command_type: type,
                handler: Callable[[object], object],
                /,
            ) -> FlextResult[None]: ...

            def register_handler(self, *_args: object) -> FlextResult[None]:
                """Register a command handler with the command bus.

                The command bus accepts both ``register_handler(handler)`` for
                automatic type detection and ``register_handler(command_type, handler)``
                for explicit type specification.

                Args:
                    handler: The handler function to register
                    command_type: Optional command type for explicit registration

                Returns:
                    FlextResult indicating success or failure

                """
                # Implementation would go here
                ...

            def unregister_handler(self, command_type: type | str) -> FlextResult[None]:
                """Remove a handler registration by type or name.

                Args:
                    command_type: The command type or name to unregister

                Returns:
                    FlextResult[None]: Success if handler was removed, failure if not found

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

        @runtime_checkable
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
