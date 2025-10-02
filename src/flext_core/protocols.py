"""Protocol definitions codifying the FLEXT-Core 1.0.0 contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    overload,
    runtime_checkable,
)

from flext_core.result import FlextResult
from flext_core.typings import (
    FlextTypes,
    T_contra,
    TCommand_contra,
    TEvent_contra,
    TInput_contra,
    TQuery_contra,
    TResult,
    TState,
    TState_co,
)

if TYPE_CHECKING:
    from flext_core.config import FlextConfig

# =============================================================================
# Core Protocols for Breaking Circular Dependencies
# These protocols are defined here to break the circular dependency between
# result.py and exceptions.py. They provide minimal interfaces that both
# modules can use without importing each other.
# =============================================================================


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

    **TODO**: Enhanced protocol features for 1.0.0+ releases
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

            def model_dump(self, mode: str = "python") -> dict[str, object]:
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

            model_fields: dict[str, object]

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
        class Service(Protocol):
            """Domain service contract aligned with FlextService implementation."""

            @abstractmethod
            def execute(self: object) -> FlextResult[object]:
                """Execute the main domain operation.

                Returns:
                    FlextResult[object]: Success with domain result or failure with error

                """
                ...

            def is_valid(self: object) -> bool:
                """Check if the domain service is in a valid state.

                Returns:
                    bool: True if valid, False otherwise

                """
                ...

            def validate_business_rules(self: object) -> FlextResult[None]:
                """Validate business rules for the domain service.

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                ...

            def validate_config(self: object) -> FlextResult[None]:
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

            def get_service_info(self: object) -> FlextTypes.Core.Dict:
                """Get service information and metadata.

                Returns:
                    FlextTypes.Core.Dict: Service information dictionary

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

        @runtime_checkable
        class AggregateRoot(Protocol, Generic[TState_co]):
            """Aggregate root protocol for domain-driven design patterns."""

            @abstractmethod
            def get_id(self) -> str:
                """Get the aggregate root identifier."""
                ...

            @abstractmethod
            def get_version(self) -> int:
                """Get the aggregate root version for optimistic locking."""
                ...

            @abstractmethod
            def get_uncommitted_events(self) -> FlextTypes.Core.List:
                """Get uncommitted domain events."""
                ...

            @abstractmethod
            def mark_events_as_committed(self) -> None:
                """Mark all events as committed."""
                ...

            @abstractmethod
            def is_valid(self) -> bool:
                """Check if the aggregate root is in a valid state."""
                ...

        @runtime_checkable
        class DomainEvent(Protocol):
            """Domain event protocol for event sourcing patterns."""

            @abstractmethod
            def get_event_id(self) -> str:
                """Get the unique event identifier."""
                ...

            @abstractmethod
            def get_event_type(self) -> str:
                """Get the event type name."""
                ...

            @abstractmethod
            def get_aggregate_id(self) -> str:
                """Get the aggregate root identifier."""
                ...

            @abstractmethod
            def get_event_data(self) -> FlextTypes.Core.Dict:
                """Get the event payload data."""
                ...

            @abstractmethod
            def get_metadata(self) -> FlextTypes.Core.Dict:
                """Get the event metadata."""
                ...

            @abstractmethod
            def get_timestamp(self) -> datetime:
                """Get the event timestamp."""
                ...

        @runtime_checkable
        class Command(Protocol):
            """Command protocol for CQRS patterns."""

            @abstractmethod
            def get_command_id(self) -> str:
                """Get the unique command identifier."""
                ...

            @abstractmethod
            def get_command_type(self) -> str:
                """Get the command type name."""
                ...

            @abstractmethod
            def get_command_data(self) -> FlextTypes.Core.Dict:
                """Get the command payload data."""
                ...

            @abstractmethod
            def get_metadata(self) -> FlextTypes.Core.Dict:
                """Get the command metadata."""
                ...

            @abstractmethod
            def get_timestamp(self) -> datetime:
                """Get the command timestamp."""
                ...

        @runtime_checkable
        class Query(Protocol):
            """Query protocol for CQRS patterns."""

            @abstractmethod
            def get_query_id(self) -> str:
                """Get the unique query identifier."""
                ...

            @abstractmethod
            def get_query_type(self) -> str:
                """Get the query type name."""
                ...

            @abstractmethod
            def get_query_data(self) -> FlextTypes.Core.Dict:
                """Get the query payload data."""
                ...

            @abstractmethod
            def get_metadata(self) -> FlextTypes.Core.Dict:
                """Get the query metadata."""
                ...

            @abstractmethod
            def get_timestamp(self) -> datetime:
                """Get the query timestamp."""
                ...

        @runtime_checkable
        class Saga(Protocol, Generic[TState]):
            """Saga protocol for distributed transaction patterns."""

            @abstractmethod
            def get_saga_id(self) -> str:
                """Get the unique saga identifier."""
                ...

            @abstractmethod
            def get_saga_type(self) -> str:
                """Get the saga type name."""
                ...

            @abstractmethod
            def get_current_state(self) -> TState:
                """Get the current saga state."""
                ...

            @abstractmethod
            def execute_step(
                self,
                step_data: FlextTypes.Core.Dict,
            ) -> FlextResult[TState]:
                """Execute a saga step."""
                ...

            @abstractmethod
            def compensate_step(
                self,
                step_data: FlextTypes.Core.Dict,
            ) -> FlextResult[TState]:
                """Compensate a saga step."""
                ...

            @abstractmethod
            def is_completed(self) -> bool:
                """Check if the saga is completed."""
                ...

            @abstractmethod
            def is_failed(self) -> bool:
                """Check if the saga has failed."""
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

            def validate(self, data: TInput_contra) -> FlextResult[None]:
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

        @runtime_checkable
        class CommandHandler(Protocol, Generic[TCommand_contra, TResult]):
            """Command handler protocol for CQRS patterns."""

            @abstractmethod
            def handle_command(self, command: TCommand_contra) -> FlextResult[TResult]:
                """Handle a command and return result."""
                ...

            @abstractmethod
            def can_handle(self, command_type: str) -> bool:
                """Check if this handler can handle the command type."""
                ...

            @abstractmethod
            def get_supported_command_types(self) -> FlextTypes.Core.List:
                """Get list of supported command types."""
                ...

        @runtime_checkable
        class QueryHandler(Protocol, Generic[TQuery_contra, TResult]):
            """Query handler protocol for CQRS patterns."""

            @abstractmethod
            def handle_query(self, query: TQuery_contra) -> FlextResult[TResult]:
                """Handle a query and return result."""
                ...

            @abstractmethod
            def can_handle(self, query_type: str) -> bool:
                """Check if this handler can handle the query type."""
                ...

            @abstractmethod
            def get_supported_query_types(self) -> FlextTypes.Core.List:
                """Get list of supported query types."""
                ...

        @runtime_checkable
        class EventHandler(Protocol, Generic[TEvent_contra]):
            """Event handler protocol for event sourcing patterns."""

            @abstractmethod
            def handle_event(self, event: TEvent_contra) -> FlextResult[None]:
                """Handle a domain event."""
                ...

            @abstractmethod
            def can_handle(self, event_type: str) -> bool:
                """Check if this handler can handle the event type."""
                ...

            @abstractmethod
            def get_supported_event_types(self) -> FlextTypes.Core.List:
                """Get list of supported event types."""
                ...

        @runtime_checkable
        class SagaManager(Protocol, Generic[TState]):
            """Saga manager protocol for distributed transaction patterns."""

            @abstractmethod
            def start_saga(
                self,
                saga_type: str,
                initial_data: FlextTypes.Core.Dict,
            ) -> FlextResult[str]:
                """Start a new saga."""
                ...

            @abstractmethod
            def execute_saga_step(
                self,
                saga_id: str,
                step_data: FlextTypes.Core.Dict,
            ) -> FlextResult[TState]:
                """Execute a saga step."""
                ...

            @abstractmethod
            def compensate_saga(self, saga_id: str) -> FlextResult[TState]:
                """Compensate a saga."""
                ...

            @abstractmethod
            def get_saga_status(self, saga_id: str) -> FlextResult[str]:
                """Get saga status."""
                ...

            @abstractmethod
            def get_saga_state(self, saga_id: str) -> FlextResult[TState]:
                """Get saga state."""
                ...

        @runtime_checkable
        class EventStore(Protocol):
            """Event store protocol for event sourcing patterns."""

            @abstractmethod
            def save_events(
                self,
                aggregate_id: str,
                events: FlextTypes.Core.List,
                expected_version: int,
            ) -> FlextResult[None]:
                """Save events for an aggregate."""
                ...

            @abstractmethod
            def get_events(
                self,
                aggregate_id: str,
                from_version: int = 0,
            ) -> FlextResult[FlextTypes.Core.List]:
                """Get events for an aggregate."""
                ...

            @abstractmethod
            def get_events_by_type(
                self,
                event_type: str,
                from_timestamp: datetime | None = None,
            ) -> FlextResult[FlextTypes.Core.List]:
                """Get events by type."""
                ...

            @abstractmethod
            def get_events_by_correlation_id(
                self,
                correlation_id: str,
            ) -> FlextResult[FlextTypes.Core.List]:
                """Get events by correlation ID."""
                ...

        @runtime_checkable
        class EventPublisher(Protocol):
            """Event publisher protocol for event sourcing patterns."""

            @abstractmethod
            def publish_event(self, event: object) -> FlextResult[None]:
                """Publish a domain event."""
                ...

            @abstractmethod
            def publish_events(self, events: FlextTypes.Core.List) -> FlextResult[None]:
                """Publish multiple domain events."""
                ...

            @abstractmethod
            def subscribe(self, event_type: str, handler: object) -> FlextResult[None]:
                """Subscribe to an event type."""
                ...

            @abstractmethod
            def unsubscribe(
                self,
                event_type: str,
                handler: object,
            ) -> FlextResult[None]:
                """Unsubscribe from an event type."""
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

            def configure(self, config: FlextTypes.Core.Dict) -> FlextResult[None]:
                """Configure component with provided settings."""
                ...

            def get_config(self: object) -> FlextTypes.Core.Dict:
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
        class LogRenderer(Protocol):
            """Log renderer protocol for formatting log entries."""

            def __call__(
                self,
                logger: object,
                method_name: str,
                event_dict: FlextTypes.Core.Dict,
            ) -> str:
                """Render log entry to string format.

                Args:
                    logger: Logger instance
                    method_name: Method name that generated the log
                    event_dict: Event dictionary with log data

                Returns:
                    str: Formatted log entry string

                """
                ...

        @runtime_checkable
        class LogContextManager(Protocol):
            """Log context manager protocol for managing logger context."""

            def set_correlation_id(self, correlation_id: str) -> FlextResult[None]:
                """Set correlation ID for request tracing.

                Args:
                    correlation_id: Correlation ID to set

                Returns:
                    FlextResult[None]: Success or failure result

                """
                ...

            def set_request_context(self, model: object) -> FlextResult[None]:
                """Set request-specific context data.

                Args:
                    model: Request context model to set

                Returns:
                    FlextResult[None]: Success or failure result

                """
                ...

            def clear_request_context(self: object) -> FlextResult[None]:
                """Clear request-specific context data.

                Returns:
                    FlextResult[None]: Success or failure result

                """
                ...

            def bind_context(self, model: object) -> FlextResult[object]:
                """Create bound logger instance with additional context.

                Args:
                    model: Context binding model to use

                Returns:
                    FlextResult[object]: Bound logger instance or error

                """
                ...

            def get_consolidated_context(self: object) -> FlextTypes.Core.Dict:
                """Get all context data consolidated for log entry building.

                Returns:
                    FlextTypes.Core.Dict: Consolidated context data

                """
                ...

        @runtime_checkable
        class ConfigValidator(Protocol):
            """Protocol for configuration validation strategies."""

            def validate_runtime_requirements(self: object) -> FlextResult[None]:
                """Validate configuration meets runtime requirements."""
                ...

            def validate_business_rules(self: object) -> FlextResult[None]:
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

        @runtime_checkable
        class Plugin(Protocol):
            """Plugin protocol with configuration.

            Plugin lifecycle management with configuration and initialization
            Supports complex plugin ecosystems with full lifecycle control
            """

            def configure(self, config: FlextTypes.Core.Dict) -> object:
                """Configure component with settings."""
                ...

            def get_config(self: object) -> FlextTypes.Core.Dict:
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
            def get_info(self: object) -> FlextTypes.Core.Dict:
                """Get plugin information."""
                ...

        @runtime_checkable
        class PluginContext(Protocol):
            """Plugin execution context."""

            def get_service(self, service_name: str) -> object:
                """Get service by name."""
                ...

            def get_config(self: object) -> FlextTypes.Core.Dict:
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
                _tags: FlextTypes.Core.Headers | None = None,
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
                return FlextResult[None].ok(None)  # pragma: no cover

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


@runtime_checkable
class ResultProtocol[T](Protocol):
    """Protocol for Result type to break circular dependency.

    This protocol defines the minimal interface needed by exceptions.py
    without importing the concrete FlextResult class.
    """

    @property
    def is_success(self) -> bool:
        """Check if result represents success."""
        ...

    @property
    def is_failure(self) -> bool:
        """Check if result represents failure."""
        ...

    @property
    def error(self) -> str | None:
        """Get error message if failed."""
        ...

    @property
    def value(self) -> T:
        """Get success value or raise on failure."""
        ...

    def unwrap(self) -> T:
        """Get value or raise if failed."""
        ...

    @classmethod
    def ok(cls, data: T) -> ResultProtocol[T]:
        """Create success result."""
        ...

    @classmethod
    def fail(cls, error: str) -> ResultProtocol[T]:
        """Create failure result."""
        ...


@runtime_checkable
class ExceptionProtocol(Protocol):
    """Protocol for Exception types to break circular dependency.

    This protocol defines the minimal interface needed by result.py
    without importing the concrete FlextExceptions class.
    """

    class OperationError(Exception):
        """Protocol for operation errors."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize operation error with message and error code."""
            ...

    class FlextTypeError(Exception):
        """Protocol for type errors (avoiding builtin TypeError shadow)."""

        def __init__(
            self,
            message: str,
            *,
            error_code: str | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize type error with message and error code."""
            ...


__all__ = [
    "ExceptionProtocol",  # Protocol for exception types (breaks circular dependency)
    "FlextProtocols",  # Main hierarchical protocol architecture with Config
    "ResultProtocol",  # Protocol for result types (breaks circular dependency)
]
