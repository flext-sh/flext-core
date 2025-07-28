"""FLEXT Core Commands Module.

Comprehensive Command Query Responsibility Segregation (CQRS) implementation with
command processing, validation, and routing. Implements consolidated architecture with
Pydantic validation, mixin inheritance, and type-safe operations.

Architecture:
    - CQRS pattern with clear command and query separation
    - Type-safe command and handler interfaces with generic constraints
    - Command bus implementation for routing and middleware processing
    - Pydantic-based validation with automatic serialization support
    - Multiple inheritance from specialized mixin classes for behavior composition
    - Event sourcing integration with correlation ID tracking

Command System Components:
    - FlextCommands.Command: Base command with metadata and validation
    - FlextCommands.Handler[T, R]: Generic command handler with lifecycle management
    - FlextCommands.Bus: Command routing and execution with middleware support
    - FlextCommands.Query: Read-only query operations with pagination
    - FlextCommands.QueryHandler[T, R]: Query handler interface for read operations
    - FlextCommands.Decorators: Function-based handler registration

Maintenance Guidelines:
    - Create domain commands by inheriting from FlextCommands.Command
    - Implement handlers by inheriting from FlextCommands.Handler[Command, Result]
    - Register handlers with command bus for automatic routing
    - Use correlation IDs for request tracking across service boundaries
    - Implement custom validation in command validate_command method
    - Follow CQRS principles with clear command/query separation

Design Decisions:
    - Nested class organization within FlextCommands for namespace management
    - Pydantic BaseModel for automatic validation and serialization
    - Frozen models for immutability and thread safety
    - Generic type parameters for compile-time type safety
    - Command bus pattern for centralized routing and middleware
    - Payload integration for transport and persistence

CQRS Implementation:
    - Commands: Write operations with side effects and state changes
    - Queries: Read operations without side effects with pagination support
    - Handlers: Processing logic with validation and error handling
    - Bus: Routing infrastructure with middleware and execution tracking
    - Event integration: Correlation tracking for event sourcing patterns

Enterprise Features:
    - Correlation ID tracking for distributed system observability
    - Automatic timestamp generation with UTC timezone handling
    - User context tracking for audit and authorization
    - Command serialization for transport and persistence
    - Middleware pipeline for cross-cutting concerns
    - Comprehensive logging and metrics integration

Dependencies:
    - pydantic: Data validation and immutable model configuration
    - mixins: Serializable, validatable, timing, and logging behavior patterns
    - payload: FlextPayload integration for transport and persistence
    - result: FlextResult pattern for consistent error handling
    - types: Generic type variables and command-specific type aliases
    - utilities: ID generation and type guards for validation

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Generic, Self
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core.loggings import FlextLoggerFactory
from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
)
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult
from flext_core.types import (
    R,
    T,
    TAnyDict,
    TCommand,
    TCorrelationId,
    TEntityId,
    TResult,
    TServiceName,
    TUserId,
)
from flext_core.utilities import FlextGenerators, FlextTypeGuards
from flext_core.validation import FlextValidators

# FlextLogger imported for class methods only - instance methods use FlextLoggableMixin


# =============================================================================
# FLEXT COMMANDS - Unified command pattern
# =============================================================================


class FlextCommands:
    """Comprehensive CQRS implementation with command and query processing.

    Unified command pattern framework providing organized access to all FLEXT command
    patterns including CQRS command processing, query handling, and command bus routing.
    Serves as the primary namespace for command-related functionality with nested class
    organization.

    Architecture:
        - CQRS (Command Query Responsibility Segregation) pattern implementation
        - Namespace organization with nested classes for related functionality
        - Command and query separation with distinct handler interfaces
        - Command bus pattern for centralized routing and middleware
        - Type-safe command and handler interfaces with generic constraints
        - Integration with event sourcing through correlation tracking

    CQRS Components:
        - Command: Write operations with side effects and state changes
        - Handler: Command processing logic with validation and lifecycle management
        - Bus: Command routing infrastructure with middleware pipeline
        - Query: Read operations without side effects with pagination support
        - QueryHandler: Query processing interface for read-only operations
        - Decorators: Function-based handler registration utilities

    Enterprise Features:
        - Type-safe command interfaces with compile-time verification
        - Automatic validation through Pydantic integration
        - Correlation ID tracking for distributed system observability
        - Command serialization for transport and persistence
        - Middleware pipeline for cross-cutting concerns
        - Comprehensive logging and metrics collection

    Usage Patterns:
        # Define domain command
        class CreateOrderCommand(FlextCommands.Command):
            customer_id: str
            items: list[OrderItem]

            def validate_command(self) -> FlextResult[None]:
                if not self.items:
                    return FlextResult.fail("Order must have items")
                return FlextResult.ok(None)

        # Implement command handler
        class CreateOrderHandler(FlextCommands.Handler[CreateOrderCommand, Order]):
            def handle(self, command: CreateOrderCommand) -> FlextResult[Order]:
                order = self.order_service.create_order(
                    command.customer_id,
                    command.items
                )
                return FlextResult.ok(order)

        # Register and execute through bus
        bus = FlextCommands.create_command_bus()
        handler = CreateOrderHandler()
        bus.register_handler(CreateOrderCommand, handler)

        command = CreateOrderCommand(
            customer_id="cust_123",
            items=[OrderItem(product_id="prod_456", quantity=2)]
        )
        result = bus.execute(command)

        # Query pattern for read operations
        class GetOrdersQuery(FlextCommands.Query):
            customer_id: str
            status: str | None = None

        class GetOrdersHandler(FlextCommands.QueryHandler[GetOrdersQuery, list[Order]]):
            def handle(self, query: GetOrdersQuery) -> FlextResult[list[Order]]:
                orders = self.order_repository.find_by_customer(
                    query.customer_id,
                    status=query.status,
                    page_size=query.page_size,
                    page_number=query.page_number
                )
                return FlextResult.ok(orders)

    Design Pattern Integration:
        - Command Pattern: Encapsulating requests as objects
        - Mediator Pattern: Command bus as communication mediator
        - Chain of Responsibility: Middleware pipeline processing
        - Template Method: Handler lifecycle with customizable hooks
        - Factory Method: Command and handler creation utilities
    """

    # =============================================================================
    # BASE COMMAND - Foundation for all commands
    # =============================================================================

    class Command(
        BaseModel,
        FlextSerializableMixin,
        FlextValidatableMixin,
        FlextLoggableMixin,
    ):
        """Base command with validation and metadata.

        Commands represent intentions to change system state.
        Uses Pydantic for automatic validation and FLEXT mixins.
        """

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            str_strip_whitespace=True,
            extra="forbid",
        )

        # Use FlextUtilities for ID generation
        command_id: TEntityId = Field(
            default_factory=FlextGenerators.generate_uuid,
            description="Unique command identifier",
        )

        command_type: TServiceName = Field(
            default="",
            description="Command type for routing",
        )

        timestamp: datetime = Field(
            default_factory=datetime.utcnow,
            description="Command creation timestamp",
        )

        user_id: TUserId | None = Field(
            default=None,
            description="User who initiated the command",
        )

        correlation_id: TCorrelationId = Field(
            default_factory=FlextGenerators.generate_uuid,
            description="Correlation ID for tracking",
        )

        @field_validator("command_type", mode="after")
        @classmethod
        def set_command_type(cls, v: TServiceName) -> TServiceName:
            """Auto-set command type from class name if not provided."""
            if not v:
                return cls.__name__
            return v

        def to_payload(self) -> FlextPayload[TAnyDict]:
            """Convert command to FlextPayload for transport."""
            self.logger.debug(
                "Converting command to payload",
                command_type=self.command_type,
                command_id=self.command_id,
            )

            # Serialize timestamp properly
            data = self.model_dump(exclude_none=True)
            if "timestamp" in data and isinstance(data["timestamp"], datetime):
                data["timestamp"] = data["timestamp"].isoformat()

            return FlextPayload.create(
                data=data,
                type=self.command_type,
            ).unwrap()

        @classmethod
        def from_payload(
            cls,
            payload: FlextPayload[TAnyDict],
        ) -> FlextResult[Self]:
            """Create command from FlextPayload with validation."""
            # Use FlextLogger directly for class methods
            logger = FlextLoggerFactory.get_logger(f"{cls.__module__}.{cls.__name__}")
            logger.debug(
                "Creating command from payload",
                payload_type=payload.metadata.get("type", "unknown"),
                expected_type=cls.__name__,
            )

            # Validate payload type matches
            expected_type = payload.metadata.get("type", "")
            if expected_type not in {cls.__name__, ""}:
                logger.warning(
                    "Payload type mismatch",
                    expected=cls.__name__,
                    actual=expected_type,
                )

            # Parse timestamp if string
            data = payload.data.copy()

            # Extract explicit parameters for Command constructor
            command_id = TEntityId(
                data.get("command_id", FlextGenerators.generate_uuid()),
            )
            command_type = TServiceName(data.get("command_type", cls.__name__))

            # Handle timestamp
            timestamp_raw = data.get("timestamp")
            if isinstance(timestamp_raw, str):
                timestamp = datetime.fromisoformat(timestamp_raw)
            elif isinstance(timestamp_raw, datetime):
                timestamp = timestamp_raw
            else:
                timestamp = datetime.now(tz=ZoneInfo("UTC"))

            # Handle user_id (optional)
            user_id_raw = data.get("user_id")
            user_id = TUserId(user_id_raw) if user_id_raw else None

            # Handle correlation_id
            correlation_id_raw = data.get("correlation_id")
            correlation_id = (
                TCorrelationId(correlation_id_raw)
                if correlation_id_raw
                else TCorrelationId(FlextGenerators.generate_uuid())
            )

            # Remove processed fields from data
            remaining_data = {
                k: v
                for k, v in data.items()
                if k
                not in {
                    "command_id",
                    "command_type",
                    "timestamp",
                    "user_id",
                    "correlation_id",
                }
            }

            command = cls(
                command_id=command_id,
                command_type=command_type,
                timestamp=timestamp,
                user_id=user_id,
                correlation_id=correlation_id,
                **remaining_data,
            )

            # Validate using FlextValidation
            if hasattr(command, "validate_command"):
                validation_result = command.validate_command()
                if validation_result.is_failure:
                    return FlextResult.fail(
                        validation_result.error or "Command validation failed",
                    )

            logger.info(
                "Command created from payload",
                command_type=command.command_type,
                command_id=command.command_id,
            )
            return FlextResult.ok(command)

        def validate_command(self) -> FlextResult[None]:
            """Validate command using FlextValidation."""
            # Override in subclasses for custom validation
            return FlextResult.ok(None)

    # =============================================================================
    # COMMAND HANDLER - Base handler interface
    # =============================================================================

    class Handler(
        ABC,
        FlextLoggableMixin,
        FlextTimingMixin,
        Generic[TCommand, TResult],
    ):
        """Base command handler interface.

        Handlers execute commands and return results.
        Uses FlextResult, FlextLogger, FlextTypes extensively.
        Uses FlextLoggableMixin to eliminate logger duplication (DRY).
        Uses FlextTimingMixin to eliminate timing duplication (DRY).
        """

        def __init__(self) -> None:
            """Initialize handler with logging."""
            self._handler_name = self.__class__.__name__
            # Logger now provided by FlextLoggableMixin - DRY principle applied

        @abstractmethod
        def handle(self, command: TCommand) -> FlextResult[TResult]:
            """Handle the command and return result.

            Args:
                command: Command to handle

            Returns:
                FlextResult with execution result or error

            """

        def can_handle(self, command: object) -> bool:
            """Check if handler can process this command.

            Uses FlextUtilities type guards for validation.
            """
            # Log handler check
            self.logger.debug(
                "Checking if handler can process command",
                command_type=type(command).__name__,
            )

            # Get expected command type from Generic parameter
            if hasattr(self, "__orig_bases__"):
                for base in self.__orig_bases__:
                    if hasattr(base, "__args__") and len(base.__args__) >= 1:
                        expected_type = base.__args__[0]
                        # Use BASE type guard directly - MAXIMIZA base funcionalidade
                        can_handle_result = FlextTypeGuards.is_instance_of(
                            command,
                            expected_type,
                        )

                        self.logger.debug(
                            "Handler check result",
                            can_handle=can_handle_result,
                            expected_type=getattr(
                                expected_type,
                                "__name__",
                                str(expected_type),
                            ),
                        )
                        return bool(can_handle_result)

            self.logger.warning("Could not determine handler type constraints")
            return True

        def execute(self, command: TCommand) -> FlextResult[TResult]:
            """Execute command with full logging and error handling."""
            self.logger.info(
                "Executing command",
                command_type=type(command).__name__,
                command_id=getattr(command, "command_id", "unknown"),
            )

            # Validate command can be handled
            if not self.can_handle(command):
                error_msg = (
                    f"{self._handler_name} cannot handle {type(command).__name__}"
                )
                self.logger.error(error_msg)
                return FlextResult.fail(error_msg)

            start_time = self._start_timing()

            try:
                result = self.handle(command)

                self.logger.info(
                    "Command executed successfully",
                    command_type=type(command).__name__,
                    execution_time_ms=self._get_execution_time_ms_rounded(start_time),
                    success=result.is_success,
                )
            except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                self.logger.exception(
                    "Command execution failed",
                    command_type=type(command).__name__,
                    execution_time_ms=self._get_execution_time_ms_rounded(start_time),
                    error=str(e),
                )
                raise
            else:
                return result

    # =============================================================================
    # COMMAND BUS - Command routing and execution
    # =============================================================================

    class Bus(FlextLoggableMixin):
        """Command bus for routing commands to handlers.

        Implements command pattern with automatic handler discovery.
        Uses FlextContainer internally for dependency injection.
        Uses FlextLoggableMixin to eliminate logger duplication (DRY principle).
        """

        def __init__(self) -> None:
            """Initialize command bus with logging."""
            self._handlers: dict[type[object], object] = {}
            self._middleware: list[object] = []
            self._execution_count = 0

            self.logger.info(
                "Command bus initialized",
                bus_id=id(self),
            )

        def register_handler(
            self,
            command_type: type[TCommand],
            handler: FlextCommands.Handler[TCommand, TResult],
        ) -> FlextResult[None]:
            """Register handler for command type with validation.

            Args:
                command_type: Type of command to handle
                handler: Handler instance

            Returns:
                FlextResult indicating registration success

            """
            # Use BASE validators directly - MAXIMIZA base usage
            if not FlextValidators.is_not_none(command_type):
                return FlextResult.fail("Command type cannot be None")

            if not FlextValidators.is_not_none(handler):
                return FlextResult.fail("Handler cannot be None")

            # Check if already registered
            if command_type in self._handlers:
                self.logger.warning(
                    "Handler already registered",
                    command_type=command_type.__name__,
                    existing_handler=self._handlers[command_type].__class__.__name__,
                )
                return FlextResult.fail(
                    f"Handler already registered for {command_type.__name__}",
                )

            # Register handler
            self._handlers[command_type] = handler

            self.logger.info(
                "Handler registered successfully",
                command_type=command_type.__name__,
                handler_type=handler.__class__.__name__,
                total_handlers=len(self._handlers),
            )

            return FlextResult.ok(None)

        def execute(self, command: TCommand) -> FlextResult[object]:
            """Execute command using registered handler with full logging.

            Args:
                command: Command to execute

            Returns:
                FlextResult with execution result

            """
            self._execution_count += 1
            command_type = type(command)

            self.logger.info(
                "Executing command via bus",
                command_type=command_type.__name__,
                command_id=getattr(command, "command_id", "unknown"),
                execution_count=self._execution_count,
            )

            # Validate handler exists
            if command_type not in self._handlers:
                self.logger.error(
                    "No handler registered",
                    command_type=command_type.__name__,
                    registered_types=[t.__name__ for t in self._handlers],
                )
                return FlextResult.fail(
                    f"No handler registered for {command_type.__name__}",
                )

            handler = self._handlers[command_type]

            # Apply middleware with logging
            for i, middleware in enumerate(self._middleware):
                self.logger.debug(
                    "Applying middleware",
                    middleware_index=i,
                    middleware_type=type(middleware).__name__,
                )

                if hasattr(middleware, "process"):
                    result = middleware.process(command, handler)
                    if hasattr(result, "is_failure") and result.is_failure:
                        self.logger.warning(
                            "Middleware rejected command",
                            middleware_type=type(middleware).__name__,
                            error=getattr(result, "error", "Unknown error"),
                        )
                        error_msg = str(
                            getattr(result, "error", "Middleware rejected command"),
                        )
                        return FlextResult.fail(error_msg)

            # Execute handler
            self.logger.debug(
                "Delegating to handler",
                handler_type=handler.__class__.__name__,
            )

            # Use handler's execute method if available
            if hasattr(handler, "execute"):
                result = handler.execute(command)
                return (
                    result if hasattr(result, "is_success") else FlextResult.ok(result)
                )
            if hasattr(handler, "handle"):
                result = handler.handle(command)
                return (
                    result if hasattr(result, "is_success") else FlextResult.ok(result)
                )
            return FlextResult.fail("Handler has no execute or handle method")

        def add_middleware(self, middleware: object) -> None:
            """Add middleware to processing pipeline."""
            self._middleware.append(middleware)

    # =============================================================================
    # COMMAND DECORATORS - Convenience decorators
    # =============================================================================

    class Decorators:
        """Decorators for command handling."""

        @staticmethod
        def command_handler(
            command_type: type[object],
        ) -> Callable[[Callable[[object], object]], Callable[[object], object]]:
            """Mark a function as command handler.

            Usage:
                @FlextCommands.Decorators.command_handler(CreateUserCommand)
                def handle_create_user(command: CreateUserCommand) -> FlextResult[User]:
                    # Handle command
                    return FlextResult.ok(user)
            """

            def decorator(
                func: Callable[[object], object],
            ) -> Callable[[object], object]:
                # Create handler class from function
                class FunctionHandler(FlextCommands.Handler[object, object]):
                    def handle(self, command: object) -> FlextResult[object]:
                        result = func(command)
                        if hasattr(result, "is_success"):
                            return result
                        return FlextResult.ok(result)

                # Store metadata for automatic registration (dynamic attributes)
                func.command_type = command_type
                func.handler_instance = FunctionHandler()

                return func

            return decorator

    # =============================================================================
    # QUERY PATTERNS - Read-only operations
    # =============================================================================

    class Query(BaseModel, FlextSerializableMixin, FlextValidatableMixin):
        """Base query for read operations.

        Queries represent requests for data without side effects.
        Uses FLEXT mixins for serialization and validation.
        """

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
        )

        query_id: TEntityId | None = None
        query_type: TServiceName | None = None
        page_size: int = 100
        page_number: int = 1
        sort_by: TServiceName | None = None
        sort_order: TServiceName = "asc"

    class QueryHandler(ABC, Generic[T, R]):
        """Base query handler interface."""

        @abstractmethod
        def handle(self, query: T) -> FlextResult[R]:
            """Handle query and return result."""

    # =============================================================================
    # FACTORY METHODS - Convenience builders
    # =============================================================================

    @staticmethod
    def create_command_bus() -> FlextCommands.Bus:
        """Create a new command bus instance."""
        return FlextCommands.Bus()

    @staticmethod
    def create_simple_handler(
        handler_func: Callable[[object], object],
    ) -> FlextCommands.Handler[object, object]:
        """Create handler from function.

        Args:
            handler_func: Function that takes command and returns FlextResult

        Returns:
            Handler instance

        """

        class SimpleHandler(FlextCommands.Handler[object, object]):
            def handle(self, command: object) -> FlextResult[object]:
                result = handler_func(command)
                if hasattr(result, "is_success"):
                    return result
                return FlextResult.ok(result)

        return SimpleHandler()


# Export API
__all__ = ["FlextCommands"]
