"""FLEXT Core Commands - CQRS Layer Command Implementation.

Command Query Responsibility Segregation (CQRS) implementation providing command
processing, validation, and routing across the 32-project FLEXT ecosystem. Foundation
for write operations with side effects, business logic encapsulation, and distributed
system coordination in data integration pipelines.

Module Role in Architecture:
    CQRS Layer → Command Processing → Write Operations with Side Effects

    This module provides CQRS command patterns used throughout FLEXT projects:
    - Command definitions for write operations with business logic validation
    - Command handlers for processing business operations with error handling
    - Command bus for routing and middleware processing with cross-cutting concerns
    - Query separation ensuring read/write operation distinction

Command Architecture Patterns:
    CQRS Separation: Clear distinction between commands (write) and queries (read)
    Command Bus: Centralized routing with middleware pipeline support
    Type Safety: Generic type parameters for compile-time command/handler safety
    Immutable Commands: Frozen models ensuring command integrity during processing

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Command base, handler interface, validation
    🚧 Active Development: Complete CQRS implementation (Priority 2 - October 2025)
    📋 TODO Integration: Query Bus and auto-discovery (Priority 2)

CQRS Command Features:
    FlextCommands.Command: Base command with metadata and correlation tracking
    FlextCommands.Handler: Generic command handler with lifecycle management
    FlextCommands.Bus: Command routing with middleware and execution tracking
    FlextCommands.Query: Read-only query operations with pagination support

Ecosystem Usage Patterns:
    # FLEXT Service Commands
    class CreateUserCommand(FlextCommands.Command):
        name: str
        email: str

        def validate_command(self) -> FlextResult[None]:
            if '@' not in self.email:
                return FlextResult.fail("Invalid email format")
            return FlextResult.ok(None)

    class CreateUserHandler(FlextCommands.Handler[CreateUserCommand, User]):
        def handle(self, command: CreateUserCommand) -> FlextResult[User]:
            user = User(name=command.name, email=command.email)
            return self.user_repository.save(user)

    # Singer Tap/Target Commands
    class ExtractOracleDataCommand(FlextCommands.Command):
        table_name: str
        batch_size: int

    # ALGAR Migration Commands
    class MigrateLdapUsersCommand(FlextCommands.Command):
        source_dn: str
        target_dn: str
        batch_size: int = 100

CQRS Patterns Implementation:
    - Commands: Write operations with side effects and state changes
    - Handlers: Business logic processing with validation and error handling
    - Bus: Routing infrastructure with middleware pipeline support
    - Correlation Tracking: Request tracing across distributed services

Quality Standards:
    - All commands must implement validate_command for business validation
    - Command handlers must be stateless and thread-safe
    - Commands must be immutable (frozen models) to prevent modification
    - Correlation IDs must be used for distributed request tracking

See Also:
    docs/TODO.md: Priority 2 - Complete CQRS implementation
    handlers.py: Handler patterns and execution lifecycle
    interfaces.py: FlextHandler interface definitions

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self, cast
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from flext_core.flext_types import (
    R,
    T,
    TAnyDict,
    TAnyList,
    TCommand,
    TCorrelationId,
    TEntityId,
    TResult,
    TServiceName,
    TUserId,
)

# Base mixins now imported from proper public API
from flext_core.loggings import FlextLoggerFactory
from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
)
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult
from flext_core.utilities import FlextGenerators, FlextTypeGuards
from flext_core.validation import FlextValidators

# FlextLogger imported for class methods only - instance methods use FlextLoggableMixin

# =============================================================================
# DOMAIN-SPECIFIC TYPES - Command Pattern Specializations
# =============================================================================

# Command pattern specific types for better domain modeling
type TCommandId = TCorrelationId  # Command instance identifier
type TCommandType = str  # Command type name for routing
type THandlerName = TServiceName  # Command handler service name
type TCommandPayload = TAnyDict  # Command data payload
type TCommandResult = FlextResult[object]  # Command execution result
type TCommandMetadata = TAnyDict  # Command metadata for middleware
type TMiddlewareName = str  # Middleware component name
type TValidationRule = str  # Command validation rule identifier
type TCommandBusId = str  # Command bus instance identifier
type TCommandPriority = int  # Command execution priority (1-10)

# Query pattern specific types
type TQueryId = TCorrelationId  # Query instance identifier
type TQueryType = str  # Query type name for routing
type TQueryResult[T] = FlextResult[T]  # Query result with type parameter
type TQueryCriteria = TAnyDict  # Query filtering criteria
type TQueryProjection = TAnyList  # Query result projection fields
type TPaginationToken = str  # Query pagination continuation token

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
        # Define domain command using shared domain models

        class CreateOrderCommand(FlextCommands.Command):
            customer_id: str
            items: list[dict[str, str]]  # Use generic dict for compatibility

            def validate_command(self) -> FlextResult[None]:
                if not self.items:
                    return FlextResult.fail("Order must have items")
                return FlextResult.ok(None)

        # Implement command handler using SharedDomainFactory
        class CreateOrderHandler(FlextCommands.Handler[CreateOrderCommand, Order]):
            def handle(self, command: CreateOrderCommand) -> FlextResult[Order]:
                # Use SharedDomainFactory for consistent object creation
                order_result = SharedDomainFactory.create_order(
                    customer_id=command.customer_id,
                    items=command.items
                )
                return order_result  # Already returns FlextResult[Order]

        # Register and execute through bus
        bus = FlextCommands.create_command_bus()
        handler = CreateOrderHandler()
        bus.register_handler(CreateOrderCommand, handler)

        command = CreateOrderCommand(
            customer_id="cust_123",
            items=[{
                "product_id": "prod_456",
                "product_name": "Product",
                "quantity": "2",
                "unit_price": "100.0",
                "currency": "USD"
            }]
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
            default_factory=lambda: datetime.now(UTC),
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

        def model_post_init(self, __context: object, /) -> None:
            """Set command_type from class name if not provided."""
            if not self.command_type:
                # Convert class name to snake_case: CreateUserCommand -> create_user
                class_name = self.__class__.__name__
                # Remove "Command" suffix
                class_name = class_name.removesuffix("Command")

                # Convert CamelCase to snake_case
                snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()

                # Since the model is frozen, use object.__setattr__
                object.__setattr__(self, "command_type", snake_name)

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

            # Parse timestamp if string - handle None case
            data = payload.data.copy() if payload.data is not None else {}

            # Extract explicit parameters for Command constructor
            command_id = TEntityId(
                data.get("command_id", FlextGenerators.generate_uuid()),
            )
            command_type = TServiceName(data.get("command_type", cls.__name__))

            # Handle timestamp (TAnyDict only allows primitive types, not datetime)
            timestamp_raw = data.get("timestamp")
            if isinstance(timestamp_raw, str):
                timestamp = datetime.fromisoformat(timestamp_raw)
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

        def require_field(
            self,
            field_name: str,
            value: object,
            error_msg: str = "",
        ) -> FlextResult[None]:
            """Validate required field with custom error."""
            if not value or (isinstance(value, str) and not value.strip()):
                msg = error_msg or f"{field_name} is required"
                return FlextResult.fail(msg)
            return FlextResult.ok(None)

        def require_email(
            self,
            email: str,
            field_name: str = "email",
        ) -> FlextResult[None]:
            """Validate email format."""
            if (
                not email
                or "@" not in email
                or "." not in email.rsplit("@", maxsplit=1)[-1]
            ):
                return FlextResult.fail(f"Invalid {field_name} format")
            return FlextResult.ok(None)

        def require_min_length(
            self,
            value: str,
            min_len: int,
            field_name: str,
        ) -> FlextResult[None]:
            """Validate minimum string length."""
            if len(value.strip()) < min_len:
                error_msg = f"{field_name} must be at least {min_len} characters"
                return FlextResult.fail(error_msg)
            return FlextResult.ok(None)

        def get_metadata(self) -> TAnyDict:
            """Get command metadata."""
            return {
                "command_id": self.command_id,
                "command_type": self.command_type,
                "command_class": self.__class__.__name__,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "user_id": self.user_id,
                "correlation_id": self.correlation_id,
            }

    # =============================================================================
    # COMMAND RESULT - Result with metadata support
    # =============================================================================

    class Result[T](FlextResult[T]):
        """Command result with metadata support."""

        def __init__(
            self,
            data: T | None = None,
            error: str | None = None,
            metadata: TAnyDict | None = None,
        ) -> None:
            """Initialize command result with metadata."""
            super().__init__(data=data, error=error)
            self.metadata = metadata or {}

        @classmethod
        def ok(
            cls,
            data: T,
            metadata: TAnyDict | None = None,
        ) -> FlextCommands.Result[T]:
            """Create successful result with metadata."""
            return cls(data=data, metadata=metadata)

        @classmethod
        def fail(
            cls,
            error: str,
            error_code: str | None = None,  # noqa: ARG003
            error_data: dict[str, object] | None = None,
        ) -> FlextCommands.Result[T]:
            """Create failed result with metadata."""
            # Convert error_data to TAnyDict for metadata compatibility
            metadata: TAnyDict | None = None
            if error_data is not None:
                metadata = {
                    k: v
                    for k, v in error_data.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
            return cls(error=error, metadata=metadata)

    # =============================================================================
    # COMMAND HANDLER - Base handler interface
    # =============================================================================

    class Handler[TCommand, TResult](
        ABC,
        FlextLoggableMixin,
        FlextTimingMixin,
    ):
        """Base command handler interface.

        Handlers execute commands and return results.
        Uses FlextResult, FlextLogger, FlextTypes extensively.
        Uses FlextLoggableMixin to eliminate logger duplication (DRY).
        Uses FlextTimingMixin to eliminate timing duplication (DRY).
        """

        def __init__(
            self,
            handler_name: TServiceName | None = None,
            handler_id: TServiceName | None = None,
        ) -> None:
            """Initialize handler with logging."""
            self._handler_name = handler_name or self.__class__.__name__
            self.handler_id = handler_id or f"{self.__class__.__name__}_{id(self)}"
            self.handler_name = self._handler_name
            # Logger now provided by FlextLoggableMixin - DRY principle applied

        @abstractmethod
        def handle(self, command: TCommand) -> FlextResult[TResult]:
            """Handle the command and return result.

            Args:
                command: Command to handle

            Returns:
                FlextResult with execution result or error

            """

        def process_command(self, command: TCommand) -> FlextResult[TResult]:
            """Process command with validation and handling.

            Args:
                command: Command to process

            Returns:
                FlextResult with processing result or error

            """
            # Validate command first
            if hasattr(command, "validate_command"):
                validation_result = command.validate_command()
                if validation_result.is_failure:
                    error = validation_result.error or "Command validation failed"
                    return FlextResult.fail(error)

            # Check if can handle
            if not self.can_handle(command):
                error = f"{self.handler_name} cannot process {type(command).__name__}"
                return FlextResult.fail(error)

            # Handle the command
            try:
                return self.handle(command)
            except (RuntimeError, OSError) as e:
                return FlextResult.fail(f"Command processing failed: {e}")

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
            self._handlers: dict[object, object] = {}
            self._middleware: list[object] = []
            self._execution_count = 0

            self.logger.info(
                "Command bus initialized",
                bus_id=id(self),
            )

        def register_handler(
            self,
            handler_or_command_type: object | type[TCommand],
            handler: FlextCommands.Handler[TCommand, TResult] | None = None,
        ) -> FlextResult[None]:
            """Register handler with flexible arguments.

            Args:
                handler_or_command_type: Handler instance or command type
                handler: Handler instance (when first arg is command type)

            Returns:
                FlextResult indicating registration success

            """
            # Handle single argument case (just the handler)
            if handler is None:
                if not hasattr(handler_or_command_type, "handle"):
                    return FlextResult.fail(
                        "Invalid handler: must have 'handle' method",
                    )
                handler_obj = handler_or_command_type
                # Use handler_id as key for uniqueness
                key = getattr(handler_obj, "handler_id", handler_obj.__class__.__name__)
                self._handlers[key] = handler_obj

                self.logger.info(
                    "Handler registered successfully",
                    handler_id=key,
                    handler_type=handler_obj.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return FlextResult.ok(None)
            # Handle two argument case (command_type, handler)
            command_type = handler_or_command_type
            handler_obj = handler

            # Use BASE validators directly - MAXIMIZA base usage
            if not FlextValidators.is_not_none(command_type):
                return FlextResult.fail("Command type cannot be None")

            if not FlextValidators.is_not_none(handler_obj):
                return FlextResult.fail("Handler cannot be None")

            # Check if already registered
            if command_type in self._handlers:
                command_name = getattr(command_type, "__name__", str(command_type))
                self.logger.warning(
                    "Handler already registered",
                    command_type=command_name,
                    existing_handler=self._handlers[command_type].__class__.__name__,
                )
                return FlextResult.fail(
                    f"Handler already registered for {command_name}",
                )

            # Register handler
            self._handlers[command_type] = handler_obj

            command_name = getattr(command_type, "__name__", str(command_type))
            self.logger.info(
                "Handler registered successfully",
                command_type=command_name,
                handler_type=handler_obj.__class__.__name__,
                total_handlers=len(self._handlers),
            )

            return FlextResult.ok(None)

        def execute(self, command: TCommand) -> FlextResult[object]:
            """Execute command using registered handler with full logging."""
            self._execution_count += 1
            command_type = type(command)

            self._log_command_execution(command, command_type)

            # Validate command
            validation_result = self._validate_command(command, command_type)
            if validation_result.is_failure:
                return FlextResult.fail(validation_result.error or "Validation failed")

            # Find handler
            handler = self._find_command_handler(command)
            if handler is None:
                return self._handle_no_handler_found(command_type)

            # Apply middleware
            middleware_result = self._apply_middleware(command, handler)
            if middleware_result.is_failure:
                return FlextResult.fail(middleware_result.error or "Middleware failed")

            # Execute handler
            return self._execute_handler(handler, command)

        def _log_command_execution(self, command: TCommand, command_type: type) -> None:
            """Log command execution start."""
            self.logger.info(
                "Executing command via bus",
                command_type=command_type.__name__,
                command_id=getattr(command, "command_id", "unknown"),
                execution_count=self._execution_count,
            )

        def _validate_command(
            self,
            command: TCommand,
            command_type: type,
        ) -> FlextResult[None]:
            """Validate command if it has validation method."""
            if hasattr(command, "validate_command"):
                validation_result = command.validate_command()
                if validation_result.is_failure:
                    self.logger.warning(
                        "Command validation failed",
                        command_type=command_type.__name__,
                        error=validation_result.error,
                    )
                    return FlextResult.fail(
                        validation_result.error or "Command validation failed",
                    )
            return FlextResult.ok(None)

        def _find_command_handler(self, command: TCommand) -> object | None:
            """Find handler that can handle this command."""
            for registered_handler in self._handlers.values():
                if hasattr(
                    registered_handler,
                    "can_handle",
                ) and registered_handler.can_handle(command):
                    return registered_handler
            return None

        def _handle_no_handler_found(self, command_type: type) -> FlextResult[object]:
            """Handle case when no handler is found."""
            handler_names = [h.__class__.__name__ for h in self._handlers.values()]
            self.logger.error(
                "No handler found",
                command_type=command_type.__name__,
                registered_handlers=handler_names,
            )
            return FlextResult.fail(f"No handler found for {command_type.__name__}")

        def _apply_middleware(
            self,
            command: TCommand,
            handler: object,
        ) -> FlextResult[None]:
            """Apply middleware pipeline."""
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
            return FlextResult.ok(None)

        def _execute_handler(
            self,
            handler: object,
            command: TCommand,
        ) -> FlextResult[object]:
            """Execute the command handler."""
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

        def get_all_handlers(self) -> list[object]:
            """Get all registered handlers."""
            return list(self._handlers.values())

        def find_handler(self, command: object) -> object | None:
            """Find handler for command type."""
            for handler in self._handlers.values():
                if hasattr(handler, "can_handle") and handler.can_handle(command):
                    return handler
            return None

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
                        if hasattr(result, "is_success") and isinstance(
                            result,
                            FlextResult,
                        ):
                            return result
                        return FlextResult.ok(result)

                # Create wrapper function with metadata instead of dynamic attributes
                def wrapper(*args: object, **kwargs: object) -> object:
                    return func(*args, **kwargs)

                # Store metadata in wrapper's __dict__ for type safety
                wrapper.__dict__["command_type"] = command_type
                wrapper.__dict__["handler_instance"] = FunctionHandler()

                return wrapper

            return decorator

    # =============================================================================
    # QUERY PATTERNS - Read-only operations
    # =============================================================================

    class Query(
        BaseModel,
        FlextValidatableMixin,
        FlextSerializableMixin,
    ):
        """Base query for read operations without side effects.

        Queries represent requests for data without side effects.
        Uses proper mixin inheritance for validation and serialization functionality.
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

        def validate_query(self) -> FlextResult[None]:
            """Validate query with business logic using proper mixin interface."""
            # Clear previous errors
            self.clear_validation_errors()

            # Perform business validation
            if self.page_size <= 0:
                self.add_validation_error("Page size must be positive")
            if self.page_number <= 0:
                self.add_validation_error("Page number must be positive")
            if self.sort_order not in {"asc", "desc"}:
                self.add_validation_error("Sort order must be 'asc' or 'desc'")

            # Check validation results
            if self.has_validation_errors():
                errors = "; ".join(self.validation_errors)
                return FlextResult.fail(f"Query validation failed: {errors}")

            # Mark as valid
            self.mark_valid()
            return FlextResult.ok(None)

        # Mixin methods are now available through proper inheritance:
        # - is_valid property (from FlextValidatableMixin)
        # - validation_errors property (from FlextValidatableMixin)
        # - has_validation_errors() method (from FlextValidatableMixin)
        # - to_dict_basic() method (from FlextSerializableMixin)
        # - All serialization methods (from FlextSerializableMixin)

    class QueryHandler[T, R](ABC):
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
                    return cast("FlextResult[object]", result)
                return FlextResult.ok(result)

        return SimpleHandler()


# =============================================================================
# MODEL REBUILDS - Resolve forward references for Pydantic
# =============================================================================

# Rebuild nested models to resolve forward references after import
FlextCommands.Command.model_rebuild()
FlextCommands.Query.model_rebuild()

# Export API
__all__ = ["FlextCommands"]
