"""CQRS command patterns for write operations.

Provides command definitions, handlers, and bus for CQRS pattern
implementation with validation and routing capabilities.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self, cast
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from flext_core.base_commands import (
    FlextAbstractCommand,
    FlextAbstractCommandBus,
    FlextAbstractCommandHandler,
    FlextAbstractQueryHandler,
)
from flext_core.loggings import FlextLoggerFactory
from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
)
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult
from flext_core.typings import (
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

if TYPE_CHECKING:
    from collections.abc import Callable
# FlextLogger imported for class methods only - instance methods use FlextLoggableMixin

# =============================================================================
# DOMAIN-SPECIFIC TYPES - Command Pattern Specializations
# =============================================================================

# Command pattern specific types for better domain modeling

# Query pattern specific types

# =============================================================================
# FLEXT COMMANDS - Unified command pattern
# =============================================================================


class FlextCommands:
    """CQRS implementation with command and query processing.

    Unified framework for CQRS patterns including command processing,
    query handling, and command bus routing.
    """

    # =============================================================================
    # BASE COMMAND - Foundation for all commands
    # =============================================================================

    class Command(  # type: ignore[misc]
        BaseModel,
        FlextAbstractCommand,
        FlextSerializableMixin,
        FlextValidatableMixin,
        FlextLoggableMixin,
    ):
        """Base command with validation and metadata.

        Implements FlextAbstractCommand.
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

        # Accept legacy 'mixin_setup' injections from tests without failing
        mixin_setup: object | None = Field(default=None, exclude=True)  # type: ignore[assignment]

        def model_post_init(self, __context: object, /) -> None:
            """Set command_type from class name if not provided."""
            if not self.command_type:
                # Convert class name to snake_case: CreateUserCommand -> create_user
                class_name = self.__class__.__name__
                # Remove "Command" suffix
                class_name = class_name.removesuffix("Command")

                # Convert CamelCase to snake_case
                snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()

                # Since the model is frozen, use object.__setattr__ if needed
                try:
                    object.__setattr__(self, "command_type", snake_name)
                except Exception:
                    # If field has no setter and is default, fallback to build new
                    data = self.model_dump()
                    data["command_type"] = snake_name
                    new_inst = type(self).model_validate(data)
                    object.__setattr__(self, "__dict__", new_inst.__dict__)

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
            error_code: str | None = None,
            error_data: dict[str, object] | None = None,
        ) -> FlextCommands.Result[T]:
            """Create failed result with metadata and optional error code."""
            # Convert error_data to TAnyDict for metadata compatibility
            metadata: TAnyDict | None = None
            if error_data is not None:
                metadata = {
                    k: v
                    for k, v in error_data.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }

            # Include error_code in metadata if provided
            if error_code is not None:
                if metadata is None:
                    metadata = {}
                metadata["error_code"] = error_code

            return cls(error=error, metadata=metadata)

    # =============================================================================
    # COMMAND HANDLER - Base handler interface
    # =============================================================================

    class Handler[TCommand, TResult](
        FlextAbstractCommandHandler[TCommand, TResult],
        FlextLoggableMixin,
        FlextTimingMixin,
    ):
        """Base command handler interface - implements FlextAbstractCommandHandler."""

        def __init__(
            self,
            handler_name: TServiceName | None = None,
            handler_id: TServiceName | None = None,
        ) -> None:
            """Initialize handler with logging."""
            self._handler_name = handler_name or self.__class__.__name__
            self.handler_id = handler_id or f"{self.__class__.__name__}_{id(self)}"
            # Logger now provided by FlextLoggableMixin - DRY principle applied

        @property
        def handler_name(self) -> str:
            """Get handler name - implements abstract method."""
            return self._handler_name

        def validate_command(self, command: TCommand) -> FlextResult[None]:
            """Validate command before handling - implements abstract method."""
            # Default implementation delegates to command's validation
            if hasattr(command, "validate_command"):
                result = command.validate_command()
                # Ensure we return FlextResult[None] type
                if isinstance(result, FlextResult):
                    return result
                return FlextResult.ok(None)
            return FlextResult.ok(None)

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
            # Validate command first via injected validator or command's own method
            if hasattr(self, "_validator") and self._validator is not None:
                try:
                    validate_method = getattr(self._validator, "validate_message", None)
                    if callable(validate_method):
                        vres = validate_method(command)
                        if hasattr(vres, "is_failure") and vres.is_failure:
                            return FlextResult.fail(vres.error or "Command validation failed")
                except Exception as e:
                    return FlextResult.fail(f"Command validation failed: {e}")
            elif hasattr(command, "validate_command"):
                validation_result = command.validate_command()
                if validation_result.is_failure:
                    error = validation_result.error or "Command validation failed"
                    return FlextResult.fail(error)

            # Check if can handle
            if not self.can_handle(command):
                error = f"{self.handler_name} cannot process {type(command).__name__}"
                return FlextResult.fail(error)

            # Handle the command and collect simple metrics when available
            try:
                result = self.handle(command)
                # Simple metrics update when collector injected
                if hasattr(self, "_metrics") and self._metrics is not None:
                    metrics = getattr(self._metrics, "get_metrics", None)
                    if callable(metrics):
                        _ = metrics()
                return result
            except (RuntimeError, OSError) as e:
                return FlextResult.fail(f"Command processing failed: {e}")

        def can_handle(self, command: object) -> bool:
            """Check if handler can process this command - implements abstract method.

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

    class Bus(FlextAbstractCommandBus, FlextLoggableMixin):
        """Command bus for routing commands to handlers.

        Implements FlextAbstractCommandBus.
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
            command_type: str,
            handler: FlextAbstractCommandHandler[object, object],
        ) -> None:
            """Register command handler - implements abstract method.

            Args:
                command_type: Command type to register
                handler: Handler instance

            """
            # Validate inputs
            if not FlextValidators.is_not_none(command_type):
                raise ValueError("Command type cannot be None")
            if not FlextValidators.is_not_none(handler):
                raise ValueError("Handler cannot be None")

            # Check if already registered
            if command_type in self._handlers:
                self.logger.warning(
                    "Handler already registered",
                    command_type=command_type,
                    existing_handler=self._handlers[command_type].__class__.__name__,
                )
                return

            # Register handler
            self._handlers[command_type] = handler

            self.logger.info(
                "Handler registered successfully",
                command_type=command_type,
                handler_type=handler.__class__.__name__,
                total_handlers=len(self._handlers),
            )

        def register_handler_flexible(
            self,
            handler_or_command_type: object | type[TCommand],
            handler: FlextCommands.Handler[TCommand, TResult] | None = None,
        ) -> FlextResult[None]:
            """Register handler with flexible arguments (backward compatibility).

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

        def unregister_handler(self, command_type: str) -> bool:
            """Unregister command handler - implements abstract method."""
            # Find handler by command type string
            for key, _handler in list(self._handlers.items()):
                if (hasattr(key, "__name__") and key.__name__ == command_type) or str(key) == command_type:
                    del self._handlers[key]
                    return True
            return False

        def send_command(self, command: FlextAbstractCommand) -> FlextResult[object]:
            """Send command - implements abstract method."""
            return self.execute(command)

        def get_registered_handlers(self) -> dict[str, object]:
            """Get registered handlers - implements abstract method."""
            return {str(k): v for k, v in self._handlers.items()}

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

            Args:
                command_type: Command type to handle

            Returns:
                Decorator function for command handler registration.

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

    class Query(  # type: ignore[misc]
        BaseModel,
        FlextValidatableMixin,
        FlextSerializableMixin,
    ):
        """Base query for read operations without side effects."""

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
            """Validate query with business logic."""
            errors: list[str] = []

            # Perform business validation
            if self.page_size <= 0:
                errors.append("Page size must be positive")
            if self.page_number <= 0:
                errors.append("Page number must be positive")
            if self.sort_order not in {"asc", "desc"}:
                errors.append("Sort order must be 'asc' or 'desc'")

            # Check validation results
            if errors:
                error_message = "; ".join(errors)
                return FlextResult.fail(f"Query validation failed: {error_message}")

            return FlextResult.ok(None)

        # Mixin methods are now available through proper inheritance:
        # - is_valid property (from FlextValidatableMixin)
        # - validation_errors property (from FlextValidatableMixin)
        # - has_validation_errors() method (from FlextValidatableMixin)
        # - to_dict_basic() method (from FlextSerializableMixin)
        # - All serialization methods (from FlextSerializableMixin)

    class QueryHandler[T, R](FlextAbstractQueryHandler[T, R]):
        """Base query handler built on abstract base.

        Provides default implementations using centralized base.
        """

        def __init__(self, handler_name: str | None = None) -> None:
            """Initialize query handler with optional name."""
            self._handler_name = handler_name or self.__class__.__name__

        @property
        def handler_name(self) -> str:
            """Get handler name for this query handler."""
            return self._handler_name

        def can_handle(self, query: object) -> bool:  # noqa: ARG002
            """Check if handler can handle query (always True for generic handler)."""
            return True

        def validate_query(self, query: T) -> FlextResult[None]:
            """Validate query object using its own validation method if available."""
            if hasattr(query, "validate_query"):
                return query.validate_query()  # type: ignore[no-any-return]
            return FlextResult.ok(None)

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
# FlextCommands.Command.model_rebuild()  # Disabled due to TAnyDict import issues
# FlextCommands.Query.model_rebuild()    # Disabled due to TAnyDict import issues

# Export API
__all__ = ["FlextCommands"]
