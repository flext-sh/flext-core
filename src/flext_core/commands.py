"""CQRS command patterns for write operations."""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Generic, Self, cast, override
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError as PydanticValidationError,
    model_validator,
)

from flext_core.loggings import FlextLogger, FlextLoggerFactory
from flext_core.mixins import (
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimingMixin,
)
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult
from flext_core.typings import (
    CommandT,
    FlextTypes,
    QueryResultT,
    QueryT,
    ResultT,
)
from flext_core.utilities import FlextGenerators, FlextTypeGuards
from flext_core.validation import FlextValidators

# TypeVars are now properly imported from typings.py


class FlextAbstractCommand(ABC):
    """Abstract base command."""

    @abstractmethod
    def validate_command(self) -> FlextResult[None]:
        """Validate command."""
        ...


class FlextAbstractCommandHandler(ABC, Generic[CommandT, ResultT]):  # noqa: UP046
    """Abstract command handler."""

    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Get handler name."""
        ...

    @abstractmethod
    def handle(self, command: CommandT) -> FlextResult[ResultT]:
        """Handle command."""
        ...

    @abstractmethod
    def can_handle(self, command: object) -> bool:
        """Check if can handle command."""
        ...


class FlextAbstractCommandBus(ABC):
    """Abstract command bus."""

    @abstractmethod
    def send_command(self, command: FlextAbstractCommand) -> FlextResult[object]:
        """Send command."""
        ...

    @abstractmethod
    def unregister_handler(self, command_type: str) -> bool:
        """Unregister handler."""
        ...

    @abstractmethod
    def get_registered_handlers(self) -> dict[str, object]:
        """Get registered handlers."""
        ...


class FlextAbstractQueryHandler(ABC, Generic[QueryT, ResultT]):  # noqa: UP046
    """Abstract query handler."""

    @property
    @abstractmethod
    def handler_name(self) -> str:
        """Get handler name."""
        ...

    @abstractmethod
    def handle(self, query: QueryT) -> FlextResult[ResultT]:
        """Handle query."""
        ...


# FlextLogger imported for class methods only - instance methods use FlextLoggableMixin

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

    class Command(
        BaseModel,
        FlextAbstractCommand,
        FlextSerializableMixin,
        FlextLoggableMixin,
    ):
        """Base command with validation and metadata."""

        model_config = ConfigDict(
            validate_assignment=True,
            str_strip_whitespace=True,
            extra="forbid",
            frozen=True,
        )

        # Use FlextUtilities for ID generation
        command_id: str = Field(
            default_factory=FlextGenerators.generate_uuid,
            description="Unique command identifier",
        )

        command_type: str = Field(
            default="",
            description="Command type for routing",
        )

        timestamp: datetime = Field(
            default_factory=lambda: datetime.now(tz=ZoneInfo("UTC")),
            description="Command creation timestamp",
        )

        user_id: str | None = Field(
            default=None,
            description="User who initiated the command",
        )

        correlation_id: str = Field(
            default_factory=FlextGenerators.generate_uuid,
            description="Correlation ID for tracking",
        )

        @model_validator(mode="before")
        @classmethod
        def set_command_type(cls, values: dict[str, object]) -> dict[str, object]:
            """Auto-generate command_type from class name if not provided."""
            if not values.get("command_type"):
                # Convert "SampleCommand" -> "sample"
                class_name = cls.__name__
                # Remove trailing 'Command' if present
                base = class_name.removesuffix("Command")
                # Convert CamelCase/PascalCase to snake_case (createUser -> create_user)
                s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
                command_type = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
                values["command_type"] = command_type
            return values

        @classmethod
        def from_payload(
            cls,
            payload: FlextPayload[FlextTypes.Core.Dict],
        ) -> FlextResult[Self]:
            """Create command from FlextPayload with validation."""
            logger = FlextLoggerFactory.get_logger(f"{cls.__module__}.{cls.__name__}")
            logger.debug(
                "Creating command from payload",
                payload_type=payload.metadata.get("type", "unknown"),
                expected_type=cls.__name__,
            )

            expected_type = payload.metadata.get("type", "")
            if expected_type not in {cls.__name__, ""}:
                logger.warning(
                    "Payload type mismatch",
                    expected=cls.__name__,
                    actual=expected_type,
                )

            # Parse timestamp if string - handle None case
            payload_dict: dict[str, object]
            # Extract dict data with explicit type handling for pyright compatibility
            raw_data = payload.value
            if raw_data is not None:
                # Explicit cast to satisfy pyright's type checking
                payload_dict = {
                    str(k): v for k, v in cast("dict[object, object]", raw_data).items()
                }
            else:
                payload_dict = {}

            # Extract explicit parameters for Command constructor
            command_id = str(
                payload_dict.get("command_id", FlextGenerators.generate_uuid())
            )
            command_type = str(payload_dict.get("command_type", cls.__name__))

            # Handle timestamp (FlextTypes.Core.Dict only allows primitive types, não datetime)
            timestamp_raw = payload_dict.get("timestamp")
            if isinstance(timestamp_raw, str):
                timestamp = datetime.fromisoformat(timestamp_raw)
            else:
                timestamp = datetime.now(tz=ZoneInfo("UTC"))

            # Handle user_id (optional)
            user_id_raw = payload_dict.get("user_id")
            user_id = str(user_id_raw) if user_id_raw else None

            # Handle correlation_id
            correlation_id_raw = payload_dict.get("correlation_id")
            correlation_id = (
                str(correlation_id_raw)
                if correlation_id_raw
                else str(FlextGenerators.generate_uuid())
            )

            # Remove processed fields from payload_dict
            remaining_data: dict[str, object] = {
                k: v
                for k, v in payload_dict.items()
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
            validate_method = getattr(command, "validate_command", None)
            if callable(validate_method):
                validation_result = validate_method()
                if (
                    isinstance(validation_result, FlextResult)
                    and validation_result.is_failure
                ):
                    val = cast("FlextResult[object]", validation_result)
                    return FlextResult[Self].fail(
                        val.error or "Command validation failed",
                    )

            logger.info(
                "Command created from payload",
                command_type=command.command_type,
                command_id=command.command_id,
            )
            return FlextResult[Self].ok(command)

        @override
        def validate_command(self) -> FlextResult[None]:
            """Validate command using FlextValidation."""
            # Override in subclasses for custom validation
            return FlextResult[None].ok(None)

        @staticmethod
        def require_field(
            field_name: str,
            value: object,
            error_msg: str = "",
        ) -> FlextResult[None]:
            """Validate a required field with custom error."""
            if not value or (isinstance(value, str) and not value.strip()):
                msg = error_msg or f"{field_name} is required"
                return FlextResult[None].fail(msg)
            return FlextResult[None].ok(None)

        @staticmethod
        def require_email(
            email: str,
            field_name: str = "email",
        ) -> FlextResult[None]:
            """Validate email format."""
            if (
                not email
                or "@" not in email
                or "." not in email.rsplit("@", maxsplit=1)[-1]
            ):
                return FlextResult[None].fail(f"Invalid {field_name} format")
            return FlextResult[None].ok(None)

        @staticmethod
        def require_min_length(
            value: str,
            min_len: int,
            field_name: str,
        ) -> FlextResult[None]:
            """Validate minimum string length."""
            if len(value.strip()) < min_len:
                error_msg = f"{field_name} must be at least {min_len} characters"
                return FlextResult[None].fail(error_msg)
            return FlextResult[None].ok(None)

        def get_metadata(self) -> dict[str, str | None]:
            """Get command metadata."""
            return {
                "command_id": self.command_id,
                "command_type": self.command_type,
                "command_class": self.__class__.__name__,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "user_id": self.user_id,
                "correlation_id": self.correlation_id,
            }

        def to_payload(self) -> FlextPayload[dict[str, str | None]]:
            """Convert command to FlextPayload for serialization."""
            # Convert the command model to a dictionary
            command_dict = self.model_dump()

            # Ensure timestamp is serialized as ISO string
            if "timestamp" in command_dict and isinstance(
                command_dict["timestamp"], datetime
            ):
                command_dict["timestamp"] = command_dict["timestamp"].isoformat()

            # Create payload with metadata
            metadata = self.get_metadata()
            metadata["type"] = self.command_type or self.__class__.__name__.lower()

            # Create and return payload
            result = FlextPayload[dict[str, object]].create(
                data=command_dict, **metadata
            )
            return cast("FlextPayload[dict[str, str | None]]", result.value)

    # =============================================================================
    # COMMAND RESULT HELPERS - Use FlextResult directly as requested
    # =============================================================================

    # Result type alias for compatibility
    Result = FlextResult

    @staticmethod
    def create_success(data: object) -> FlextResult[object]:
        """Create successful result using FlextResult pattern."""
        return FlextResult[object].ok(data)

    @staticmethod
    def create_failure(
        error: str,
        error_code: str | None = None,
        error_data: dict[str, object] | None = None,
    ) -> FlextResult[object]:
        """Create failed result using FlextResult pattern."""
        return FlextResult[object].fail(
            error, error_code=error_code, error_data=error_data
        )

    # =============================================================================
    # COMMAND HANDLER - Base handler interface
    # =============================================================================

    class Handler(
        FlextAbstractCommandHandler[CommandT, ResultT],
        FlextLoggableMixin,
        FlextTimingMixin,
        Generic[CommandT, ResultT],
    ):
        """Base command handler interface - implements FlextAbstractCommandHandler."""

        def __init__(
            self,
            handler_name: FlextTypes.Service.ServiceName | None = None,
            handler_id: FlextTypes.Service.ServiceName | None = None,
        ) -> None:
            """Initialize handler with logging."""
            super().__init__()
            self._metrics_state: dict[str, int] | None = None
            self._handler_name = handler_name or self.__class__.__name__
            self.handler_id = handler_id or f"{self.__class__.__name__}_{id(self)}"
            # Logger now provided by FlextLoggableMixin - DRY principle applied

        @property
        @override
        def handler_name(self) -> str:
            """Get handler name - implements abstract method."""
            return self._handler_name

        def validate_command(self, command: object) -> FlextResult[None]:
            """Validate command before handling - implements abstract method."""
            # Default implementation delegates to command's validation
            validate_method = getattr(command, "validate_command", None)
            if callable(validate_method):
                result = validate_method()
                # Ensure we return FlextResult[None] type
                if hasattr(result, "success") and hasattr(result, "data"):
                    return cast("FlextResult[None]", result)
                return FlextResult[None].ok(None)
            return FlextResult[None].ok(None)

        @abstractmethod
        @override
        def handle(self, command: CommandT) -> FlextResult[ResultT]:
            """Handle the command and return result.

            Args:
                command: Command to handle

            Returns:
                FlextResult with execution result or error

            """

        @override
        def _start_timing(self) -> float:
            """Start timing operation - implements timing interface."""
            return time.perf_counter()

        def _get_execution_time_ms_rounded(self, start_time: float) -> float:
            """Get execution time in milliseconds rounded to 2 decimal places."""
            return round((time.perf_counter() - start_time) * 1000, 2)

        def process_command(self, command: CommandT) -> FlextResult[ResultT]:
            """Process command with validation, execution, and metrics collection."""
            # Initialize metrics
            self._initialize_metrics()

            # Validate command
            validation_result = self._validate_command(command)
            if validation_result.is_failure:
                return FlextResult[ResultT].fail(
                    validation_result.error or "Validation failed",
                )

            # Check capability
            if not self.can_handle(command):
                error = f"{self.handler_name} cannot process {type(command).__name__}"
                return FlextResult[ResultT].fail(error)

            # Execute command
            return self._execute_command(command)

        def _initialize_metrics(self) -> None:
            """Initialize metrics state."""
            metrics_state = getattr(self, "_metrics_state", None)
            if metrics_state is None:
                self._metrics_state = {"total": 0, "success": 0}
                metrics_state = self._metrics_state
            metrics_state["total"] = int(metrics_state.get("total", 0)) + 1

        def _validate_command(self, command: CommandT) -> FlextResult[None]:
            """Validate command using injected validator or command's own method."""
            validator = getattr(self, "_validator", None)
            if validator is not None:
                return self._validate_with_injected_validator(validator, command)
            return self._validate_with_command_method(command)

        def _validate_with_injected_validator(
            self,
            validator: object,
            command: object,
        ) -> FlextResult[None]:
            """Validate using injected validator."""
            try:
                validate_method = getattr(validator, "validate_message", None)
                if callable(validate_method):
                    validated_result = validate_method(command)
                    if (
                        isinstance(validated_result, FlextResult)
                        and validated_result.is_failure
                    ):
                        val = cast("FlextResult[object]", validated_result)
                        return FlextResult[None].fail(val.error or "Validation failed")
            except (
                TypeError,
                ValueError,
                AttributeError,
                PydanticValidationError,
            ) as e:
                return FlextResult[None].fail(f"Command validation failed: {e}")
            return FlextResult[None].ok(None)

        def _validate_with_command_method(self, command: object) -> FlextResult[None]:
            """Validate using command's own validation method."""
            validate_cmd_method = getattr(command, "validate_command", None)
            if callable(validate_cmd_method):
                validation_result = validate_cmd_method()
                if (
                    isinstance(validation_result, FlextResult)
                    and validation_result.is_failure
                ):
                    val = cast("FlextResult[object]", validation_result)
                    return FlextResult[None].fail(
                        val.error or "Command validation failed"
                    )
            return FlextResult[None].ok(None)

        def _execute_command(self, command: CommandT) -> FlextResult[ResultT]:
            """Execute command and update metrics."""
            try:
                result: FlextResult[ResultT] = self.handle(command)
                self._update_metrics(result)
                return result
            except (RuntimeError, OSError) as e:
                return FlextResult[ResultT].fail(f"Command processing failed: {e}")

        def _update_metrics(self, result: FlextResult[ResultT]) -> None:
            """Update metrics after command execution."""
            # Simple metrics update when collector injected
            metrics_collector = getattr(self, "_metrics", None)
            if metrics_collector is not None:
                get_metrics_method = getattr(metrics_collector, "get_metrics", None)
                if callable(get_metrics_method):
                    _ = get_metrics_method()

            # Update success counter only if successful
            if result.is_success and self._metrics_state is not None:
                self._metrics_state["success"] = (
                    int(self._metrics_state.get("success", 0)) + 1
                )

        @override
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
            orig_bases = getattr(self, "__orig_bases__", None)
            if orig_bases is not None:
                for base in orig_bases:
                    args = getattr(base, "__args__", None)
                    if args is not None and len(args) >= 1:
                        expected_type = base.__args__[0]
                        # Use BASE type guard directly - MAXIMIZE base functionality
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

        def handle_command(self, command: CommandT) -> FlextResult[ResultT]:
            """Alias for handle used by some tests/utilities."""
            return self.handle(command)

        def get_command_type(self) -> str:
            """Return command type name; subclasses typically override."""
            return self.__class__.__name__

        def execute(self, command: CommandT) -> FlextResult[ResultT]:
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
                return FlextResult[ResultT].fail(error_msg)

            # Validate the command's data
            validation_result = self.validate_command(command)
            if validation_result.is_failure:
                self.logger.warning(
                    "Command validation failed",
                    command_type=type(command).__name__,
                    error=validation_result.error,
                )
                return FlextResult[ResultT].fail(
                    validation_result.error or "Validation failed",
                )

            start_time = self._start_timing()

            try:
                result: FlextResult[ResultT] = self.handle(command)

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
        """Command bus for routing and executing commands."""

        def __init__(self) -> None:
            """Initialize command bus."""
            super().__init__()
            # Handlers registry: command type -> handler instance
            # Use object for values because handlers can be different Handler generics
            self._handlers: dict[object, object] = {}
            # Middleware pipeline
            self._middleware: list[object] = []
            # Execution counter
            self._execution_count: int = 0
            # Logger instance

            self._logger: FlextLogger = FlextLoggerFactory.get_logger(
                f"{self.__module__}.{self.__class__.__name__}",
            )
            # Optional validator
            self._validator: object | None = None

        # Core execution/validation methods moved below to keep a single implementation

        # Single initializer above handles setup for Bus; duplicate initializer removed

        def register_handler(
            self,
            *args: object,
        ) -> None:
            """Register command handler (compatible signature).

            Supports both (command_type: str, handler) and (handler) forms.
            """
            one = 1
            two = 2
            if len(args) == one:
                handler = args[0]
                if handler is None:
                    msg = "Handler cannot be None"
                    raise TypeError(msg)
                handle_method = getattr(handler, "handle", None)
                if not callable(handle_method):
                    msg = "Invalid handler: must have callable 'handle' method"
                    raise TypeError(msg)
                key = getattr(handler, "handler_id", handler.__class__.__name__)
                if key in self._handlers:
                    self.logger.warning(
                        "Handler already registered",
                        command_type=str(key),
                        existing_handler=self._handlers[key].__class__.__name__,
                    )
                    return
                self._handlers[key] = handler
                self.logger.info(
                    "Handler registered successfully",
                    command_type=str(key),
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return

            if len(args) == two:
                command_type_obj, handler = args
                if command_type_obj is None:
                    msg = "Command type cannot be None"
                    raise ValueError(msg)
                if handler is None:
                    msg = "Handler cannot be None"
                    raise ValueError(msg)
                name_attr = getattr(command_type_obj, "__name__", None)
                key = name_attr if name_attr is not None else str(command_type_obj)
                if key in self._handlers:
                    self.logger.warning(
                        "Handler already registered",
                        command_type=key,
                        existing_handler=self._handlers[key].__class__.__name__,
                    )
                    return
                self._handlers[key] = handler
                self.logger.info(
                    "Handler registered successfully",
                    command_type=key,
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return

            msg = "register_handler() takes 1 or 2 positional arguments"
            raise TypeError(msg)

        def register_handler_flexible(
            self,
            handler_or_command_type: object | type[CommandT],
            handler: FlextCommands.Handler[CommandT, ResultT] | None = None,
        ) -> FlextResult[None]:
            """Register handler with flexible arguments (compatibility).

            Args:
                handler_or_command_type: Handler instance or command type
                handler: Handler instance (when first arg is a command type)

            Returns:
                FlextResult indicating registration success

            """
            # Handle a single argument case (just the handler)
            if handler is None:
                handle_method = getattr(handler_or_command_type, "handle", None)
                if not callable(handle_method):
                    return FlextResult[None].fail(
                        "Invalid handler: must have callable 'handle' method",
                    )
                handler_obj = handler_or_command_type
                # Use handler_id as a key for uniqueness
                key = getattr(handler_obj, "handler_id", handler_obj.__class__.__name__)
                self._handlers[key] = handler_obj

                self.logger.info(
                    "Handler registered successfully",
                    handler_id=key,
                    handler_type=handler_obj.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return FlextResult[None].ok(None)
            # Handle two arguments case (command_type, handler)
            command_type = handler_or_command_type
            handler_obj = handler

            # Use BASE validators directly - MAXIMIZE base usage
            if not FlextValidators.is_not_none(command_type):
                return FlextResult[None].fail("Command type cannot be None")

            if not FlextValidators.is_not_none(handler_obj):
                return FlextResult[None].fail("Handler cannot be None")

            # Check if already registered
            if command_type in self._handlers:
                command_name = getattr(command_type, "__name__", str(command_type))
                self.logger.warning(
                    "Handler already registered",
                    command_type=command_name,
                    existing_handler=self._handlers[command_type].__class__.__name__,
                )
                return FlextResult[None].fail(
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

            return FlextResult[None].ok(None)

        # ...existing code...

        def _log_command_execution(self, command: object, command_type: type) -> None:
            """Log command execution start."""
            self.logger.info(
                "Executing command via bus",
                command_type=command_type.__name__,
                command_id=getattr(command, "command_id", "unknown"),
                execution_count=self._execution_count,
            )

        def _find_command_handler(self, command: object) -> object | None:
            """Find a handler that can handle this command."""
            for registered_handler in self._handlers.values():
                can_handle_method = getattr(registered_handler, "can_handle", None)
                if callable(can_handle_method) and can_handle_method(command):
                    return registered_handler
            return None

        def _handle_no_handler_found(self, command_type: type) -> FlextResult[object]:
            """Handle a case when no handler is found."""
            handler_names = [h.__class__.__name__ for h in self._handlers.values()]
            self.logger.error(
                "No handler found",
                command_type=command_type.__name__,
                registered_handlers=handler_names,
            )
            return FlextResult[object].fail(
                f"No handler found for {command_type.__name__}"
            )

        def _apply_middleware(
            self,
            command: object,
            handler: object,
        ) -> FlextResult[None]:
            """Apply a middleware pipeline."""
            for i, middleware in enumerate(self._middleware):
                self.logger.debug(
                    "Applying middleware",
                    middleware_index=i,
                    middleware_type=type(middleware).__name__,
                )
                process_method = getattr(middleware, "process", None)
                if callable(process_method):
                    result = process_method(command, handler)
                    if isinstance(result, FlextResult) and result.is_failure:
                        val = cast("FlextResult[object]", result)
                        self.logger.warning(
                            "Middleware rejected command",
                            middleware_type=type(middleware).__name__,
                            error=val.error or "Unknown error",
                        )
                        error_msg = str(val.error or "Middleware rejected command")
                        return FlextResult[None].fail(error_msg)
            return FlextResult[None].ok(None)

        def _execute_handler(
            self,
            handler: object,
            command: object,
        ) -> FlextResult[object]:
            """Execute the command handler."""
            self.logger.debug(
                "Delegating to handler",
                handler_type=handler.__class__.__name__,
            )

            # Try different handler methods in order of preference
            handler_methods = ["process_command", "execute", "handle"]

            for method_name in handler_methods:
                method = getattr(handler, method_name, None)
                if callable(method):
                    result = method(command)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

            # No valid handler method found
            return FlextResult[object].fail(
                "Handler has no callable process_command, execute, or handle method",
            )

        def execute(self, command: FlextAbstractCommand) -> FlextResult[object]:
            """Find handler, apply middleware and execute command via handler."""
            self._execution_count = int(self._execution_count) + 1
            command_type = type(command)
            self._log_command_execution(command, command_type)

            handler = self._find_command_handler(command)
            if handler is None:
                return self._handle_no_handler_found(command_type)

            # Apply middleware pipeline
            mw_result = self._apply_middleware(command, handler)
            if mw_result.is_failure:
                # propagate middleware error as object failure
                return FlextResult[object].fail(
                    cast("FlextResult[object]", mw_result).error
                    or "Middleware rejected command"
                )

            # Execute the handler
            return self._execute_handler(handler, command)

        def add_middleware(self, middleware: object) -> None:
            """Add middleware to processing pipeline."""
            self._middleware.append(middleware)

        def get_all_handlers(self) -> list[object]:
            """Get all registered handlers."""
            return list(self._handlers.values())

        def find_handler(self, command: object) -> object | None:
            """Find handler for command type."""
            for handler in self._handlers.values():
                can_handle_method = getattr(handler, "can_handle", None)
                if callable(can_handle_method) and can_handle_method(command):
                    return handler
            return None

        @override
        def unregister_handler(self, command_type: str) -> bool:
            """Unregister command handler - implements abstract method."""
            # Find handler by command type string
            for key, _handler in list(self._handlers.items()):
                key_name = getattr(key, "__name__", None)
                if (key_name is not None and key_name == command_type) or str(
                    key,
                ) == command_type:
                    del self._handlers[key]
                    return True
            return False

        @override
        def send_command(self, command: FlextAbstractCommand) -> FlextResult[object]:
            """Send command - implements abstract method."""
            return self.execute(command)

        @override
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
                    @override
                    def handle(self, command: object) -> FlextResult[object]:
                        result = func(command)
                        if isinstance(result, FlextResult):
                            return cast("FlextResult[object]", result)
                        return FlextResult[object].ok(result)

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
        FlextSerializableMixin,
    ):
        """Base query for read operations without side effects."""

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
        )

        query_id: str | None = None
        query_type: str | None = None
        page_size: int = 100
        page_number: int = 1
        sort_by: str | None = None
        sort_order: str = "asc"

        def validate_query(self) -> FlextResult[None]:
            """Validate a query with business logic."""
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
                return FlextResult[None].fail(
                    f"Query validation failed: {error_message}",
                )

            return FlextResult[None].ok(None)

        # Mixin methods are now available through proper inheritance:
        # - is_valid property (from FlextValidatableMixin)
        # - validation_errors property (from FlextValidatableMixin)
        # - has_validation_errors() method (from FlextValidatableMixin)
        # - to_dict_basic() method (from FlextSerializableMixin)
        # - All serialization methods (from FlextSerializableMixin)

    class QueryHandler(
        FlextAbstractQueryHandler[QueryT, QueryResultT],
        Generic[QueryT, QueryResultT],
    ):
        """Base query handler built on abstract base.

        Provides default implementations using centralized base.
        """

        def __init__(self, handler_name: str | None = None) -> None:
            """Initialize query handler with optional name."""
            self._handler_name = handler_name or self.__class__.__name__

        @property
        @override
        def handler_name(self) -> str:
            """Get handler name for this query handler."""
            return self._handler_name

        def can_handle(self, query: QueryT) -> bool:
            """Return True for generic handler capability check."""
            _ = query
            return True

        def validate_query(self, query: QueryT) -> FlextResult[None]:
            """Validate a query object using its own validation method if available."""
            validate_method = getattr(query, "validate_query", None)
            if callable(validate_method):
                result = validate_method()
                if isinstance(result, FlextResult):
                    return cast("FlextResult[None]", result)
            return FlextResult[None].ok(None)

        @abstractmethod
        @override
        def handle(self, query: QueryT) -> FlextResult[QueryResultT]:
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
            @override
            def handle(self, command: object) -> FlextResult[object]:
                result = handler_func(command)
                if isinstance(result, FlextResult):
                    return cast("FlextResult[object]", result)
                return FlextResult[object].ok(result)

        return SimpleHandler()


# =============================================================================
# MODEL REBUILDS - Resolve forward references for Pydantic
# =============================================================================

# Rebuild nested models to resolve forward references after import
try:  # Defensive: avoid failing import-time on environments without Pydantic
    _types_ns: dict[str, object] = {
        "FlextTypes.Service.ServiceName": str,
        "TUserId": str,
        "TCorrelationId": str,
        "FlextTypes.Domain.EntityId": str,
        "FlextTypes.Core.Dict": dict[str, object],
    }
    # Pydantic v2 expects _types_namespace keyword
    FlextCommands.Command.model_rebuild(_types_namespace=_types_ns)
    FlextCommands.Query.model_rebuild(_types_namespace=_types_ns)
except Exception as _e:  # noqa: BLE001 - import-time best effort
    # Best-effort logging at import time without failing import
    try:
        _logger = FlextLoggerFactory.get_logger(__name__)
        _logger.debug(
            "Pydantic model_rebuild skipped",
            error=str(_e),
        )
    except Exception as _inner_e:  # noqa: BLE001 - ignore logging init errors
        # Last-resort: swallow to avoid import-time hard failure. No print.
        from contextlib import suppress as _suppress

        with _suppress(Exception):
            _ = _inner_e

# Export API
__all__: list[str] = ["FlextCommands"]
