"""FLEXT Core Command Pattern - Unified Command System.

Enterprise-grade command pattern implementation with type safety,
validation, and standardized error handling.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Protocol
from typing import TypeVar

from flext_core.patterns.typedefs import FlextCommandId
from flext_core.patterns.typedefs import FlextCommandType
from flext_core.patterns.typedefs import FlextHandlerId
from flext_core.result import FlextResult

# =============================================================================
# TYPE VARIABLES - Generic command handling
# =============================================================================

TCommand = TypeVar("TCommand", bound="FlextCommand")
TResult = TypeVar("TResult")

# =============================================================================
# COMMAND INTERFACES - Abstract base for all commands
# =============================================================================


class FlextCommand(ABC):
    """Base class for all FLEXT commands.

    Commands represent business operations that can be executed
    with proper validation, logging, and error handling.
    """

    def __init__(
        self,
        command_id: FlextCommandId | None = None,
        command_type: FlextCommandType | None = None,
    ) -> None:
        """Initialize command with optional ID and type."""
        self.command_id = command_id or FlextCommandId(
            f"{self.__class__.__name__}_{id(self)}",
        )
        self.command_type = command_type or FlextCommandType(
            self.__class__.__name__,
        )

    @abstractmethod
    def validate(self) -> FlextResult[None]:
        """Validate command data before execution.

        Returns:
            FlextResult indicating validation success or failure

        """

    def get_metadata(self) -> dict[str, Any]:
        """Get command metadata for logging and tracing."""
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "command_class": self.__class__.__name__,
        }

    def get_command_metadata(self) -> dict[str, Any]:
        """Get command metadata (alias for get_metadata)."""
        return self.get_metadata()

    def validate_command(self) -> FlextResult[None]:
        """Validate command (alias for validate for test compatibility)."""
        return self.validate()

    def get_payload(self) -> dict[str, Any]:
        """Get command payload data.

        Returns:
            Dictionary containing command payload data

        """
        return {}


class FlextCommandHandler[TCommand: "FlextCommand", TResult](ABC):
    """Base class for command handlers.

    Handlers process specific command types with proper error handling,
    logging, and result management.
    """

    def __init__(self, *, handler_id: str | None = None) -> None:
        """Initialize command handler with optional ID."""
        self.handler_id = FlextHandlerId(
            handler_id or f"{self.__class__.__name__}_{id(self)}",
        )

    @abstractmethod
    def can_handle(self, command: FlextCommand) -> bool:
        """Check if this handler can process the given command.

        Args:
            command: Command to check

        Returns:
            True if handler can process command

        """

    @abstractmethod
    def handle(self, command: TCommand) -> FlextResult[TResult]:
        """Handle the command and return result.

        Args:
            command: Command to process

        Returns:
            FlextResult with success data or error information

        """

    def handle_command(self, command: TCommand) -> FlextResult[TResult]:
        """Handle command (alias for handle for test compatibility)."""
        return self.handle(command)

    def process_command(self, command: FlextCommand) -> FlextResult[TResult]:
        """Process command with validation and error handling."""
        # Validate command first
        validation_result = command.validate()
        if validation_result.is_failure:
            return FlextResult.fail(
                f"Command validation failed: {validation_result.error}",
            )

        # Check if can handle
        if not self.can_handle(command):
            return FlextResult.fail(
                f"Handler cannot process command of type "
                f"{command.command_type}",
            )

        # Handle the command
        try:
            return self.handle(command)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"Command processing failed: {e!s}")

    def get_handler_info(self) -> dict[str, Any]:
        """Get handler information for debugging."""
        return {
            "handler_class": self.__class__.__name__,
            "handler_id": str(self.handler_id),
        }


# =============================================================================
# COMMAND RESULT - Standardized command results
# =============================================================================


class FlextCommandResult[TResult]:
    """Standardized result for command execution.

    Wraps FlextResult with command-specific metadata and context.
    """

    def __init__(
        self,
        result: FlextResult[TResult],
        command: FlextCommand,
        handler_info: dict[str, Any] | None = None,
    ) -> None:
        """Initialize command result.

        Args:
            result: Underlying FlextResult
            command: Command that was executed
            handler_info: Optional handler information

        """
        self._result = result
        self._command = command
        self._handler_info = handler_info or {}

    @property
    def is_success(self) -> bool:
        """Check if command execution was successful."""
        return self._result.is_success

    @property
    def is_failure(self) -> bool:
        """Check if command execution failed."""
        return self._result.is_failure

    @property
    def data(self) -> FlextCommandResult[TResult]:
        """Get command result wrapped in this result object."""
        return self

    @property
    def result(self) -> TResult | None:
        """Get command result data."""
        return self._result.data

    @property
    def error(self) -> str | None:
        """Get command error message."""
        return self._result.error

    @property
    def command(self) -> FlextCommand:
        """Get the command that was executed."""
        return self._command

    @property
    def command_metadata(self) -> dict[str, Any]:
        """Get command metadata."""
        return self._command.get_metadata()

    @property
    def handler_metadata(self) -> dict[str, Any]:
        """Get handler metadata."""
        return self._handler_info

    def get_result_metadata(self) -> dict[str, Any]:
        """Get result metadata for testing."""
        return {
            "command_id": self._command.command_id,
            "command_type": self._command.command_type,
            "is_success": self.is_success,
            "execution_time": "mock_time",  # Mock for testing
        }

    def get_full_context(self) -> dict[str, Any]:
        """Get complete execution context."""
        return {
            "command": self.command_metadata,
            "handler": self.handler_metadata,
            "success": self.is_success,
            "error": self.error,
        }

    @classmethod
    def success(
        cls,
        command: FlextCommand,
        result_data: TResult,
    ) -> FlextCommandResult[TResult]:
        """Create successful command result."""
        return cls(FlextResult.ok(result_data), command)

    @classmethod
    def failure(
        cls,
        command: FlextCommand,
        error_message: str,
    ) -> FlextCommandResult[TResult]:
        """Create failed command result."""
        return cls(FlextResult.fail(error_message), command)


# =============================================================================
# COMMAND BUS - Central command processing
# =============================================================================


class FlextCommandBus:
    """Central bus for command processing.

    Manages command handlers and provides unified command execution
    with proper error handling and logging.
    """

    def __init__(self) -> None:
        """Initialize empty command bus."""
        self._handlers: list[FlextCommandHandler[Any, Any]] = []

    def register_handler(
        self,
        handler: object,
    ) -> FlextResult[None]:
        """Register a command handler.

        Args:
            handler: Handler to register

        Returns:
            FlextResult indicating registration success

        """
        if not isinstance(handler, FlextCommandHandler):
            return FlextResult.fail(
                "Handler must be instance of FlextCommandHandler",
            )

        self._handlers.append(handler)
        return FlextResult.ok(None)

    def execute(
        self,
        command: FlextCommand,
    ) -> FlextCommandResult[Any]:
        """Execute a command using appropriate handler.

        Args:
            command: Command to execute

        Returns:
            FlextCommandResult with execution results

        """
        # Validate command first
        validation_result = command.validate()
        if validation_result.is_failure:
            validation_error_result: FlextResult[Any] = FlextResult.fail(
                f"Command validation failed: {validation_result.error}",
            )
            return FlextCommandResult(validation_error_result, command)

        # Find appropriate handler
        handler = self._find_handler(command)
        if handler is None:
            no_handler_result: FlextResult[Any] = FlextResult.fail(
                f"No handler found for command {command.command_type}",
            )
            return FlextCommandResult(no_handler_result, command)

        # Execute command
        try:
            execution_result = handler.handle(command)
            return FlextCommandResult(
                execution_result,
                command,
                handler.get_handler_info(),
            )
        except (ValueError, TypeError, AttributeError) as e:
            execution_error_result: FlextResult[Any] = FlextResult.fail(
                f"Command execution failed: {e!s}",
            )
            return FlextCommandResult(
                execution_error_result,
                command,
                handler.get_handler_info(),
            )
        except Exception as e:  # noqa: BLE001
            unexpected_error_result: FlextResult[Any] = FlextResult.fail(
                f"Unexpected command error: {e!s}",
            )
            return FlextCommandResult(
                unexpected_error_result,
                command,
                handler.get_handler_info(),
            )

    def _find_handler(
        self,
        command: FlextCommand,
    ) -> FlextCommandHandler[Any, Any] | None:
        """Find handler that can process the command."""
        for handler in self._handlers:
            if handler.can_handle(command):
                return handler
        return None

    def get_registered_handlers(self) -> list[dict[str, Any]]:
        """Get information about registered handlers."""
        return [handler.get_handler_info() for handler in self._handlers]

    def get_all_handlers(self) -> list[FlextCommandHandler[Any, Any]]:
        """Get all registered handlers."""
        return self._handlers.copy()

    def find_handler(
        self,
        command: FlextCommand,
    ) -> FlextCommandHandler[Any, Any] | None:
        """Find handler for command (public interface)."""
        return self._find_handler(command)

    def register(
        self,
        handler: FlextCommandHandler[Any, Any],
    ) -> FlextResult[None]:
        """Register handler (alias for register_handler)."""
        return self.register_handler(handler)


# =============================================================================
# COMMAND PROTOCOL - For type checking
# =============================================================================


class CommandExecutor(Protocol):
    """Protocol for command execution."""

    def execute(self, command: FlextCommand) -> FlextCommandResult[Any]:
        """Execute a command."""
        ...


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    "CommandExecutor",
    "FlextCommand",
    "FlextCommandBus",
    "FlextCommandHandler",
    "FlextCommandResult",
    "TCommand",
    "TResult",
]
