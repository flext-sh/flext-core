"""Comprehensive tests for FLEXT command pattern."""

from __future__ import annotations

from typing import Any

from flext_core.patterns.commands import FlextCommand
from flext_core.patterns.commands import FlextCommandBus
from flext_core.patterns.commands import FlextCommandHandler
from flext_core.patterns.commands import FlextCommandResult
from flext_core.patterns.typedefs import FlextCommandId
from flext_core.patterns.typedefs import FlextCommandType
from flext_core.patterns.typedefs import FlextHandlerId
from flext_core.result import FlextResult

# =============================================================================
# TEST COMMAND IMPLEMENTATIONS
# =============================================================================


class CreateUserCommand(FlextCommand):
    """Test command for creating users."""

    def __init__(
        self,
        username: str,
        email: str,
        command_id: FlextCommandId | None = None,
    ) -> None:
        """Initialize create user command with user details."""
        super().__init__(
            command_id=command_id,
            command_type=FlextCommandType("create_user"),
        )
        self.username = username
        self.email = email

    def get_payload(self) -> dict[str, Any]:
        """Get command payload."""
        return {
            "username": self.username,
            "email": self.email,
        }

    def validate(self) -> FlextResult[None]:
        """Validate command data."""
        if not self.username:
            return FlextResult.fail("Username is required")
        if not self.email:
            return FlextResult.fail("Email is required")
        if "@" not in self.email:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(None)

    def validate_command(self) -> FlextResult[None]:
        """Validate command data (alias for validate)."""
        return self.validate()


class UpdateUserCommand(FlextCommand):
    """Test command for updating users."""

    def __init__(
        self,
        user_id: str,
        updates: dict[str, Any],
        command_id: FlextCommandId | None = None,
    ) -> None:
        """Initialize update user command with user ID and updates."""
        super().__init__(
            command_id=command_id,
            command_type=FlextCommandType("update_user"),
        )
        self.user_id = user_id
        self.updates = updates

    def get_payload(self) -> dict[str, Any]:
        """Get command payload."""
        return {
            "user_id": self.user_id,
            "updates": self.updates,
        }

    def validate(self) -> FlextResult[None]:
        """Validate command data."""
        if not self.user_id:
            return FlextResult.fail("User ID is required")
        if not self.updates:
            return FlextResult.fail("Updates are required")
        return FlextResult.ok(None)

    def validate_command(self) -> FlextResult[None]:
        """Validate command data (alias for validate)."""
        return self.validate()


class FailingCommand(FlextCommand):
    """Test command that always fails validation."""

    def __init__(self) -> None:
        """Initialize failing command."""
        super().__init__(command_type=FlextCommandType("failing"))

    def get_payload(self) -> dict[str, Any]:
        """Get command payload."""
        return {}

    def validate(self) -> FlextResult[None]:
        """Fail validation intentionally."""
        return FlextResult.fail("This command always fails")

    def validate_command(self) -> FlextResult[None]:
        """Fail validation intentionally (alias for validate)."""
        return self.validate()


# =============================================================================
# TEST COMMAND HANDLER IMPLEMENTATIONS
# =============================================================================


class CreateUserCommandHandler(
    FlextCommandHandler[CreateUserCommand, dict[str, Any]],
):
    """Test handler for CreateUserCommand."""

    def __init__(self) -> None:
        """Initialize create user command handler."""
        super().__init__(handler_id="create_user_handler")
        self.created_users: list[dict[str, Any]] = []

    def get_command_type(self) -> FlextCommandType:
        """Get command type this handler processes."""
        return FlextCommandType("create_user")

    def can_handle(self, command: FlextCommand) -> bool:
        """Check if can handle command."""
        return isinstance(
            command,
            CreateUserCommand,
        ) and command.command_type == FlextCommandType("create_user")

    def handle(
        self,
        command: CreateUserCommand,
    ) -> FlextResult[dict[str, Any]]:
        """Handle the create user command."""
        user_data = {
            "id": f"user_{len(self.created_users) + 1}",
            "username": command.username,
            "email": command.email,
        }
        self.created_users.append(user_data)

        return FlextResult.ok(user_data)

    def handle_command(
        self,
        command: CreateUserCommand,
    ) -> FlextResult[dict[str, Any]]:
        """Handle the create user command (alias for handle)."""
        return self.handle(command)


class UpdateUserCommandHandler(
    FlextCommandHandler[UpdateUserCommand, dict[str, Any]],
):
    """Test handler for UpdateUserCommand."""

    def __init__(self) -> None:
        """Initialize update user command handler."""
        super().__init__(handler_id="update_user_handler")
        self.updated_users: dict[str, dict[str, Any]] = {}

    def get_command_type(self) -> FlextCommandType:
        """Get command type this handler processes."""
        return FlextCommandType("update_user")

    def can_handle(self, command: FlextCommand) -> bool:
        """Check if can handle command."""
        return isinstance(
            command,
            UpdateUserCommand,
        ) and command.command_type == FlextCommandType("update_user")

    def handle(
        self,
        command: UpdateUserCommand,
    ) -> FlextResult[dict[str, Any]]:
        """Handle the update user command."""
        if command.user_id not in self.updated_users:
            self.updated_users[command.user_id] = {}

        self.updated_users[command.user_id].update(command.updates)

        result_data = {
            "user_id": command.user_id,
            "updated_fields": list(command.updates.keys()),
        }

        return FlextResult.ok(result_data)

    def handle_command(
        self,
        command: UpdateUserCommand,
    ) -> FlextResult[dict[str, Any]]:
        """Handle the update user command (alias for handle)."""
        return self.handle(command)


class FailingCommandHandler(FlextCommandHandler[FailingCommand, None]):
    """Test handler that always fails."""

    def get_command_type(self) -> FlextCommandType:
        """Get command type this handler processes."""
        return FlextCommandType("failing")

    def can_handle(self, command: FlextCommand) -> bool:
        """Check if can handle command."""
        return isinstance(
            command,
            FailingCommand,
        ) and command.command_type == FlextCommandType("failing")

    def handle(self, command: FailingCommand) -> FlextResult[None]:  # noqa: ARG002
        """Fail to handle command intentionally."""
        return FlextResult.fail("Handler processing failed")

    def handle_command(self, command: FailingCommand) -> FlextResult[None]:
        """Fail to handle command intentionally (alias for handle)."""
        return self.handle(command)


# =============================================================================
# TEST FLEXT COMMAND
# =============================================================================


class TestFlextCommand:
    """Test FlextCommand functionality."""

    def test_command_creation_with_auto_id(self) -> None:
        """Test creating command with auto-generated ID."""
        command = CreateUserCommand("john_doe", "john@example.com")

        assert command.command_id is not None
        assert command.command_type == FlextCommandType("create_user")
        assert command.username == "john_doe"
        assert command.email == "john@example.com"

    def test_command_creation_with_custom_id(self) -> None:
        """Test creating command with custom ID."""
        command_id = FlextCommandId("custom_cmd_123")
        command = CreateUserCommand(
            "jane_doe",
            "jane@example.com",
            command_id=command_id,
        )

        assert command.command_id == command_id
        assert command.command_type == FlextCommandType("create_user")

    def test_get_payload(self) -> None:
        """Test getting command payload."""
        command = CreateUserCommand("test_user", "test@example.com")
        payload = command.get_payload()

        assert payload["username"] == "test_user"
        assert payload["email"] == "test@example.com"

    def test_validate_command_success(self) -> None:
        """Test successful command validation."""
        command = CreateUserCommand("valid_user", "valid@example.com")
        result = command.validate_command()

        assert result.is_success is True

    def test_validate_command_failure_no_username(self) -> None:
        """Test command validation failure for missing username."""
        command = CreateUserCommand("", "test@example.com")
        result = command.validate_command()

        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "username is required" in result.error.lower()

    def test_validate_command_failure_no_email(self) -> None:
        """Test command validation failure for missing email."""
        command = CreateUserCommand("test_user", "")
        result = command.validate_command()

        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "email is required" in result.error.lower()

    def test_validate_command_failure_invalid_email(self) -> None:
        """Test command validation failure for invalid email."""
        command = CreateUserCommand("test_user", "invalid_email")
        result = command.validate_command()

        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "invalid email" in result.error.lower()

    def test_get_command_metadata(self) -> None:
        """Test getting command metadata."""
        command = CreateUserCommand("test_user", "test@example.com")
        metadata = command.get_command_metadata()

        assert "command_id" in metadata
        assert "command_type" in metadata
        assert "command_class" in metadata
        assert metadata["command_class"] == "CreateUserCommand"


# =============================================================================
# TEST FLEXT COMMAND HANDLER
# =============================================================================


class TestFlextCommandHandler:
    """Test FlextCommandHandler functionality."""

    def test_handler_creation(self) -> None:
        """Test creating command handler."""
        handler = CreateUserCommandHandler()

        assert handler.handler_id == FlextHandlerId("create_user_handler")
        assert handler.get_command_type() == FlextCommandType("create_user")

    def test_can_handle_correct_command_type(self) -> None:
        """Test can_handle with correct command type."""
        handler = CreateUserCommandHandler()
        command = CreateUserCommand("test", "test@example.com")

        assert handler.can_handle(command) is True

    def test_can_handle_wrong_command_type(self) -> None:
        """Test can_handle with wrong command type."""
        handler = CreateUserCommandHandler()
        command = UpdateUserCommand("123", {"name": "new_name"})

        assert handler.can_handle(command) is False

    def test_can_handle_non_command_object(self) -> None:
        """Test can_handle with non-command object."""
        handler = CreateUserCommandHandler()

        assert handler.can_handle("not_a_command") is False

    def test_handle_command_success(self) -> None:
        """Test successful command handling."""
        handler = CreateUserCommandHandler()
        command = CreateUserCommand("john", "john@example.com")

        result = handler.handle_command(command)

        assert result.is_success is True
        assert result.data is not None
        assert result.data["username"] == "john"
        assert result.data["email"] == "john@example.com"
        assert "id" in result.data

    def test_process_command_success(self) -> None:
        """Test complete command processing flow."""
        handler = CreateUserCommandHandler()
        command = CreateUserCommand("jane", "jane@example.com")

        result = handler.process_command(command)

        assert result.is_success is True
        assert len(handler.created_users) == 1

    def test_process_command_validation_failure(self) -> None:
        """Test processing with command validation failure."""
        handler = CreateUserCommandHandler()
        command = CreateUserCommand("", "invalid")  # Invalid command

        result = handler.process_command(command)

        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "validation failed" in result.error.lower()

    def test_process_command_cannot_handle(self) -> None:
        """Test processing command that cannot be handled."""
        handler = CreateUserCommandHandler()
        wrong_command = UpdateUserCommand("123", {"name": "test"})

        result = handler.process_command(wrong_command)

        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "cannot process" in result.error.lower()

    def test_process_command_handling_failure(self) -> None:
        """Test processing when handler fails."""
        handler = FailingCommandHandler()
        command = FailingCommand()

        result = handler.process_command(command)

        assert result.is_failure is True


# =============================================================================
# TEST FLEXT COMMAND BUS
# =============================================================================


class TestFlextCommandBus:
    """Test FlextCommandBus functionality."""

    def test_command_bus_creation(self) -> None:
        """Test creating command bus."""
        bus = FlextCommandBus()

        assert len(bus.get_all_handlers()) == 0

    def test_register_handler_success(self) -> None:
        """Test successful handler registration."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()

        result = bus.register_handler(handler)

        assert result.is_success is True
        assert len(bus.get_all_handlers()) == 1

    def test_register_invalid_handler(self) -> None:
        """Test registering invalid handler."""
        bus = FlextCommandBus()

        result = bus.register_handler("not_a_handler")

        assert result.is_failure is True

    def test_execute_command_success(self) -> None:
        """Test successful command execution."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()
        bus.register_handler(handler)

        command = CreateUserCommand("alice", "alice@example.com")
        result = bus.execute(command)

        assert result.is_success is True
        assert result.data is not None
        assert result.data.result is not None
        assert result.data.result["username"] == "alice"

    def test_execute_command_no_handler(self) -> None:
        """Test executing command with no registered handler."""
        bus = FlextCommandBus()
        command = CreateUserCommand("bob", "bob@example.com")

        result = bus.execute(command)

        assert result.is_failure is True
        assert result.error is not None
        assert result.error
        assert "no handler found" in result.error.lower()

    def test_execute_command_validation_failure(self) -> None:
        """Test executing invalid command."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()
        bus.register_handler(handler)

        command = CreateUserCommand("", "")  # Invalid command
        result = bus.execute(command)

        assert result.is_failure is True

    def test_find_handler_for_command(self) -> None:
        """Test finding handler for command."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()
        bus.register_handler(handler)

        command = CreateUserCommand("test", "test@example.com")
        found_handler = bus.find_handler(command)

        assert found_handler == handler

    def test_find_handler_not_found(self) -> None:
        """Test finding handler when none exists."""
        bus = FlextCommandBus()
        command = CreateUserCommand("test", "test@example.com")

        found_handler = bus.find_handler(command)

        assert found_handler is None

    def test_get_all_handlers(self) -> None:
        """Test getting all registered handlers."""
        bus = FlextCommandBus()
        handler1 = CreateUserCommandHandler()
        handler2 = UpdateUserCommandHandler()

        bus.register_handler(handler1)
        bus.register_handler(handler2)

        all_handlers = bus.get_all_handlers()

        assert len(all_handlers) == 2
        assert handler1 in all_handlers
        assert handler2 in all_handlers


# =============================================================================
# TEST FLEXT COMMAND RESULT
# =============================================================================


class TestFlextCommandResult:
    """Test FlextCommandResult functionality."""

    def test_success_result_creation(self) -> None:
        """Test creating successful command result."""
        command = CreateUserCommand("test", "test@example.com")
        result_data = {"id": "123", "username": "test"}

        command_result = FlextCommandResult.success(command, result_data)

        assert command_result.is_success is True
        assert command_result.command == command
        assert command_result.result == result_data
        assert command_result.error is None

    def test_failure_result_creation(self) -> None:
        """Test creating failed command result."""
        command = CreateUserCommand("test", "test@example.com")
        error_message = "Command execution failed"

        command_result: FlextCommandResult[None] = FlextCommandResult.failure(
            command,
            error_message,
        )

        assert command_result.is_success is False
        assert command_result.command == command
        assert command_result.result is None
        assert command_result.error == error_message

    def test_result_metadata(self) -> None:
        """Test getting result metadata."""
        command = CreateUserCommand("test", "test@example.com")
        result_data = {"id": "123"}

        command_result = FlextCommandResult.success(command, result_data)
        metadata = command_result.get_result_metadata()

        assert "command_id" in metadata
        assert "command_type" in metadata
        assert "is_success" in metadata
        assert "execution_time" in metadata
        assert metadata["is_success"] is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCommandPatternIntegration:
    """Integration tests for command pattern."""

    def test_complete_command_flow(self) -> None:
        """Test complete command execution flow."""
        # Setup command bus with handlers
        bus = FlextCommandBus()
        create_handler = CreateUserCommandHandler()
        update_handler = UpdateUserCommandHandler()

        bus.register_handler(create_handler)
        bus.register_handler(update_handler)

        # Execute create user command
        create_command = CreateUserCommand("john_doe", "john@example.com")
        create_result = bus.execute(create_command)

        assert create_result.is_success is True
        assert create_result.data is not None
        user_data = create_result.data.result
        assert user_data is not None
        user_id = user_data["id"]

        # Execute update user command
        update_command = UpdateUserCommand(
            user_id,
            {"email": "newemail@example.com"},
        )
        update_result = bus.execute(update_command)

        assert update_result.is_success is True
        assert update_result.data is not None
        assert update_result.data.result is not None
        assert update_result.data.result["user_id"] == user_id

    def test_multiple_command_types(self) -> None:
        """Test handling multiple command types."""
        bus = FlextCommandBus()

        # Register different handlers
        create_handler = CreateUserCommandHandler()
        update_handler = UpdateUserCommandHandler()

        bus.register_handler(create_handler)
        bus.register_handler(update_handler)

        # Execute different command types
        commands = [
            CreateUserCommand("user1", "user1@example.com"),
            CreateUserCommand("user2", "user2@example.com"),
            UpdateUserCommand("user1", {"status": "active"}),
            UpdateUserCommand("user2", {"bio": "Developer"}),
        ]

        results = []
        for command in commands:
            result = bus.execute(command)
            results.append(result)

        # Verify all commands executed successfully
        assert all(result.is_success for result in results)
        assert len(create_handler.created_users) == 2
        assert len(update_handler.updated_users) == 2

    def test_command_validation_and_processing_chain(self) -> None:
        """Test command validation and processing chain."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()
        bus.register_handler(handler)

        # Test with valid command
        valid_command = CreateUserCommand("valid_user", "valid@example.com")
        result = bus.execute(valid_command)
        assert result.is_success is True

        # Test with invalid command (validation should fail)
        invalid_command = CreateUserCommand("", "invalid")
        result = bus.execute(invalid_command)
        assert result.is_failure is True

        # Verify handler state is consistent
        assert len(handler.created_users) == 1  # Only valid command processed

    def test_error_handling_throughout_command_flow(self) -> None:
        """Test error handling at different stages."""
        bus = FlextCommandBus()

        # Test 1: No handler registered
        command = CreateUserCommand("test", "test@example.com")
        result = bus.execute(command)
        assert result.is_failure is True

        # Test 2: Handler that fails processing
        failing_handler = FailingCommandHandler()
        bus.register_handler(failing_handler)

        failing_command = FailingCommand()
        result = bus.execute(failing_command)
        assert result.is_failure is True

        # Test 3: Command validation failure
        create_handler = CreateUserCommandHandler()
        bus.register_handler(create_handler)

        invalid_command = CreateUserCommand("", "")
        result = bus.execute(invalid_command)
        assert result.is_failure is True

    def test_command_bus_handler_management(self) -> None:
        """Test command bus handler management features."""
        bus = FlextCommandBus()

        # Initially empty
        assert len(bus.get_all_handlers()) == 0

        # Add handlers
        handler1 = CreateUserCommandHandler()
        handler2 = UpdateUserCommandHandler()

        bus.register_handler(handler1)
        bus.register_handler(handler2)

        # Verify handlers are registered
        all_handlers = bus.get_all_handlers()
        assert len(all_handlers) == 2

        # Test finding specific handlers
        create_command = CreateUserCommand("test", "test@example.com")
        found_handler = bus.find_handler(create_command)
        assert found_handler == handler1

        update_command = UpdateUserCommand("123", {"name": "updated"})
        found_handler = bus.find_handler(update_command)
        assert found_handler == handler2

        # Test handler not found
        failing_command = FailingCommand()
        found_handler = bus.find_handler(failing_command)
        assert found_handler is None
