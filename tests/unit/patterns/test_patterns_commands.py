"""Comprehensive tests for FLEXT command pattern."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.commands import FlextCommands
from flext_core.result import FlextResult

if TYPE_CHECKING:
    # Type aliases for command patterns
    FlextCommandId = str
    FlextCommandType = str

# =============================================================================
# Extract classes from FlextCommands
# Constants
EXPECTED_BULK_SIZE = 2

FlextCommand = FlextCommands.Command
FlextCommandHandler = FlextCommands.Handler
FlextCommandBus = FlextCommands.Bus
FlextCommandResult = FlextCommands.Result  # FlextCommands.Result with metadata

# TEST COMMAND IMPLEMENTATIONS
# =============================================================================


class CreateUserCommand(FlextCommand):
    """Test command for creating users."""

    username: str
    email: str

    def __init__(
        self,
        username: str,
        email: str,
        command_id: FlextCommandId | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize create user command with user details."""
        # Only pass command_id if it's provided, let default factory work otherwise
        if command_id is not None:
            super().__init__(
                command_id=command_id,
                command_type="create_user",
                username=username,
                email=email,
                **kwargs,
            )
        else:
            super().__init__(
                command_type="create_user",
                username=username,
                email=email,
                **kwargs,
            )

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

    user_id: str
    updates: dict[str, Any]

    def __init__(
        self,
        user_id: str,
        updates: dict[str, Any],
        command_id: FlextCommandId | None = None,
    ) -> None:
        """Initialize update user command with user ID and updates."""
        # Only pass command_id if it's provided, let default factory work otherwise
        if command_id is not None:
            super().__init__(
                command_id=command_id,
                command_type="update_user",
                user_id=user_id,
                updates=updates,
            )
        else:
            super().__init__(
                command_type="update_user",
                user_id=user_id,
                updates=updates,
            )

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
        super().__init__(command_type="failing")

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
        return "create_user"

    def can_handle(self, command: FlextCommand) -> bool:
        """Check if can handle command."""
        return (
            isinstance(
                command,
                CreateUserCommand,
            )
            and command.command_type == "create_user"
        )

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
        return "update_user"

    def can_handle(self, command: FlextCommand) -> bool:
        """Check if can handle command."""
        return (
            isinstance(
                command,
                UpdateUserCommand,
            )
            and command.command_type == "update_user"
        )

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
        return "failing"

    def can_handle(self, command: FlextCommand) -> bool:
        """Check if can handle command."""
        return (
            isinstance(
                command,
                FailingCommand,
            )
            and command.command_type == "failing"
        )

    def handle(self, command: FailingCommand) -> FlextResult[None]:
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
        if command.command_type != "create_user":
            raise AssertionError(
                f"Expected {'create_user'}, got {command.command_type}"
            )
        assert command.username == "john_doe"
        if command.email != "john@example.com":
            raise AssertionError(f"Expected {'john@example.com'}, got {command.email}")

    def test_command_creation_with_custom_id(self) -> None:
        """Test creating command with custom ID."""
        command_id = "custom_cmd_123"
        command = CreateUserCommand(
            "jane_doe",
            "jane@example.com",
            command_id=command_id,
        )

        if command.command_id != command_id:
            raise AssertionError(f"Expected {command_id}, got {command.command_id}")
        assert command.command_type == "create_user"

    def test_get_payload(self) -> None:
        """Test getting command payload."""
        command = CreateUserCommand("test_user", "test@example.com")
        payload = command.get_payload()

        if payload["username"] != "test_user":
            raise AssertionError(f"Expected {'test_user'}, got {payload['username']}")
        assert payload["email"] == "test@example.com"

    def test_validate_command_success(self) -> None:
        """Test successful command validation."""
        command = CreateUserCommand("valid_user", "valid@example.com")
        result = command.validate_command()

        if not (result.is_success):
            raise AssertionError(f"Expected True, got {result.is_success}")

    def test_validate_command_failure_no_username(self) -> None:
        """Test command validation failure for missing username."""
        command = CreateUserCommand("", "test@example.com")
        result = command.validate_command()

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        assert result.error is not None
        assert result.error
        if "username is required" not in result.error.lower():
            raise AssertionError(
                f"Expected {'username is required'} in {result.error.lower()}"
            )

    def test_validate_command_failure_no_email(self) -> None:
        """Test command validation failure for missing email."""
        command = CreateUserCommand("test_user", "")
        result = command.validate_command()

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        assert result.error is not None
        assert result.error
        if "email is required" not in result.error.lower():
            raise AssertionError(
                f"Expected {'email is required'} in {result.error.lower()}"
            )

    def test_validate_command_failure_invalid_email(self) -> None:
        """Test command validation failure for invalid email."""
        command = CreateUserCommand("test_user", "invalid_email")
        result = command.validate_command()

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        assert result.error is not None
        assert result.error
        if "invalid email" not in result.error.lower():
            raise AssertionError(
                f"Expected {'invalid email'} in {result.error.lower()}"
            )

    def test_get_command_metadata(self) -> None:
        """Test getting command metadata."""
        command = CreateUserCommand("test_user", "test@example.com")
        metadata = command.get_metadata()

        if "command_id" not in metadata:
            raise AssertionError(f"Expected {'command_id'} in {metadata}")
        assert "command_type" in metadata
        if "command_class" not in metadata:
            raise AssertionError(f"Expected {'command_class'} in {metadata}")
        if metadata["command_class"] != "CreateUserCommand":
            raise AssertionError(
                f"Expected {'CreateUserCommand'}, got {metadata['command_class']}"
            )


# =============================================================================
# TEST FLEXT COMMAND HANDLER
# =============================================================================


class TestFlextCommandHandler:
    """Test FlextCommandHandler functionality."""

    def test_handler_creation(self) -> None:
        """Test creating command handler."""
        handler = CreateUserCommandHandler()

        if handler.handler_id != "create_user_handler":
            raise AssertionError(
                f"Expected {'create_user_handler'}, got {handler.handler_id}"
            )
        assert handler.get_command_type() == "create_user"

    def test_can_handle_correct_command_type(self) -> None:
        """Test can_handle with correct command type."""
        handler = CreateUserCommandHandler()
        command = CreateUserCommand("test", "test@example.com")

        if not (handler.can_handle(command)):
            raise AssertionError(f"Expected True, got {handler.can_handle(command)}")

    def test_can_handle_wrong_command_type(self) -> None:
        """Test can_handle with wrong command type."""
        handler = CreateUserCommandHandler()
        command = UpdateUserCommand("123", {"name": "new_name"})

        if handler.can_handle(command):
            raise AssertionError(f"Expected False, got {handler.can_handle(command)}")

    def test_can_handle_non_command_object(self) -> None:
        """Test can_handle with non-command object."""
        handler = CreateUserCommandHandler()

        if handler.can_handle("not_a_command"):
            raise AssertionError(
                f"Expected False, got {handler.can_handle('not_a_command')}"
            )

    def test_handle_command_success(self) -> None:
        """Test successful command handling."""
        handler = CreateUserCommandHandler()
        command = CreateUserCommand("john", "john@example.com")

        result = handler.handle_command(command)

        if not (result.is_success):
            raise AssertionError(f"Expected True, got {result.is_success}")
        assert result.data is not None
        if result.data["username"] != "john":
            raise AssertionError(f"Expected {'john'}, got {result.data['username']}")
        assert result.data["email"] == "john@example.com"
        if "id" not in result.data:
            raise AssertionError(f"Expected {'id'} in {result.data}")

    def test_process_command_success(self) -> None:
        """Test complete command processing flow."""
        handler = CreateUserCommandHandler()
        command = CreateUserCommand("jane", "jane@example.com")

        result = handler.process_command(command)

        if not (result.is_success):
            raise AssertionError(f"Expected True, got {result.is_success}")
        if len(handler.created_users) != 1:
            raise AssertionError(f"Expected {1}, got {len(handler.created_users)}")

    def test_process_command_validation_failure(self) -> None:
        """Test processing with command validation failure."""
        handler = CreateUserCommandHandler()
        command = CreateUserCommand("", "invalid")  # Invalid command

        result = handler.process_command(command)

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        assert result.error is not None
        assert result.error
        if "username is required" not in result.error.lower():
            raise AssertionError(
                f"Expected {'username is required'} in {result.error.lower()}"
            )

    def test_process_command_cannot_handle(self) -> None:
        """Test processing command that cannot be handled."""
        handler = CreateUserCommandHandler()
        wrong_command = UpdateUserCommand("123", {"name": "test"})

        result = handler.process_command(wrong_command)

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        assert result.error is not None
        assert result.error
        if "cannot process" not in result.error.lower():
            raise AssertionError(
                f"Expected {'cannot process'} in {result.error.lower()}"
            )

    def test_process_command_handling_failure(self) -> None:
        """Test processing when handler fails."""
        handler = FailingCommandHandler()
        command = FailingCommand()

        result = handler.process_command(command)

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")


# =============================================================================
# TEST FLEXT COMMAND BUS
# =============================================================================


class TestFlextCommandBus:
    """Test FlextCommandBus functionality."""

    def test_command_bus_creation(self) -> None:
        """Test creating command bus."""
        bus = FlextCommandBus()

        if len(bus.get_all_handlers()) != 0:
            raise AssertionError(f"Expected {0}, got {len(bus.get_all_handlers())}")

    def test_register_handler_success(self) -> None:
        """Test successful handler registration."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()

        bus.register_handler(handler)

        if len(bus.get_all_handlers()) != 1:
            raise AssertionError(f"Expected {1}, got {len(bus.get_all_handlers())}")

    def test_register_invalid_handler(self) -> None:
        """Test registering invalid handler."""
        bus = FlextCommandBus()

        # Register a string that will fail when trying to call handler methods
        bus.register_handler("not_a_handler")

        # The failure will happen when trying to execute a command
        command = CreateUserCommand("test", "test@example.com")

        # This should fail because "not_a_handler" string doesn't have required methods
        try:
            result = bus.execute(command)
            # If we get here, there might be no handlers or the string handler failed
            assert result.is_failure
        except AttributeError:
            pass  # Expected - string doesn't have can_handle method

    def test_execute_command_success(self) -> None:
        """Test successful command execution."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()
        bus.register_handler(handler)

        command = CreateUserCommand("alice", "alice@example.com")
        result = bus.execute(command)

        if not (result.is_success):
            raise AssertionError(f"Expected True, got {result.is_success}")
        assert result.data is not None
        if result.data["username"] != "alice":
            raise AssertionError(f"Expected {'alice'}, got {result.data['username']}")

    def test_execute_command_no_handler(self) -> None:
        """Test executing command with no registered handler."""
        bus = FlextCommandBus()
        command = CreateUserCommand("bob", "bob@example.com")

        result = bus.execute(command)

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        assert result.error is not None
        assert result.error
        if "no handler found" not in result.error.lower():
            raise AssertionError(
                f"Expected {'no handler found'} in {result.error.lower()}"
            )

    def test_execute_command_validation_failure(self) -> None:
        """Test executing invalid command."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()
        bus.register_handler(handler)

        command = CreateUserCommand("", "")  # Invalid command
        result = bus.execute(command)

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")

    def test_find_handler_for_command(self) -> None:
        """Test finding handler for command."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()
        bus.register_handler(handler)

        command = CreateUserCommand("test", "test@example.com")
        found_handler = bus.find_handler(command)

        if found_handler != handler:
            raise AssertionError(f"Expected {handler}, got {found_handler}")

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

        if len(all_handlers) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(all_handlers)}")
        if handler1 not in all_handlers:
            raise AssertionError(f"Expected {handler1} in {all_handlers}")
        assert handler2 in all_handlers


# =============================================================================
# TEST FLEXT COMMAND RESULT
# =============================================================================


class TestFlextCommandResult:
    """Test FlextCommandResult functionality."""

    def test_success_result_creation(self) -> None:
        """Test creating successful command result."""
        result_data = {"id": "123", "username": "test"}

        command_result = FlextCommandResult.ok(result_data)

        if not (command_result.is_success):
            raise AssertionError(f"Expected True, got {command_result.is_success}")
        if command_result.data != result_data:
            raise AssertionError(f"Expected {result_data}, got {command_result.data}")
        assert command_result.error is None

    def test_failure_result_creation(self) -> None:
        """Test creating failed command result."""
        error_message = "Command execution failed"

        command_result: FlextCommandResult[None] = FlextCommandResult.fail(
            error_message,
        )

        if command_result.is_success:
            raise AssertionError(f"Expected False, got {command_result.is_success}")
        assert command_result.data is None
        if command_result.error != error_message:
            raise AssertionError(
                f"Expected {error_message}, got {command_result.error}"
            )

    def test_result_metadata(self) -> None:
        """Test result metadata properties."""
        result_data = {"id": "123"}

        command_result = FlextCommandResult.ok(result_data, metadata={"test": "value"})

        if not (command_result.is_success):
            raise AssertionError(f"Expected True, got {command_result.is_success}")
        if command_result.metadata["test"] != "value":
            raise AssertionError(
                f"Expected {'value'}, got {command_result.metadata['test']}"
            )


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

        if not (create_result.is_success):
            raise AssertionError(f"Expected True, got {create_result.is_success}")
        assert create_result.data is not None
        user_data = create_result.data
        assert user_data is not None
        user_id = user_data["id"]

        # Execute update user command
        update_command = UpdateUserCommand(
            user_id,
            {"email": "newemail@example.com"},
        )
        update_result = bus.execute(update_command)

        if not (update_result.is_success):
            raise AssertionError(f"Expected True, got {update_result.is_success}")
        assert update_result.data is not None
        if update_result.data["user_id"] != user_id:
            raise AssertionError(
                f"Expected {user_id}, got {update_result.data['user_id']}"
            )

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
        if not all(result.is_success for result in results):
            raise AssertionError(
                f"Expected {all(result.is_success for result in results)} in {results}"
            )
        if len(create_handler.created_users) != EXPECTED_BULK_SIZE:
            raise AssertionError(
                f"Expected {2}, got {len(create_handler.created_users)}"
            )
        assert len(update_handler.updated_users) == EXPECTED_BULK_SIZE

    def test_command_validation_and_processing_chain(self) -> None:
        """Test command validation and processing chain."""
        bus = FlextCommandBus()
        handler = CreateUserCommandHandler()
        bus.register_handler(handler)

        # Test with valid command
        valid_command = CreateUserCommand("valid_user", "valid@example.com")
        result = bus.execute(valid_command)
        if not (result.is_success):
            raise AssertionError(f"Expected True, got {result.is_success}")

        # Test with invalid command (validation should fail)
        invalid_command = CreateUserCommand("", "invalid")
        result = bus.execute(invalid_command)
        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")

        # Verify handler state is consistent
        if len(handler.created_users) != 1:  # Only valid command processed:
            raise AssertionError(
                f"Expeced {1} # Only valid command processed, got {len(handler.created_users)}"
            )

    def test_error_handling_throughout_command_flow(self) -> None:
        """Test error handling at different stages."""
        bus = FlextCommandBus()

        # Test 1: No handler registered
        command = CreateUserCommand("test", "test@example.com")
        result = bus.execute(command)
        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")

        # Test 2: Handler that fails processing
        failing_handler = FailingCommandHandler()
        bus.register_handler(failing_handler)

        failing_command = FailingCommand()
        result = bus.execute(failing_command)
        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")

        # Test 3: Command validation failure
        create_handler = CreateUserCommandHandler()
        bus.register_handler(create_handler)

        invalid_command = CreateUserCommand("", "")
        result = bus.execute(invalid_command)
        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")

    def test_command_bus_handler_management(self) -> None:
        """Test command bus handler management features."""
        bus = FlextCommandBus()

        # Initially empty
        if len(bus.get_all_handlers()) != 0:
            raise AssertionError(f"Expected {0}, got {len(bus.get_all_handlers())}")

        # Add handlers
        handler1 = CreateUserCommandHandler()
        handler2 = UpdateUserCommandHandler()

        bus.register_handler(handler1)
        bus.register_handler(handler2)

        # Verify handlers are registered
        all_handlers = bus.get_all_handlers()
        if len(all_handlers) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(all_handlers)}")

        # Test finding specific handlers
        create_command = CreateUserCommand("test", "test@example.com")
        found_handler = bus.find_handler(create_command)
        if found_handler != handler1:
            raise AssertionError(f"Expected {handler1}, got {found_handler}")

        update_command = UpdateUserCommand("123", {"name": "updated"})
        found_handler = bus.find_handler(update_command)
        if found_handler != handler2:
            raise AssertionError(f"Expected {handler2}, got {found_handler}")

        # Test handler not found
        failing_command = FailingCommand()
        found_handler = bus.find_handler(failing_command)
        assert found_handler is None
