"""Comprehensive tests for FLEXT command pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import Annotated, override

from pydantic import Field

from flext_core import FlextConstants, FlextHandlers, FlextModels, r

from ...models import m

EXPECTED_BULK_SIZE = 2
FlextCommandId = str
FlextCommandType = str


class CreateUserCommand(FlextModels.Command):
    """Test command for creating users."""

    username: str
    email: str

    def get_payload(self) -> m.UserPayloadDict:
        """Get command payload."""
        return m.UserPayloadDict({
            "username": self.username,
            "email": self.email,
        })

    def validate_command(self) -> r[bool]:
        """Validate command data."""
        if not self.username:
            return r[bool].fail("Username is required")
        if not self.email:
            return r[bool].fail("Email is required")
        if "@" not in self.email:
            return r[bool].fail("Invalid email format")
        return r[bool].ok(True)


class UpdateUserCommand(FlextModels.Command):
    """Test command for updating users."""

    target_user_id: str
    updates: dict[str, object]

    def get_payload(self) -> m.UpdatePayloadDict:
        """Get command payload."""
        typed_updates: dict[str, m.UpdateFieldDict] = {
            key: m.UpdateFieldDict({
                "field_name": key,
                "new_value": value
                if isinstance(value, (str, int, bool))
                else str(value),
            })
            for key, value in self.updates.items()
        }
        return m.UpdatePayloadDict({
            "target_user_id": self.target_user_id,
            "updates": typed_updates,
        })

    def validate_command(self) -> r[bool]:
        """Validate command data."""
        if not self.target_user_id:
            return r[bool].fail("Target User ID is required")
        if not self.updates:
            return r[bool].fail("Updates are required")
        return r[bool].ok(True)


class FailingCommand(FlextModels.Command):
    """Test command that always fails validation."""

    def get_payload(self) -> m.CommandPayloadDict:
        """Get command payload."""
        return m.CommandPayloadDict({})

    def validate_command(self) -> r[bool]:
        """Fail validation intentionally."""
        return r[bool].fail("This command always fails")


def _create_user_command(*, username: str, email: str) -> CreateUserCommand:
    return CreateUserCommand({"username": username, "email": email})


def _update_user_command(
    *,
    target_user_id: str,
    updates: dict[str, object],
) -> UpdateUserCommand:
    return UpdateUserCommand({
        "target_user_id": target_user_id,
        "updates": updates,
    })


class CreateUserCommandHandler(
    FlextHandlers[CreateUserCommand, dict[str, object]],
):
    """Test handler for CreateUserCommand."""

    created_users: Annotated[list[dict[str, object]], Field(default_factory=list)]

    def __init__(self) -> None:
        """Initialize create user command handler."""
        config = FlextModels.Handler(
            handler_id="create_user_handler",
            handler_name="Create User Handler",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        super().__init__(config=config)

    def get_command_type(self) -> FlextCommandType:
        """Get command type this handler processes."""
        return "create_user"

    @override
    def can_handle(self, message_type: object) -> bool:
        """Check if can handle command."""
        return message_type == CreateUserCommand or str(message_type) == "create_user"

    @override
    def validate(self, data: object) -> r[bool]:
        """Validate command using command's validate_command method."""
        if isinstance(data, CreateUserCommand):
            return data.validate_command()
        return r[bool].fail("Cannot handle this command type")

    @override
    def handle(
        self,
        message: CreateUserCommand,
    ) -> r[dict[str, object]]:
        """Handle the create user command."""
        user_data: dict[str, object] = {
            "id": f"user_{len(self.created_users) + 1}",
            "username": message.username,
            "email": message.email,
        }
        self.created_users.append(user_data)
        return r[dict[str, object]].ok(user_data)

    def handle_command(
        self,
        command: CreateUserCommand,
    ) -> r[dict[str, object]]:
        """Handle the create user command (alias for handle)."""
        return self.handle(command)


class UpdateUserCommandHandler(
    FlextHandlers[UpdateUserCommand, dict[str, object]],
):
    """Test handler for UpdateUserCommand."""

    def __init__(self) -> None:
        """Initialize update user command handler."""
        config = FlextModels.Handler(
            handler_id="update_user_handler",
            handler_name="Update User Handler",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        super().__init__(config=config)
        self.updated_users: dict[str, object] = {}

    def get_command_type(self) -> FlextCommandType:
        """Get command type this handler processes."""
        return "update_user"

    @override
    def can_handle(self, message_type: object) -> bool:
        """Check if can handle command."""
        return message_type == UpdateUserCommand or str(message_type) == "update_user"

    @override
    def validate(self, data: object) -> r[bool]:
        """Validate command using command's validate_command method."""
        if isinstance(data, UpdateUserCommand):
            return data.validate_command()
        return r[bool].fail("Cannot handle this command type")

    @override
    def handle(
        self,
        message: UpdateUserCommand,
    ) -> r[dict[str, object]]:
        """Handle the update user command."""
        if message.target_user_id not in self.updated_users:
            self.updated_users[message.target_user_id] = {}
        user_updates = self.updated_users[message.target_user_id]
        if isinstance(user_updates, dict):
            user_updates.update(message.updates)
        result_data: dict[str, object] = {
            "target_user_id": message.target_user_id,
            "updated_fields": list(message.updates.keys()),
        }
        return r[dict[str, object]].ok(result_data)

    def handle_command(
        self,
        command: UpdateUserCommand,
    ) -> r[dict[str, object]]:
        """Handle the update user command (alias for handle)."""
        return self.handle(command)


class FailingCommandHandler(FlextHandlers[FailingCommand, bool]):
    """Test handler that always fails."""

    def get_command_type(self) -> FlextCommandType:
        """Get command type this handler processes."""
        return "failing"

    @override
    def can_handle(self, message_type: object) -> bool:
        """Check if can handle command."""
        return message_type == FailingCommand or str(message_type) == "failing"

    @override
    def validate(self, data: object) -> r[bool]:
        """Validate command using command's validate_command method."""
        if isinstance(data, FailingCommand):
            return data.validate_command()
        return r[bool].fail("Cannot handle this command type")

    @override
    def handle(self, message: FailingCommand) -> r[bool]:
        """Fail to handle command intentionally."""
        error_msg = (
            f"Handler processing failed for command: {message.__class__.__name__}"
        )
        return r[bool].fail(error_msg)

    def handle_command(self, command: FailingCommand) -> r[bool]:
        """Fail to handle command intentionally (alias for handle)."""
        return self.handle(command)


class TestFlextCommand:
    """Test FlextCommand functionality."""

    def test_command_creation_with_auto_id(self) -> None:
        """Test creating command with auto-generated ID."""
        command: CreateUserCommand = _create_user_command(
            username="john_doe",
            email="john@example.com",
        )
        assert command.username == "john_doe"
        assert command.email == "john@example.com"
        assert command.username == "john_doe"
        if command.email != "john@example.com":
            msg = f"Expected {'john@example.com'}, got {command.email}"
            raise AssertionError(msg)

    def test_command_creation_with_custom_id(self) -> None:
        """Test creating command with custom ID."""
        command = _create_user_command(username="jane_doe", email="jane@example.com")
        assert command.username == "jane_doe"
        assert command.email == "jane@example.com"

    def test_get_payload(self) -> None:
        """Test getting command payload."""
        command = _create_user_command(username="test_user", email="test@example.com")
        payload = command.get_payload()
        username = payload.username
        if username != "test_user":
            msg = f"Expected {'test_user'}, got {username}"
            raise AssertionError(msg)
        assert payload.email == "test@example.com"

    def test_validate_command_success(self) -> None:
        """Test successful command validation."""
        command = _create_user_command(username="valid_user", email="valid@example.com")
        result = command.validate_command()
        if not result.is_success:
            msg = f"Expected True, got {result.is_success}"
            raise AssertionError(msg)

    def test_validate_command_failure_no_username(self) -> None:
        """Test command validation failure for missing username."""
        command = _create_user_command(username="", email="test@example.com")
        result = command.validate_command()
        if not result.is_failure:
            msg = f"Expected True, got {result.is_failure}"
            raise AssertionError(msg)
        assert result.error is not None
        assert result.error
        if "username is required" not in (result.error or "").lower():
            msg = f"Expected {'username is required'} in {(result.error or '').lower()}"
            raise AssertionError(msg)

    def test_validate_command_failure_no_email(self) -> None:
        """Test command validation failure for missing email."""
        command = _create_user_command(username="test_user", email="")
        result = command.validate_command()
        if not result.is_failure:
            msg = f"Expected True, got {result.is_failure}"
            raise AssertionError(msg)
        assert result.error is not None
        assert result.error
        if "email is required" not in (result.error or "").lower():
            msg = f"Expected {'email is required'} in {(result.error or '').lower()}"
            raise AssertionError(msg)

    def test_validate_command_failure_invalid_email(self) -> None:
        """Test command validation failure for invalid email."""
        command = _create_user_command(username="test_user", email="invalid_email")
        result = command.validate_command()
        if not result.is_failure:
            msg = f"Expected True, got {result.is_failure}"
            raise AssertionError(msg)
        assert result.error is not None
        assert result.error
        if "invalid email" not in (result.error or "").lower():
            msg = f"Expected {'invalid email'} in {(result.error or '').lower()}"
            raise AssertionError(msg)

    def test_get_command_metadata(self) -> None:
        """Test command basic properties."""
        command = _create_user_command(username="test_user", email="test@example.com")
        assert command.username == "test_user"
        assert command.email == "test@example.com"
        assert isinstance(command, CreateUserCommand)


class TestFlextCommandHandler:
    """Test FlextCommandHandler functionality."""

    def test_handler_creation(self) -> None:
        """Test creating command handler."""
        handler = CreateUserCommandHandler()
        handler_id = (
            handler._config_model.handler_id
            if hasattr(handler, "_config_model")
            else ""
        )
        if handler_id != "create_user_handler":
            msg = f"Expected {'create_user_handler'}, got {handler_id}"
            raise AssertionError(msg)
        assert handler.get_command_type() == "create_user"

    def test_can_handle_correct_command_type(self) -> None:
        """Test can_handle with correct command type."""
        handler: CreateUserCommandHandler = CreateUserCommandHandler()
        if not handler.can_handle(CreateUserCommand):
            msg = f"Expected True, got {handler.can_handle(CreateUserCommand)}"
            raise AssertionError(msg)

    def test_can_handle_wrong_command_type(self) -> None:
        """Test can_handle with wrong command type."""
        handler: FlextHandlers[CreateUserCommand, dict[str, object]] = (
            CreateUserCommandHandler()
        )
        if handler.can_handle(UpdateUserCommand):
            msg = f"Expected False, got {handler.can_handle(UpdateUserCommand)}"
            raise AssertionError(msg)

    def test_can_handle_non_command_object(self) -> None:
        """Test can_handle with non-command object."""
        handler: FlextHandlers[CreateUserCommand, dict[str, object]] = (
            CreateUserCommandHandler()
        )
        if handler.can_handle(str):
            msg = f"Expected False, got {handler.can_handle(str)}"
            raise AssertionError(msg)

    def test_handle_command_success(self) -> None:
        """Test successful command handling."""
        handler: FlextHandlers[CreateUserCommand, dict[str, object]] = (
            CreateUserCommandHandler()
        )
        command: CreateUserCommand = _create_user_command(
            username="john",
            email="john@example.com",
        )
        result = handler.handle(command)
        if not result.is_success:
            msg = f"Expected True, got {result.is_success}"
            raise AssertionError(msg)
        assert result.value is not None
        if (result.value or {})["username"] != "john":
            msg = f"Expected {'john'}, got {(result.value or {})['username']}"
            raise AssertionError(msg)
        assert (result.value or {})["email"] == "john@example.com"
        if "id" not in result.value:
            msg = f"Expected {'id'} in {result.value}"
            raise AssertionError(msg)

    def test_process_command_success(self) -> None:
        """Test complete command processing flow."""
        handler: CreateUserCommandHandler = CreateUserCommandHandler()
        command: CreateUserCommand = _create_user_command(
            username="jane",
            email="jane@example.com",
        )
        result = handler.handle(command)
        if not result.is_success:
            msg = f"Expected True, got {result.is_success}"
            raise AssertionError(msg)
        if len(handler.created_users) != 1:
            msg = f"Expected {1}, got {len(handler.created_users)}"
            raise AssertionError(msg)

    def test_process_command_validation_failure(self) -> None:
        """Test processing with command validation failure."""
        handler: FlextHandlers[CreateUserCommand, dict[str, object]] = (
            CreateUserCommandHandler()
        )
        command: CreateUserCommand = _create_user_command(username="", email="invalid")
        result = handler.execute(command)
        if not result.is_failure:
            msg = f"Expected True, got {result.is_failure}"
            raise AssertionError(msg)
        assert result.error is not None
        assert result.error
        if "username is required" not in (result.error or "").lower():
            msg = f"Expected {'username is required'} in {(result.error or '').lower()}"
            raise AssertionError(msg)

    def test_process_command_cannot_handle(self) -> None:
        """Test validation failure for wrong command type."""
        wrong_command: UpdateUserCommand = _update_user_command(
            target_user_id="123",
            updates={"name": "test"},
        )
        result = CreateUserCommandHandler().validate(wrong_command)
        if not result.is_failure:
            msg = f"Expected True, got {result.is_failure}"
            raise AssertionError(msg)
        assert result.error is not None
        assert result.error
        if "cannot handle" not in (result.error or "").lower():
            msg = f"Expected {'cannot handle'} in {(result.error or '').lower()}"
            raise AssertionError(msg)


class TestFlextCommandResults:
    """Test FlextCommandResults functionality."""

    def test_success_result_creation(self) -> None:
        """Test creating successful command result."""
        result_data: dict[str, object] = {"id": "123", "username": "test"}
        command_result: r[dict[str, object]] = r[dict[str, object]].ok(result_data)
        if not command_result.is_success:
            msg = f"Expected True, got {command_result.is_success}"
            raise AssertionError(msg)
        if command_result.value != result_data:
            msg = f"Expected {result_data}, got {command_result.value}"
            raise AssertionError(msg)
        assert command_result.error is None

    def test_failure_result_creation(self) -> None:
        """Test creating failed command result."""
        error_message = "Command execution failed"
        command_result: r[bool] = r[bool].fail(error_message)
        if command_result.is_success:
            msg = f"Expected False, got {command_result.is_success}"
            raise AssertionError(msg)
        assert command_result.is_failure
        if command_result.error != error_message:
            msg = f"Expected {error_message}, got {command_result.error}"
            raise AssertionError(msg)

    def test_result_metadata(self) -> None:
        """Test result metadata properties."""
        result_data = {"id": "123"}
        command_result = r[dict[str, str]].ok(result_data)
        if not command_result.is_success:
            msg = f"Expected True, got {command_result.is_success}"
            raise AssertionError(msg)
        assert command_result.error_data is None
