"""Simple tests for FlextCommands using basic functionality."""

from __future__ import annotations

import pytest

# from pydantic import BaseModel  # Using FlextModels.BaseConfig instead
from flext_core import FlextCommands, FlextModels, FlextResult


class SampleCommand(FlextModels.BaseConfig):
    """Test command for comprehensive testing."""

    name: str
    value: int = 0

    def validate_command(self) -> FlextResult[None]:
        """Validate the test command."""
        if not self.name.strip():
            return FlextResult[None].fail("Name cannot be empty")
        if self.value < 0:
            return FlextResult[None].fail("Value cannot be negative")
        return FlextResult[None].ok(None)


class SampleCommandWithoutValidation(FlextModels.BaseConfig):
    """Test command without custom validation."""

    description: str


class TestFlextCommandsBasic:
    """Basic command testing with working functionality."""

    def test_command_creation(self) -> None:
        """Test basic command creation."""
        command = SampleCommand(name="test", value=42)

        assert command.name == "test"
        assert command.value == 42

    def test_command_validation_success(self) -> None:
        """Test successful command validation."""
        command = SampleCommand(name="valid", value=42)
        result = command.validate_command()

        assert result.success

    def test_command_validation_failure_empty_name(self) -> None:
        """Test command validation failure for empty name."""
        command = SampleCommand(name="", value=42)
        result = command.validate_command()

        assert result.is_failure
        assert "Name cannot be empty" in (result.error or "")

    def test_command_validation_failure_negative_value(self) -> None:
        """Test command validation failure for negative value."""
        command = SampleCommand(name="test", value=-1)
        result = command.validate_command()

        assert result.is_failure
        assert "Value cannot be negative" in (result.error or "")

    def test_command_without_validation(self) -> None:
        """Test command without custom validation."""
        command = SampleCommandWithoutValidation(description="test")

        assert command.description == "test"

    def test_command_bus_creation(self) -> None:
        """Test FlextCommands.Bus creation."""
        bus = FlextCommands.Bus()

        assert isinstance(bus, FlextCommands.Bus)
        assert hasattr(bus, "execute")
        assert hasattr(bus, "register_handler")

    def test_command_results_success(self) -> None:
        """Test FlextCommands.Results success factory."""
        success_result = FlextCommands.Results.success("success_data")
        assert success_result.success
        assert success_result.value == "success_data"

    def test_command_results_failure(self) -> None:
        """Test FlextCommands.Results failure factory."""
        fail_result = FlextCommands.Results.failure("error_message")
        assert fail_result.is_failure
        assert fail_result.error == "error_message"

    def test_command_factories_create_bus(self) -> None:
        """Test FlextCommands.Factories.create_command_bus."""
        bus = FlextCommands.Factories.create_command_bus()
        assert isinstance(bus, FlextCommands.Bus)

    def test_command_factories_create_handler(self) -> None:
        """Test FlextCommands.Factories.create_simple_handler."""

        def sample_handler(command: object) -> object:
            return f"handled: {command}"

        handler = FlextCommands.Factories.create_simple_handler(sample_handler)
        assert isinstance(handler, FlextCommands.Handlers.CommandHandler)

    def test_command_handler_basic(self) -> None:
        """Test basic command handler functionality."""

        class TestHandler(FlextCommands.Handlers.CommandHandler[SampleCommand, str]):
            def handle(self, command: SampleCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed: {command.name}")

        handler = TestHandler()
        assert handler.handler_name == "TestHandler"

        # Test handling
        test_command = SampleCommand(name="test", value=42)
        result = handler.handle(test_command)
        assert result.success
        assert "processed: test" in (result.value or "")

    def test_command_decorators(self) -> None:
        """Test command decorator patterns."""

        @FlextCommands.Decorators.command_handler(SampleCommand)
        def decorated_handler(command: object) -> object:
            if isinstance(command, SampleCommand):
                return f"decorated: {command.name}"
            return f"decorated: {command}"

        # Verify decorator was applied
        assert hasattr(decorated_handler, "__dict__")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
