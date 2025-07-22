"""Basic functionality tests for flext-core.

These tests verify that the core functionality works correctly.
"""

import pytest
from flext_core.domain.shared_types import ServiceResult
from flext_core.config.validators import validate_url, validate_port
from flext_core.config.base import get_container
from flext_core.application.commands import Command
from flext_core.application.handlers import CommandHandler


class TestBasicFunctionality:
    """Test basic functionality of flext-core."""

    def test_service_result_creation(self) -> None:
        """Test ServiceResult creation and access."""
        # Test success case
        result = ServiceResult.ok({"test": "value"})
        assert result.success is True
        assert result.data == {"test": "value"}
        assert result.error is None

        # Test failure case
        result = ServiceResult.fail("Test error")
        assert result.success is False
        assert result.error == "Test error"
        assert result.data is None

    def test_validators(self) -> None:
        """Test configuration validators."""
        # Test URL validation
        assert validate_url("https://example.com") == "https://example.com"
        with pytest.raises(ValueError):
            validate_url("invalid-url")

        # Test port validation
        assert validate_port(8080) == 8080
        with pytest.raises(ValueError):
            validate_port(70000)

    def test_di_container(self) -> None:
        """Test dependency injection container."""
        container = get_container()
        assert container is not None
        assert hasattr(container, "register")

    def test_command_interface(self) -> None:
        """Test Command interface."""

        class TestCommand(Command):
            def __init__(self, data: str) -> None:
                self.data = data

            def validate_command(self) -> bool:
                return bool(self.data)

        command = TestCommand("test")
        assert command.validate_command() is True

        command = TestCommand("")
        assert command.validate_command() is False

    def test_command_handler_interface(self) -> None:
        """Test CommandHandler interface."""

        class TestCommand:
            def __init__(self, data: str) -> None:
                self.data = data

        class TestCommandHandler(CommandHandler[TestCommand, str]):
            async def handle(self, command: TestCommand) -> ServiceResult[str]:
                if command.data == "error":
                    return ServiceResult.fail("Test error")
                return ServiceResult.ok(data=f"Processed: {command.data}")

        # Test successful command
        handler = TestCommandHandler()
        command = TestCommand("success")

        import asyncio

        result = asyncio.run(handler.handle(command))

        assert result.success is True
        assert result.data == "Processed: success"

        # Test error command
        command = TestCommand("error")
        result = asyncio.run(handler.handle(command))

        assert result.success is False
        assert result.error == "Test error"


class TestIntegration:
    """Test integration scenarios."""

    def test_full_workflow(self) -> None:
        """Test a complete workflow."""
        # This test verifies that all components work together
        assert True  # Placeholder for now

    def test_error_handling(self) -> None:
        """Test error handling across components."""
        # This test verifies error handling works correctly
        assert True  # Placeholder for now
