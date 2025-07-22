"""Core integration tests for flext-core.

These tests validate that the core components work together correctly.
"""

from typing import Any, cast

import pytest
from flext_core.domain.shared_types import ServiceResult
from flext_core.domain.shared_types import JsonDict
from flext_core.config.validators import (
    validate_url,
    validate_port,
    validate_database_url,
)
from flext_core.config.base import get_container, configure_container
from flext_core.application.commands import Command
from flext_core.application.handlers import CommandHandler, QueryHandler
from flext_core.domain.core import Repository
from flext_core.domain.shared_models import ValidationResult


class TestCoreIntegration:
    """Test core integration scenarios."""

    def test_service_result_with_json_dict(self) -> None:
        """Test ServiceResult works with JsonDict."""
        data: JsonDict = {"key": "value", "number": 42, "list": [1, 2, 3]}
        result = ServiceResult.ok(data)

        assert result.success is True
        assert result.data == data
        assert isinstance(result.data, dict)

    def test_validation_result(self) -> None:
        """Test ValidationResult functionality."""
        # Test valid case
        validation = ValidationResult(is_valid=True)
        assert validation.is_valid is True
        assert len(validation.errors) == 0
        assert len(validation.warnings) == 0

        # Test invalid case with errors
        validation = ValidationResult(is_valid=False)
        validation.errors.append("Field is required")
        validation.errors.append("Invalid format")

        assert validation.is_valid is False
        assert len(validation.errors) == 2
        assert "Field is required" in validation.errors

        # Test with warnings
        validation.warnings.append("Deprecated field")
        assert len(validation.warnings) == 1
        assert "Deprecated field" in validation.warnings

    def test_validators_integration(self) -> None:
        """Test all validators work together."""
        # Test URL validation
        assert (
            validate_url("https://api.example.com/v1") == "https://api.example.com/v1"
        )
        assert validate_url("http://localhost:8080") == "http://localhost:8080"

        # Test port validation
        assert validate_port(80) == 80
        assert validate_port(8080) == 8080
        assert validate_port(65535) == 65535

        # Test database URL validation
        assert (
            validate_database_url("postgresql://user:pass@localhost:5432/db")
            == "postgresql://user:pass@localhost:5432/db"
        )
        assert (
            validate_database_url("mysql://user:pass@localhost:3306/db")
            == "mysql://user:pass@localhost:3306/db"
        )

    def test_di_container_integration(self) -> None:
        """Test DI container integration."""
        container = get_container()

        # Test container has required methods
        assert hasattr(container, "register")
        assert hasattr(container, "resolve")
        assert hasattr(container, "register_factory")
        assert hasattr(container, "register_singleton")
        assert hasattr(container, "get_all")

        # Test container is functional
        assert container is not None

    def test_command_handler_integration(self) -> None:
        """Test command handler integration."""

        class CreateUserCommand:
            def __init__(self, name: str, email: str) -> None:
                self.name = name
                self.email = email

        class CreateUserHandler(CommandHandler[CreateUserCommand, dict[str, Any]]):
            async def handle(
                self, command: CreateUserCommand
            ) -> ServiceResult[dict[str, Any]]:
                if not command.name or not command.email:
                    return ServiceResult.fail("Name and email required")

                user_data = {
                    "id": "user_123",
                    "name": command.name,
                    "email": command.email,
                    "status": "active",
                }
                return ServiceResult.ok({"result": user_data})

        # Test successful creation
        handler = CreateUserHandler()
        command = CreateUserCommand("John Doe", "john@example.com")

        import asyncio

        result = asyncio.run(handler.handle(command))

        assert result.success is True
        assert cast(dict[str, Any], result.data)["result"]["name"] == "John Doe"
        assert (
            cast(dict[str, Any], result.data)["result"]["email"] == "john@example.com"
        )

        # Test validation failure
        command = CreateUserCommand("", "")
        result = asyncio.run(handler.handle(command))

        assert result.success is False
        assert result.error == "Name and email required"

    def test_repository_interface(self) -> None:
        """Test repository interface."""

        class MockEntity:
            def __init__(self, id: str, name: str) -> None:
                self.id = id
                self.name = name

        class MockRepository(Repository[MockEntity, str]):
            def __init__(self) -> None:
                self._data: dict[str, MockEntity] = {}

            async def save(self, entity: MockEntity) -> MockEntity:
                self._data[entity.id] = entity
                return entity

            async def find_by_id(self, entity_id: str) -> MockEntity | None:
                return self._data.get(entity_id)

            async def delete(self, entity_id: str) -> bool:
                if entity_id in self._data:
                    del self._data[entity_id]
                    return True
                return False

            async def find_all(self) -> list[MockEntity]:
                return list(self._data.values())

            async def count(self) -> int:
                return len(self._data)

        # Test repository operations
        repo = MockRepository()
        entity = MockEntity("1", "Test Entity")

        import asyncio

        # Test save
        saved = asyncio.run(repo.save(entity))
        assert saved.id == "1"
        assert saved.name == "Test Entity"

        # Test find
        found = asyncio.run(repo.find_by_id("1"))
        assert found is not None
        assert found.name == "Test Entity"

        # Test delete
        deleted = asyncio.run(repo.delete("1"))
        assert deleted is True

        # Test find after delete
        found = asyncio.run(repo.find_by_id("1"))
        assert found is None


class TestErrorHandling:
    """Test error handling across components."""

    def test_validator_errors(self) -> None:
        """Test validator error handling."""
        with pytest.raises(ValueError):
            validate_url("invalid-url")

        with pytest.raises(ValueError):
            validate_port(0)

        with pytest.raises(ValueError):
            validate_port(70000)

    def test_service_result_error_handling(self) -> None:
        """Test ServiceResult error handling."""
        result = ServiceResult.fail("Database connection failed")
        assert result.success is False
        assert result.error == "Database connection failed"
        assert result.data is None

    def test_validation_result_error_handling(self) -> None:
        """Test ValidationResult error handling."""
        validation = ValidationResult(is_valid=False)
        validation.errors.append("Connection timeout")
        validation.errors.append("Authentication failed")

        assert validation.is_valid is False
        assert len(validation.errors) == 2
        assert "Connection timeout" in validation.errors
        assert "Authentication failed" in validation.errors
