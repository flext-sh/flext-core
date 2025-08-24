"""Real functional tests for FlextCore - the main system facade.

Tests the real business functionality without mocks, focusing on
the integration and orchestration capabilities of FlextCore.
"""

from __future__ import annotations

import threading
import time
from typing import cast

import pytest

from flext_core import (
    FlextCore,
    FlextLoggerFactory,
    FlextResult,
    generate_uuid,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextCoreRealFunctionality:
    """Test real FlextCore functionality without mocks."""

    def test_flext_core_singleton_behavior(self) -> None:
        """Test FlextCore singleton pattern works correctly."""
        # Get two instances
        core1 = FlextCore.get_instance()
        core2 = FlextCore.get_instance()

        # Should be the same instance
        assert core1 is core2
        assert isinstance(core1, FlextCore)

    def test_container_integration_real_functionality(self) -> None:
        """Test FlextCore container integration with real services."""
        core = FlextCore.get_instance()

        # Register a real service
        class DatabaseService:
            def __init__(self) -> None:
                self.connections = 0

            def connect(self) -> str:
                self.connections += 1
                return f"Connected #{self.connections}"

        db_service = DatabaseService()
        result = core.register_service("database", db_service)

        assert result.success

        # Retrieve the service and use it
        service_result = core.get_service("database")
        assert service_result.success

        retrieved_db = cast("DatabaseService", service_result.value)
        connection_result = retrieved_db.connect()
        assert connection_result == "Connected #1"
        assert retrieved_db.connections == 1

    def test_factory_registration_real_functionality(self) -> None:
        """Test service factory registration and usage."""
        core = FlextCore.get_instance()

        # Register a factory that creates new instances each time
        counter = {"value": 0}

        def logger_factory() -> dict[str, object]:
            counter["value"] += 1
            return {
                "logger_id": f"logger_{counter['value']}",
                "created_at": time.time(),
                "level": "INFO",
            }

        result = core.register_factory("logger_factory", logger_factory)
        assert result.success

        # Get multiple instances from factory
        logger1_result = core.get_service("logger_factory")
        logger2_result = core.get_service("logger_factory")

        assert logger1_result.success
        assert logger2_result.success

        logger1 = cast("dict[str, object]", logger1_result.value)
        logger2 = cast("dict[str, object]", logger2_result.value)

        # Container caches factory results by default (singleton pattern)
        # Both calls return the same instance
        assert logger1 is logger2  # Same instance due to caching
        assert logger1["logger_id"] == logger2["logger_id"]
        assert logger1["logger_id"] == "logger_1"
        assert counter["value"] == 1  # Factory called only once

    def test_logging_integration_real_functionality(self) -> None:
        """Test FlextCore logging integration."""
        core = FlextCore.get_instance()

        # Configure logging through FlextCore
        core.configure_logging(log_level="DEBUG", _json_output=False)

        # Use the logging system
        logger = FlextLoggerFactory.get_logger("test_module")

        # Should not raise exceptions
        logger.info("Test message from FlextCore integration")
        logger.debug("Debug message", extra={"test": "value"})
        logger.warning("Warning message")

    def test_exception_handling_integration(self) -> None:
        """Test FlextCore exception handling and metrics."""
        core = FlextCore.get_instance()

        # Create validation error through FlextCore
        result = core.create_validation_error(
            "Test validation failed", field_name="email"
        )

        assert result is not None
        assert "Test validation failed" in str(result)

    def test_id_generation_utilities(self) -> None:
        """Test FlextCore ID generation utilities."""
        core = FlextCore.get_instance()

        # Test various ID generators
        uuid_result = core.generate_uuid()
        assert isinstance(uuid_result, str)
        assert len(uuid_result) == 36  # Standard UUID format
        assert "-" in uuid_result

        # Generate multiple UUIDs - should be unique
        uuid1 = core.generate_uuid()
        uuid2 = core.generate_uuid()
        assert uuid1 != uuid2

        # Test correlation ID generation
        corr_id_result = core.generate_correlation_id()
        assert isinstance(corr_id_result, str)
        assert corr_id_result.startswith("corr_")

    def test_configuration_management_real_workflow(self) -> None:
        """Test FlextCore configuration management."""
        core = FlextCore.get_instance()

        # Configure the container with real settings
        config_result = core.configure_container(
            max_services=100, enable_metrics=True, service_timeout=30.0
        )

        assert config_result.success

    def test_observability_integration(self) -> None:
        """Test FlextCore observability features."""
        core = FlextCore.get_instance()

        # Test health check functionality
        health_result = core.health_check()
        assert health_result is not None

        # Test system info
        system_info = core.get_system_info()
        assert system_info is not None

    def test_field_creation_integration(self) -> None:
        """Test FlextCore field creation functionality."""
        core = FlextCore.get_instance()

        # Create various field types
        string_field = core.create_string_field(
            name="username", min_length=3, max_length=20, required=True
        )

        assert string_field.field_name == "username"
        assert string_field.min_length == 3
        assert string_field.max_length == 20
        assert string_field.required is True

        # Create integer field
        integer_field = core.create_integer_field(
            name="age", min_value=0, max_value=150
        )

        assert integer_field.field_name == "age"
        assert integer_field.min_value == 0
        assert integer_field.max_value == 150

    def test_handler_registration_integration(self) -> None:
        """Test FlextCore handler registration functionality."""
        core = FlextCore.get_instance()

        # Create a simple command handler
        class TestCommand:
            def __init__(self, name: str) -> None:
                self.name = name

        class TestCommandHandler:
            def handle(self, command: TestCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Handled: {command.name}")

        # Register the handler using core method
        handler = TestCommandHandler()
        result = core.register_handler("test_command", handler)
        assert result.success

        # Retrieve and use the handler
        handler_result = core.get_handler("test_command")
        assert handler_result.success

        retrieved_handler = cast("TestCommandHandler", handler_result.value)
        command_result = retrieved_handler.handle(TestCommand("test"))
        assert command_result.success
        assert command_result.value == "Handled: test"


class TestFlextCoreEdgeCases:
    """Test FlextCore edge cases and error handling."""

    def test_service_not_found_handling(self) -> None:
        """Test handling of non-existent services."""
        core = FlextCore.get_instance()

        # Try to get non-existent service
        result = core.get_service("non_existent_service")

        assert result.is_failure
        assert "Service not found" in result.error or "not found" in result.error

    def test_invalid_logging_configuration(self) -> None:
        """Test handling of invalid logging configuration."""
        core = FlextCore.get_instance()

        # Try invalid log level - should not raise exception and return None
        result = core.configure_logging(log_level="INVALID_LEVEL", _json_output=True)

        # Should handle gracefully (returns None, no exception)
        assert result is None

    def test_container_configuration_edge_cases(self) -> None:
        """Test container configuration with edge cases."""
        core = FlextCore.get_instance()

        # Configure with various types
        result = core.configure_container(
            string_setting="test",
            numeric_setting=42,
            boolean_setting=True,
            none_setting=None,
        )

        # Should handle gracefully
        assert result.success

    def test_concurrent_singleton_access(self) -> None:
        """Test concurrent access to FlextCore singleton."""
        instances: list[FlextCore] = []

        def get_instance() -> None:
            instances.append(FlextCore.get_instance())

        # Create multiple threads
        threads = [threading.Thread(target=get_instance) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All instances should be the same
        assert len(instances) == 10
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance


class TestFlextCoreIntegrationScenarios:
    """Test complete integration scenarios using FlextCore."""

    def test_complete_user_management_workflow(self) -> None:
        """Test a complete user management workflow through FlextCore."""
        core = FlextCore.get_instance()

        # 1. Set up logging for the workflow
        core.configure_logging(log_level="INFO", _json_output=False)
        # configure_logging returns None - no need to check result

        # 2. Create and register user service
        class UserService:
            def __init__(self) -> None:
                self.users: dict[str, dict[str, object]] = {}

            def create_user(
                self, name: str, email: str
            ) -> FlextResult[dict[str, object]]:
                if not name or not email:
                    return FlextResult[dict[str, object]].fail(
                        "Name and email required"
                    )

                user_id = generate_uuid()
                user = {
                    "id": user_id,
                    "name": name,
                    "email": email,
                    "created_at": time.time(),
                }

                self.users[user_id] = user
                return FlextResult[dict[str, object]].ok(user)

            def get_user(self, user_id: str) -> FlextResult[dict[str, object]]:
                if user_id not in self.users:
                    return FlextResult[dict[str, object]].fail("User not found")
                return FlextResult[dict[str, object]].ok(self.users[user_id])

        user_service = UserService()
        register_result = core.register_service("user_service", user_service)
        assert register_result.success

        # 3. Use the service through FlextCore
        service_result = core.get_service("user_service")
        assert service_result.success

        service = cast("UserService", service_result.value)

        # 4. Create user
        create_result = service.create_user("John Doe", "john@example.com")
        assert create_result.success

        user = create_result.value
        assert user["name"] == "John Doe"
        assert user["email"] == "john@example.com"
        assert isinstance(user["id"], str)

        # 5. Retrieve user
        user_id = cast("str", user["id"])
        get_result = service.get_user(user_id)
        assert get_result.success

        retrieved_user = get_result.value
        assert retrieved_user["name"] == "John Doe"
        assert retrieved_user["email"] == "john@example.com"

    def test_event_driven_workflow_through_core(self) -> None:
        """Test event-driven workflow using FlextCore components."""
        core = FlextCore.get_instance()

        # Create event processor
        class EventProcessor:
            def __init__(self) -> None:
                self.events: list[dict[str, object]] = []

            def process_event(
                self, event_type: str, data: dict[str, object]
            ) -> FlextResult[None]:
                event = {
                    "type": event_type,
                    "data": data,
                    "timestamp": time.time(),
                    "event_id": generate_uuid(),
                }
                self.events.append(event)
                return FlextResult[None].ok(None)

            def get_events(self) -> list[dict[str, object]]:
                return self.events.copy()

        processor = EventProcessor()
        register_result = core.register_service("event_processor", processor)
        assert register_result.success

        # Process some events
        service_result = core.get_service("event_processor")
        assert service_result.success

        event_processor = cast("EventProcessor", service_result.value)

        # Process multiple events
        result1 = event_processor.process_event(
            "user_created", {"user_id": "123", "name": "Alice"}
        )
        result2 = event_processor.process_event(
            "user_updated", {"user_id": "123", "name": "Alice Smith"}
        )

        assert result1.success
        assert result2.success

        # Verify events
        events = event_processor.get_events()
        assert len(events) == 2
        assert events[0]["type"] == "user_created"
        assert events[1]["type"] == "user_updated"
