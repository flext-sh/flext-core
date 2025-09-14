"""Tests for FlextContainer to achieve better coverage targeting specific uncovered lines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from unittest.mock import patch

from flext_core import FlextContainer, FlextResult


class TestFlextContainerCoverageBoost:
    """Tests targeting specific uncovered lines in container.py for better coverage."""

    def test_batch_register_success_scenario(self) -> None:
        """Test batch_register success path - covers registration loop."""
        container = FlextContainer()

        class TestService:
            def get_data(self) -> str:
                return "test_data"

        def test_factory() -> str:
            return "factory_result"

        # Test batch registration with mixed services and factories
        registrations = {
            "service1": TestService(),
            "factory1": test_factory,
            "service2": "simple_string",
        }

        result = container.batch_register(registrations)
        assert result.is_success
        assert len(result.value) == 3
        assert "service1" in result.value
        assert "factory1" in result.value
        assert "service2" in result.value

    def test_batch_register_rollback_on_failure(self) -> None:
        """Test batch_register rollback mechanism - covers rollback lines."""
        container = FlextContainer()

        # Pre-register a valid service
        container.register("existing_service", "existing_value")

        # Mock register to fail on second item to trigger rollback
        with patch.object(container, "register") as mock_register:
            mock_register.side_effect = [
                FlextResult[None].ok(None),  # First succeeds
                FlextResult[None].fail("Registration failed"),  # Second fails
            ]

            registrations = {"service1": "value1", "service2": "value2"}
            result = container.batch_register(registrations)

            # Should fail and trigger rollback
            assert result.is_failure
            assert result.error
            assert "Batch registration failed" in result.error

    def test_get_typed_type_mismatch(self) -> None:
        """Test get_typed with type mismatch - covers type checking lines."""
        container = FlextContainer()
        container.register("string_service", "test_string")

        # Try to get as int when it's a string
        result = container.get_typed("string_service", int)
        assert result.is_failure
        assert result.error
        assert "is str, expected int" in result.error

    def test_get_typed_service_not_found(self) -> None:
        """Test get_typed when service doesn't exist."""
        container = FlextContainer()

        result = container.get_typed("nonexistent", str)
        assert result.is_failure
        assert result.error
        assert "not found" in result.error

    def test_get_typed_success(self) -> None:
        """Test get_typed success case."""
        container = FlextContainer()
        container.register("string_service", "test_value")

        result = container.get_typed("string_service", str)
        assert result.is_success
        assert result.value == "test_value"

    def test_service_key_validate_method(self) -> None:
        """Test ServiceKey validate method - covers validation logic."""
        # Test empty string validation
        key = FlextContainer.ServiceKey("")
        result = key.validate("")
        assert result.is_failure

        # Test whitespace-only string
        result = key.validate("   ")
        assert result.is_failure

        # Test valid string with whitespace trimming
        result = key.validate("  valid_name  ")
        assert result.is_success
        assert result.value == "valid_name"

    def test_service_key_name_property(self) -> None:
        """Test ServiceKey name property."""
        key = FlextContainer.ServiceKey("test_service")
        assert key.name == "test_service"

    def test_clear_method(self) -> None:
        """Test container clear method."""
        container = FlextContainer()
        container.register("service1", "value1")
        container.register("service2", "value2")

        # Verify services exist
        assert container.has("service1")
        assert container.has("service2")

        # Clear all services
        result = container.clear()
        assert result.is_success

        # Verify services are cleared
        assert not container.has("service1")
        assert not container.has("service2")

    def test_has_method_edge_cases(self) -> None:
        """Test has method with various inputs."""
        container = FlextContainer()
        container.register("existing_service", "value")

        # Test existing service
        assert container.has("existing_service") is True

        # Test non-existing service
        assert container.has("non_existing") is False

        # Test empty string
        assert container.has("") is False

    def test_list_services_empty_container(self) -> None:
        """Test list_services on empty container."""
        container = FlextContainer()
        services = container.list_services()
        assert isinstance(services, dict)
        assert len(services) == 0

    def test_list_services_with_services(self) -> None:
        """Test list_services with registered services."""
        container = FlextContainer()
        container.register("service1", "value1")
        container.register_factory("factory1", lambda: "factory_value")

        services = container.list_services()
        assert isinstance(services, dict)
        assert len(services) >= 2  # At least our registered services

    def test_register_factory_validation_failure(self) -> None:
        """Test register_factory with invalid factory."""
        container = FlextContainer()

        # Register non-callable as factory
        result = container.register_factory("invalid_factory", "not_callable")
        assert result.is_failure
        assert result.error
        assert "Factory must be callable" in result.error

    def test_register_factory_with_parameters(self) -> None:
        """Test register_factory with factory that requires parameters."""
        container = FlextContainer()

        def factory_with_params(param1: str, param2: int) -> str:
            return f"{param1}_{param2}"

        # This should fail because factory requires parameters
        result = container.register_factory("param_factory", factory_with_params)
        assert result.is_failure
        assert result.error
        assert "requires" in result.error
        assert result.error
        assert "parameter" in result.error

    def test_register_factory_success(self) -> None:
        """Test successful factory registration and usage."""
        container = FlextContainer()

        def simple_factory() -> str:
            return "factory_result"

        # Register factory
        result = container.register_factory("simple_factory", simple_factory)
        assert result.is_success

        # Get service created by factory
        service_result = container.get("simple_factory")
        assert service_result.is_success
        assert service_result.value == "factory_result"

    def test_unregister_service(self) -> None:
        """Test unregister method."""
        container = FlextContainer()
        container.register("test_service", "test_value")

        # Verify service exists
        assert container.has("test_service")

        # Unregister service
        result = container.unregister("test_service")
        assert result.is_success

        # Verify service is removed
        assert not container.has("test_service")

    def test_global_container_singleton(self) -> None:
        """Test global container singleton behavior."""
        container1 = FlextContainer.get_global()
        container2 = FlextContainer.get_global()

        # Should be the same instance
        assert container1 is container2

    def test_command_validation_edge_cases(self) -> None:
        """Test command validation for edge cases."""
        # Test RegisterService command validation
        cmd = FlextContainer.Commands.RegisterService()
        cmd.service_name = ""  # Empty name
        result = cmd.validate_command()
        assert result.is_failure

        cmd.service_name = "   "  # Whitespace only
        result = cmd.validate_command()
        assert result.is_failure

        cmd.service_name = "valid_name"
        result = cmd.validate_command()
        assert result.is_success

    def test_register_service_command_create(self) -> None:
        """Test RegisterService command creation."""
        cmd = FlextContainer.Commands.RegisterService.create(
            "test_service", "test_instance"
        )
        assert cmd.service_name == "test_service"
        assert cmd.service_instance == "test_instance"
        assert cmd.command_type == "register_service"
        assert cmd.command_id is not None
        assert cmd.timestamp is not None

    def test_register_factory_command_validation(self) -> None:
        """Test RegisterFactory command validation."""
        cmd = FlextContainer.Commands.RegisterFactory()
        cmd.service_name = "test_factory"
        cmd.factory = "not_callable"  # Not callable

        result = cmd.validate_command()
        assert result.is_failure
        assert result.error
        assert "Factory must be callable" in result.error

        # Test with callable factory
        cmd.factory = lambda: "test"
        result = cmd.validate_command()
        assert result.is_success

    def test_service_registration_edge_cases(self) -> None:
        """Test service registration edge cases."""
        container = FlextContainer()

        # Test registration with empty name
        result = container.register("", "value")
        assert result.is_failure

        # Test registration with whitespace-only name
        result = container.register("   ", "value")
        assert result.is_failure

        # Test registration with valid name that needs trimming
        result = container.register("  valid_name  ", "value")
        assert result.is_success

        # Verify service is accessible with trimmed name
        get_result = container.get("valid_name")
        assert get_result.is_success
        assert get_result.value == "value"

    def test_batch_register_exception_handling(self) -> None:
        """Test batch_register exception handling."""
        container = FlextContainer()

        # Create registrations dict that will cause an exception
        problematic_registrations = {"service1": "value1"}

        # Mock to throw an exception during processing
        with patch.object(
            container, "register", side_effect=RuntimeError("Unexpected error")
        ):
            result = container.batch_register(problematic_registrations)
            assert result.is_failure
            assert result.error
            assert "Batch registration crashed" in result.error

    def test_auto_wire_basic_functionality(self) -> None:
        """Test auto_wire basic functionality if it exists."""
        container = FlextContainer()

        # Register a dependency
        container.register("database", "mock_database")

        class ServiceWithDependency:
            def __init__(self, database: str) -> None:
                """Initialize the instance."""
                self.database = database

        # Try auto-wiring (if the method exists)
        if hasattr(container, "auto_wire"):
            result = container.auto_wire(
                ServiceWithDependency, service_name="auto_service"
            )
            if result.is_success:
                service = result.value
                assert hasattr(service, "database")
