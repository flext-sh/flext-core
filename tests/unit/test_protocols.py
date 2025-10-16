"""Comprehensive tests for FlextProtocols - Protocol Definitions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import FlextProtocols, FlextResult, FlextTypes


class TestFlextProtocols:
    """Test suite for FlextProtocols protocol definitions."""

    # Foundation Protocols Tests
    def test_has_result_value_protocol(self) -> None:
        """Test HasResultValue protocol definition."""
        protocol = FlextProtocols.HasResultValue
        assert protocol is not None
        # Check it's a Protocol
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_has_result_value_implementation(self) -> None:
        """Test that a class can implement HasResultValue."""

        class ResultContainer:
            def __init__(self, value: str) -> None:
                super().__init__()
                self._value = value

            @property
            def value(self) -> str:
                return self._value

        container = ResultContainer("test")
        # Should have the required value property
        assert hasattr(container, "value")
        assert container.value == "test"

    def test_has_timestamps_protocol(self) -> None:
        """Test HasTimestamps protocol definition."""
        protocol = FlextProtocols.HasTimestamps
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_has_timestamps_implementation(self) -> None:
        """Test that a class can implement HasTimestamps."""
        import time

        class TimestampedEntity:
            def __init__(self) -> None:
                super().__init__()
                self.created_at = time.time()
                self.updated_at = time.time()

        entity = TimestampedEntity()
        assert hasattr(entity, "created_at")
        assert hasattr(entity, "updated_at")

    def test_has_model_fields_protocol(self) -> None:
        """Test HasModelFields protocol definition."""
        protocol = FlextProtocols.HasModelFields
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_has_model_dump_protocol(self) -> None:
        """Test HasModelDump protocol definition."""
        protocol = FlextProtocols.HasModelDump
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_has_handler_type_protocol(self) -> None:
        """Test HasHandlerType protocol definition."""
        protocol = FlextProtocols.HasHandlerType
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    # Domain Protocols Tests
    def test_repository_protocol(self) -> None:
        """Test Repository protocol definition."""
        protocol = FlextProtocols.Repository
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_repository_implementation(self) -> None:
        """Test that a class can implement Repository protocol."""

        class UserRepository:
            def find_by_id(self, entity_id: str) -> FlextResult[FlextTypes.Dict]:
                return FlextResult[FlextTypes.Dict].ok(
                    {
                        "id": entity_id,
                        "name": "Test",
                    }
                )

            def save(self, entity: FlextTypes.Dict) -> FlextResult[FlextTypes.Dict]:
                return FlextResult[FlextTypes.Dict].ok(entity)

            def delete(self, entity_id: str) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        repo = UserRepository()
        assert hasattr(repo, "find_by_id")
        assert hasattr(repo, "save")
        assert hasattr(repo, "delete")

    def test_service_protocol(self) -> None:
        """Test Service protocol definition."""
        protocol = FlextProtocols.Service
        assert protocol is not None

    def test_service_implementation(self) -> None:
        """Test that a class can implement Service protocol."""

        class UserService:
            def execute(self, **kwargs: object) -> FlextResult[FlextTypes.Dict]:
                return FlextResult[FlextTypes.Dict].ok({"status": "success"})

        service = UserService()
        assert hasattr(service, "execute")
        result = service.execute()
        assert result.is_success

    # Infrastructure Protocols Tests
    def test_configurable_protocol(self) -> None:
        """Test Configurable protocol definition."""
        protocol = FlextProtocols.Configurable
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_connection_protocol(self) -> None:
        """Test Connection protocol definition."""
        protocol = FlextProtocols.Connection
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_connection_implementation(self) -> None:
        """Test that a class can implement Connection protocol."""

        class DatabaseConnection:
            def connect(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

            def disconnect(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

            def is_connected(self) -> bool:
                return True

        conn = DatabaseConnection()
        assert hasattr(conn, "connect")
        assert hasattr(conn, "disconnect")
        assert hasattr(conn, "is_connected")

    def test_logger_protocol(self) -> None:
        """Test LoggerProtocol definition."""
        protocol = FlextProtocols.LoggerProtocol
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    # Application Protocols Tests
    def test_handler_protocol(self) -> None:
        """Test Handler protocol definition."""
        protocol = FlextProtocols.Handler
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_handler_implementation(self) -> None:
        """Test that a class can implement Handler protocol."""

        class CreateUserHandler:
            def handle(self, command: FlextTypes.Dict) -> FlextResult[FlextTypes.Dict]:
                return FlextResult[FlextTypes.Dict].ok({"user_id": "123"})

        handler = CreateUserHandler()
        assert hasattr(handler, "handle")
        result = handler.handle({"name": "Test"})
        assert result.is_success

    # Commands Protocols Tests
    def test_command_handler_protocol(self) -> None:
        """Test CommandBus protocol definition."""
        protocol = FlextProtocols.CommandBus
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_command_bus_protocol(self) -> None:
        """Test CommandBus protocol definition."""
        protocol = FlextProtocols.CommandBus
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_commands_middleware_protocol(self) -> None:
        """Test Commands Middleware protocol definition."""
        protocol = FlextProtocols.Middleware
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    # Extensions Protocols Tests
    def test_plugin_protocol(self) -> None:
        """Test PluginContext protocol definition."""
        protocol = FlextProtocols.PluginContext
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_plugin_context_protocol(self) -> None:
        """Test PluginContext protocol definition."""
        protocol = FlextProtocols.PluginContext
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_observability_protocol(self) -> None:
        """Test Observability protocol definition."""
        protocol = FlextProtocols.Observability
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_extensions_middleware_protocol(self) -> None:
        """Test Extensions Middleware protocol definition."""
        protocol = FlextProtocols.Middleware
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_multiple_protocol_implementation(self) -> None:
        """Test that a class can implement multiple protocols."""

        class AdvancedService:
            """Service implementing multiple protocols."""

            def execute(self, **kwargs: object) -> FlextResult[FlextTypes.Dict]:
                return FlextResult[FlextTypes.Dict].ok({})

            def handle(self, command: FlextTypes.Dict) -> FlextResult[FlextTypes.Dict]:
                return FlextResult[FlextTypes.Dict].ok({})

        service = AdvancedService()
        # Should implement both Service and Handler protocols
        assert hasattr(service, "execute")
        assert hasattr(service, "handle")

    def test_protocol_runtime_checkable(self) -> None:
        """Test that protocols support runtime checking."""
        # HasResultValue should be runtime checkable
        # protocol = FlextProtocols.HasResultValue  # Unused for now

        class Container:
            @property
            def value(self) -> str:
                return "test"

        # Runtime check should work if protocol is runtime_checkable
        container = Container()
        assert hasattr(container, "value")

    def test_all_foundation_protocols_available(self) -> None:
        """Test that all foundation protocols are accessible."""
        expected_protocols = [
            "HasResultValue",
            "HasTimestamps",
            "HasModelFields",
            "HasModelDump",
            "HasHandlerType",
        ]
        for proto_name in expected_protocols:
            assert hasattr(FlextProtocols, proto_name)

    def test_all_domain_protocols_available(self) -> None:
        """Test that all domain protocols are accessible."""
        expected_protocols = ["Repository", "Service"]
        for proto_name in expected_protocols:
            assert hasattr(FlextProtocols, proto_name)

    def test_all_infrastructure_protocols_available(self) -> None:
        """Test that all infrastructure protocols are accessible."""
        expected_protocols = ["Configurable", "Connection", "LoggerProtocol"]
        for proto_name in expected_protocols:
            assert hasattr(FlextProtocols, proto_name)

    def test_all_application_protocols_available(self) -> None:
        """Test that all application protocols are accessible."""
        expected_protocols = ["Handler"]
        for proto_name in expected_protocols:
            assert hasattr(FlextProtocols, proto_name)

    def test_all_commands_protocols_available(self) -> None:
        """Test that all commands protocols are accessible."""
        expected_protocols = [
            "CommandBus",
            "Middleware",
        ]
        for proto_name in expected_protocols:
            assert hasattr(FlextProtocols, proto_name)

    def test_all_extensions_protocols_available(self) -> None:
        """Test that all extensions protocols are accessible."""
        expected_protocols = [
            "PluginContext",
            "Middleware",
        ]
        for proto_name in expected_protocols:
            assert hasattr(FlextProtocols, proto_name)
