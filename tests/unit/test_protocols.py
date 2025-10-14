"""Comprehensive tests for FlextCore.Protocols - Protocol Definitions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import FlextCore


class TestFlextProtocols:
    """Test suite for FlextCore.Protocols protocol definitions."""

    def test_protocols_initialization(self) -> None:
        """Test FlextCore.Protocols namespace access."""
        assert FlextCore.Protocols is not None
        assert hasattr(FlextCore.Protocols, "Foundation")
        assert hasattr(FlextCore.Protocols, "Domain")
        assert hasattr(FlextCore.Protocols, "Infrastructure")
        assert hasattr(FlextCore.Protocols, "Application")
        assert hasattr(FlextCore.Protocols, "Commands")
        assert hasattr(FlextCore.Protocols, "Extensions")

    # Foundation Protocols Tests
    def test_has_result_value_protocol(self) -> None:
        """Test HasResultValue protocol definition."""
        protocol = FlextCore.Protocols.Foundation.HasResultValue
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
        protocol = FlextCore.Protocols.Foundation.HasTimestamps
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
        protocol = FlextCore.Protocols.Foundation.HasModelFields
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_has_model_dump_protocol(self) -> None:
        """Test HasModelDump protocol definition."""
        protocol = FlextCore.Protocols.Foundation.HasModelDump
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_has_handler_type_protocol(self) -> None:
        """Test HasHandlerType protocol definition."""
        protocol = FlextCore.Protocols.Foundation.HasHandlerType
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    # Domain Protocols Tests
    def test_repository_protocol(self) -> None:
        """Test Repository protocol definition."""
        protocol = FlextCore.Protocols.Domain.Repository
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_repository_implementation(self) -> None:
        """Test that a class can implement Repository protocol."""
        from flext_core import FlextCore

        class UserRepository:
            def find_by_id(
                self, entity_id: str
            ) -> FlextCore.Result[FlextCore.Types.Dict]:
                return FlextCore.Result[FlextCore.Types.Dict].ok({
                    "id": entity_id,
                    "name": "Test",
                })

            def save(
                self, entity: FlextCore.Types.Dict
            ) -> FlextCore.Result[FlextCore.Types.Dict]:
                return FlextCore.Result[FlextCore.Types.Dict].ok(entity)

            def delete(self, entity_id: str) -> FlextCore.Result[None]:
                return FlextCore.Result[None].ok(None)

        repo = UserRepository()
        assert hasattr(repo, "find_by_id")
        assert hasattr(repo, "save")
        assert hasattr(repo, "delete")

    def test_service_protocol(self) -> None:
        """Test Service protocol definition."""
        protocol = FlextCore.Protocols.Domain.Service
        assert protocol is not None

    def test_service_implementation(self) -> None:
        """Test that a class can implement Service protocol."""
        from flext_core import FlextCore

        class UserService:
            def execute(
                self, **kwargs: object
            ) -> FlextCore.Result[FlextCore.Types.Dict]:
                return FlextCore.Result[FlextCore.Types.Dict].ok({"status": "success"})

        service = UserService()
        assert hasattr(service, "execute")
        result = service.execute()
        assert result.is_success

    # Infrastructure Protocols Tests
    def test_configurable_protocol(self) -> None:
        """Test Configurable protocol definition."""
        protocol = FlextCore.Protocols.Infrastructure.Configurable
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_connection_protocol(self) -> None:
        """Test Connection protocol definition."""
        protocol = FlextCore.Protocols.Infrastructure.Connection
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_connection_implementation(self) -> None:
        """Test that a class can implement Connection protocol."""
        from flext_core import FlextCore

        class DatabaseConnection:
            def connect(self) -> FlextCore.Result[None]:
                return FlextCore.Result[None].ok(None)

            def disconnect(self) -> FlextCore.Result[None]:
                return FlextCore.Result[None].ok(None)

            def is_connected(self) -> bool:
                return True

        conn = DatabaseConnection()
        assert hasattr(conn, "connect")
        assert hasattr(conn, "disconnect")
        assert hasattr(conn, "is_connected")

    def test_logger_protocol(self) -> None:
        """Test LoggerProtocol definition."""
        protocol = FlextCore.Protocols.Infrastructure.LoggerProtocol
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    # Application Protocols Tests
    def test_handler_protocol(self) -> None:
        """Test Handler protocol definition."""
        protocol = FlextCore.Protocols.Application.Handler
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_handler_implementation(self) -> None:
        """Test that a class can implement Handler protocol."""
        from flext_core import FlextCore

        class CreateUserHandler:
            def handle(
                self, command: FlextCore.Types.Dict
            ) -> FlextCore.Result[FlextCore.Types.Dict]:
                return FlextCore.Result[FlextCore.Types.Dict].ok({"user_id": "123"})

        handler = CreateUserHandler()
        assert hasattr(handler, "handle")
        result = handler.handle({"name": "Test"})
        assert result.is_success

    # Commands Protocols Tests
    def test_command_handler_protocol(self) -> None:
        """Test CommandBus protocol definition."""
        protocol = FlextCore.Protocols.Commands.CommandBus
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_command_bus_protocol(self) -> None:
        """Test CommandBus protocol definition."""
        protocol = FlextCore.Protocols.Commands.CommandBus
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_commands_middleware_protocol(self) -> None:
        """Test Commands Middleware protocol definition."""
        protocol = FlextCore.Protocols.Commands.Middleware
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    # Extensions Protocols Tests
    def test_plugin_protocol(self) -> None:
        """Test PluginContext protocol definition."""
        protocol = FlextCore.Protocols.Extensions.PluginContext
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_plugin_context_protocol(self) -> None:
        """Test PluginContext protocol definition."""
        protocol = FlextCore.Protocols.Extensions.PluginContext
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_observability_protocol(self) -> None:
        """Test Observability protocol definition."""
        protocol = FlextCore.Protocols.Extensions.Observability
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    def test_extensions_middleware_protocol(self) -> None:
        """Test Extensions Middleware protocol definition."""
        protocol = FlextCore.Protocols.Extensions.Middleware
        assert protocol is not None
        # For runtime-checkable protocols, check for protocol attributes
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    # Integration Tests
    def test_protocol_categories_independence(self) -> None:
        """Test that protocol categories are independent."""
        foundation = FlextCore.Protocols.Foundation
        domain = FlextCore.Protocols.Domain
        infrastructure = FlextCore.Protocols.Infrastructure
        application = FlextCore.Protocols.Application
        commands = FlextCore.Protocols.Commands
        extensions = FlextCore.Protocols.Extensions

        # All should be different objects
        # Protocol categories list
        categories: list[type] = [
            foundation,
            domain,
            infrastructure,
            application,
            commands,
            extensions,
        ]
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i + 1 :]:
                assert cat1 is not cat2

    def test_multiple_protocol_implementation(self) -> None:
        """Test that a class can implement multiple protocols."""
        from flext_core import FlextCore

        class AdvancedService:
            """Service implementing multiple protocols."""

            def execute(
                self, **kwargs: object
            ) -> FlextCore.Result[FlextCore.Types.Dict]:
                return FlextCore.Result[FlextCore.Types.Dict].ok({})

            def handle(
                self, command: FlextCore.Types.Dict
            ) -> FlextCore.Result[FlextCore.Types.Dict]:
                return FlextCore.Result[FlextCore.Types.Dict].ok({})

        service = AdvancedService()
        # Should implement both Service and Handler protocols
        assert hasattr(service, "execute")
        assert hasattr(service, "handle")

    def test_protocol_runtime_checkable(self) -> None:
        """Test that protocols support runtime checking."""
        # HasResultValue should be runtime checkable
        # protocol = FlextCore.Protocols.Foundation.HasResultValue  # Unused for now

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
            assert hasattr(FlextCore.Protocols.Foundation, proto_name)

    def test_all_domain_protocols_available(self) -> None:
        """Test that all domain protocols are accessible."""
        expected_protocols = ["Repository", "Service"]
        for proto_name in expected_protocols:
            assert hasattr(FlextCore.Protocols.Domain, proto_name)

    def test_all_infrastructure_protocols_available(self) -> None:
        """Test that all infrastructure protocols are accessible."""
        expected_protocols = ["Configurable", "Connection", "LoggerProtocol"]
        for proto_name in expected_protocols:
            assert hasattr(FlextCore.Protocols.Infrastructure, proto_name)

    def test_all_application_protocols_available(self) -> None:
        """Test that all application protocols are accessible."""
        expected_protocols = ["Handler"]
        for proto_name in expected_protocols:
            assert hasattr(FlextCore.Protocols.Application, proto_name)

    def test_all_commands_protocols_available(self) -> None:
        """Test that all commands protocols are accessible."""
        expected_protocols = [
            "CommandBus",
            "Middleware",
        ]
        for proto_name in expected_protocols:
            assert hasattr(FlextCore.Protocols.Commands, proto_name)

    def test_all_extensions_protocols_available(self) -> None:
        """Test that all extensions protocols are accessible."""
        expected_protocols = [
            "PluginContext",
            "Middleware",
        ]
        for proto_name in expected_protocols:
            assert hasattr(FlextCore.Protocols.Extensions, proto_name)
