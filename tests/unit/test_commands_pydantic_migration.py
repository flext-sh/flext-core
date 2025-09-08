"""Comprehensive tests for Commands Pydantic migration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

import pytest

from flext_core import FlextCommands, FlextModels, FlextResult


class TestCommandsPydanticMigration:
    """Test the complete migration of Commands to Pydantic."""

    def test_command_model_uses_pydantic(self) -> None:
        """Test that Command class now uses Pydantic CommandModel."""

        # Create a command
        class TestCommand(FlextCommands.Models.Command):
            action: str
            target: str

        # Should be able to create with Pydantic validation
        cmd = TestCommand(command_type="test_command", action="create", target="user")

        # Should have all Pydantic fields from CommandModel
        assert hasattr(cmd, "command_id")
        assert hasattr(cmd, "command_type")
        assert hasattr(cmd, "correlation_id")
        assert hasattr(cmd, "created_at")
        assert hasattr(cmd, "user_id")
        assert hasattr(cmd, "metadata")

        # Should be frozen (immutable)
        with pytest.raises(Exception):  # Pydantic will raise validation error
            cmd.action = "modified"

        # Should have proper ID
        assert cmd.id == cmd.command_id
        assert cmd.command_id.startswith("cmd_")

    def test_query_model_uses_pydantic(self) -> None:
        """Test that Query class now uses Pydantic QueryModel."""

        # Create a query
        class TestQuery(FlextCommands.Models.Query):
            entity_type: str

        query = TestQuery(query_type="test_query", entity_type="user")

        # Should have all Pydantic fields from QueryModel
        assert hasattr(query, "query_id")
        assert hasattr(query, "query_type")
        assert hasattr(query, "created_at")
        assert hasattr(query, "user_id")
        assert hasattr(query, "filters")
        assert hasattr(query, "pagination")

        # Should be frozen
        with pytest.raises(Exception):
            query.entity_type = "modified"

        # Should have proper ID
        assert query.id == query.query_id
        assert query.query_id.startswith("qry_")

    def test_command_handler_with_pydantic_config(self) -> None:
        """Test CommandHandler with HandlerConfig."""
        # Create handler config
        config = FlextModels.SystemConfigs.HandlerConfig(
            handler_id="test_handler_1",
            handler_name="TestHandler",
            handler_type="command",
            can_retry=True,
            max_retries=5,
            timeout_seconds=60,
            priority=8,
        )

        # Create handler with config
        class TestCommandHandler(FlextCommands.Handlers.CommandHandler[dict, str]):
            def handle(self, command: dict) -> FlextResult[str]:
                return FlextResult[str].ok("success")

        handler = TestCommandHandler(handler_config=config)

        # Should use config values
        assert handler.handler_id == "test_handler_1"
        assert handler.handler_name == "TestHandler"
        assert handler._config.can_retry is True
        assert handler._config.max_retries == 5
        assert handler._config.timeout_seconds == 60
        assert handler._config.priority == 8

    def test_command_handler_without_config_creates_default(self) -> None:
        """Test CommandHandler creates default config when not provided."""

        class TestCommandHandler(FlextCommands.Handlers.CommandHandler[dict, str]):
            def handle(self, command: dict) -> FlextResult[str]:
                return FlextResult[str].ok("success")

        handler = TestCommandHandler(handler_name="MyHandler")

        # Should create default config
        assert handler._config is not None
        assert isinstance(handler._config, FlextModels.SystemConfigs.HandlerConfig)
        assert handler._config.handler_name == "MyHandler"
        assert handler._config.handler_type == "command"
        assert handler._config.can_retry is True  # default
        assert handler._config.max_retries == 3  # default

    def test_query_handler_with_pydantic_config(self) -> None:
        """Test QueryHandler with HandlerConfig."""
        config = FlextModels.SystemConfigs.HandlerConfig(
            handler_id="query_handler_1",
            handler_name="TestQueryHandler",
            handler_type="query",
            can_retry=False,
            max_retries=0,
            timeout_seconds=10,
        )

        class TestQueryHandler(FlextCommands.Handlers.QueryHandler[dict, list]):
            def handle(self, query: dict) -> FlextResult[list]:
                return FlextResult[list].ok([])

        handler = TestQueryHandler(handler_config=config)

        # Should use config values
        assert handler.handler_id == "query_handler_1"
        assert handler.handler_name == "TestQueryHandler"
        assert handler._config.handler_type == "query"
        assert handler._config.can_retry is False
        assert handler._config.timeout_seconds == 10

    def test_bus_with_pydantic_config(self) -> None:
        """Test Bus with BusConfig."""
        config = FlextModels.SystemConfigs.BusConfig(
            bus_id="test_bus_1",
            enable_middleware=True,
            enable_async=False,
            max_queue_size=500,
            worker_threads=8,
            enable_metrics=True,
            enable_tracing=False,
        )

        bus = FlextCommands.Bus(bus_config=config)

        # Should use config values
        assert bus._config.bus_id == "test_bus_1"
        assert bus._config.enable_middleware is True
        assert bus._config.enable_async is False
        assert bus._config.max_queue_size == 500
        assert bus._config.worker_threads == 8
        assert bus._config.enable_metrics is True
        assert bus._config.enable_tracing is False

    def test_bus_without_config_creates_default(self) -> None:
        """Test Bus creates default config when not provided."""
        bus = FlextCommands.Bus()

        # Should create default config
        assert bus._config is not None
        assert isinstance(bus._config, FlextModels.SystemConfigs.BusConfig)
        assert bus._config.bus_id.startswith("bus_")
        assert bus._config.enable_middleware is True  # default
        assert bus._config.max_queue_size == 1000  # default
        assert bus._config.worker_threads == 4  # default

    def test_middleware_with_pydantic_config(self) -> None:
        """Test middleware with MiddlewareConfig."""
        bus = FlextCommands.Bus()

        # Create middleware config
        mw_config = FlextModels.SystemConfigs.MiddlewareConfig(
            middleware_id="auth_middleware",
            middleware_type="AuthenticationMiddleware",
            enabled=True,
            order=1,
            config={"auth_required": True, "roles": ["admin", "user"]},
        )

        # Create dummy middleware
        class AuthMiddleware:
            def process(self, command: object, handler: object) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        middleware = AuthMiddleware()

        # Add with config
        bus.add_middleware(middleware, mw_config)

        # Should store config
        assert len(bus._middleware) == 1
        assert bus._middleware[0].middleware_id == "auth_middleware"
        assert bus._middleware[0].middleware_type == "AuthenticationMiddleware"
        assert bus._middleware[0].order == 1
        assert bus._middleware[0].config["auth_required"] is True

    def test_middleware_without_config_creates_default(self) -> None:
        """Test middleware creates default config when not provided."""
        bus = FlextCommands.Bus()

        class LoggingMiddleware:
            def process(self, command: object, handler: object) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        middleware = LoggingMiddleware()
        bus.add_middleware(middleware)

        # Should create default config
        assert len(bus._middleware) == 1
        assert bus._middleware[0].middleware_id == "mw_0"
        assert bus._middleware[0].middleware_type == "LoggingMiddleware"
        assert bus._middleware[0].enabled is True
        assert bus._middleware[0].order == 0

    def test_bus_respects_middleware_config(self) -> None:
        """Test that bus respects middleware enable/disable settings."""
        # Create bus with middleware disabled
        config = FlextModels.SystemConfigs.BusConfig(enable_middleware=False)
        bus = FlextCommands.Bus(bus_config=config)

        # Try to add middleware
        class TestMiddleware:
            def process(self, command: object, handler: object) -> FlextResult[None]:
                return FlextResult[None].ok(None)

        middleware = TestMiddleware()
        bus.add_middleware(middleware)

        # Should not add middleware when disabled
        # (it logs a warning but doesn't add)
        assert len(bus._middleware) == 0

    def test_command_validation_through_pydantic(self) -> None:
        """Test that command validation works through Pydantic."""

        class ValidatedCommand(FlextCommands.Models.Command):
            action: str
            count: int

            def validate_command(self) -> FlextResult[None]:
                if self.count < 0:
                    return FlextResult[None].fail("Count must be positive")
                return FlextResult[None].ok(None)

        # Valid command
        cmd = ValidatedCommand(command_type="validated", action="test", count=5)
        result = cmd.validate_command()
        assert result.success

        # Invalid command
        cmd2 = ValidatedCommand(command_type="validated", action="test", count=-1)
        result2 = cmd2.validate_command()
        assert not result2.success
        assert "must be positive" in result2.error

    def test_query_validation_through_pydantic(self) -> None:
        """Test that query validation works through Pydantic."""

        class ValidatedQuery(FlextCommands.Models.Query):
            entity_type: str
            limit: int = 10

            def validate_query(self) -> FlextResult[None]:
                if self.limit > 100:
                    return FlextResult[None].fail("Limit too high")
                return super().validate_query()

        # Valid query
        query = ValidatedQuery(query_type="search", entity_type="user", limit=50)
        result = query.validate_query()
        assert result.success

        # Invalid query
        query2 = ValidatedQuery(query_type="search", entity_type="user", limit=200)
        result2 = query2.validate_query()
        assert not result2.success
        assert "Limit too high" in result2.error

    def test_handler_config_validation(self) -> None:
        """Test HandlerConfig validation."""
        # Valid config
        config = FlextModels.SystemConfigs.HandlerConfig(
            handler_id="handler_1",
            handler_name="TestHandler",
            handler_type="command",
            max_retries=5,
            timeout_seconds=30,
            priority=7,
        )
        assert config.handler_type == "command"
        assert config.max_retries == 5

        # Invalid handler type
        with pytest.raises(Exception):  # Pydantic validation error
            FlextModels.SystemConfigs.HandlerConfig(
                handler_id="handler_2",
                handler_name="BadHandler",
                handler_type="invalid_type",  # Should fail pattern validation
            )

        # Invalid max_retries
        with pytest.raises(Exception):
            FlextModels.SystemConfigs.HandlerConfig(
                handler_id="handler_3",
                handler_name="BadHandler",
                handler_type="query",
                max_retries=100,  # Over limit
            )

    def test_bus_config_validation(self) -> None:
        """Test BusConfig validation."""
        # Valid config
        config = FlextModels.SystemConfigs.BusConfig(
            max_queue_size=5000, worker_threads=16
        )
        assert config.max_queue_size == 5000
        assert config.worker_threads == 16

        # Invalid queue size
        with pytest.raises(Exception):
            FlextModels.SystemConfigs.BusConfig(
                max_queue_size=50000  # Over limit
            )

        # Invalid worker threads
        with pytest.raises(Exception):
            FlextModels.SystemConfigs.BusConfig(
                worker_threads=100  # Over limit
            )

    def test_middleware_config_validation(self) -> None:
        """Test MiddlewareConfig validation."""
        config = FlextModels.SystemConfigs.MiddlewareConfig(
            middleware_id="mw_1",
            middleware_type="LoggingMiddleware",
            enabled=True,
            order=5,
            config={"log_level": "DEBUG", "format": "json"},
        )

        assert config.middleware_id == "mw_1"
        assert config.middleware_type == "LoggingMiddleware"
        assert config.order == 5
        assert config.config["log_level"] == "DEBUG"

    def test_complete_command_execution_with_pydantic(self) -> None:
        """Test complete command execution flow with Pydantic models."""

        # Create command
        class CreateUserCommand(FlextCommands.Models.Command):
            username: str
            email: str

        # Create handler with config
        handler_config = FlextModels.SystemConfigs.HandlerConfig(
            handler_id="create_user_handler",
            handler_name="CreateUserHandler",
            handler_type="command",
        )

        class CreateUserHandler(
            FlextCommands.Handlers.CommandHandler[CreateUserCommand, str]
        ):
            def handle(self, command: CreateUserCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"User {command.username} created")

        # Create bus with config
        bus_config = FlextModels.SystemConfigs.BusConfig(
            enable_middleware=True, enable_metrics=True
        )
        bus = FlextCommands.Bus(bus_config=bus_config)

        # Register handler
        handler = CreateUserHandler(handler_config=handler_config)
        bus.register_handler(handler)

        # Execute command
        cmd = CreateUserCommand(
            command_type="create_user", username="testuser", email="test@example.com"
        )

        result = bus.execute(cmd)

        # Should succeed with Pydantic models throughout
        assert result.success
        assert "testuser created" in result.value

    def test_middleware_pipeline_with_pydantic(self) -> None:
        """Test middleware pipeline execution with Pydantic configs."""
        # Create bus
        bus = FlextCommands.Bus()

        # Track middleware execution
        execution_order = []

        # Create middleware with different orders
        class FirstMiddleware:
            def process(self, command: object, handler: object) -> FlextResult[None]:
                execution_order.append("first")
                return FlextResult[None].ok(None)

        class SecondMiddleware:
            def process(self, command: object, handler: object) -> FlextResult[None]:
                execution_order.append("second")
                return FlextResult[None].ok(None)

        # Add with specific order
        bus.add_middleware(
            SecondMiddleware(),
            FlextModels.SystemConfigs.MiddlewareConfig(
                middleware_id="mw_second", middleware_type="SecondMiddleware", order=2
            ),
        )

        bus.add_middleware(
            FirstMiddleware(),
            FlextModels.SystemConfigs.MiddlewareConfig(
                middleware_id="mw_first", middleware_type="FirstMiddleware", order=1
            ),
        )

        # Create handler
        class TestHandler(FlextCommands.Handlers.CommandHandler[dict, str]):
            def handle(self, command: dict) -> FlextResult[str]:
                return FlextResult[str].ok("done")

        handler = TestHandler()

        # Apply middleware
        result = bus._apply_middleware({}, handler)

        # Should execute in order
        assert result.success
        assert execution_order == ["first", "second"]
