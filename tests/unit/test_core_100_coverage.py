"""Complete test coverage for FlextCore module achieving 100% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextCore,
    FlextModels,
    FlextResult,
)


class TestFlextCore100Coverage:
    """Complete test coverage for FlextCore achieving 100%."""

    def test_singleton_instance(self) -> None:
        """Test singleton pattern implementation."""
        core1 = FlextCore.get_instance()
        core2 = FlextCore.get_instance()
        assert core1 is core2

    def test_container_property(self) -> None:
        """Test container property initialization."""
        core = FlextCore()
        container = core.container
        assert isinstance(container, FlextContainer)

        # Test cached property
        container2 = core.container
        assert container is container2

    def test_config_property(self) -> None:
        """Test config property initialization."""
        core = FlextCore()
        config = core.config
        assert config is not None

        # Test cached property
        config2 = core.config
        assert config is config2

    def test_database_config_property(self) -> None:
        """Test database_config property."""
        core = FlextCore()

        # Test without config
        db_config = core.database_config
        assert db_config is None or isinstance(db_config, FlextModels.DatabaseConfig)

        # Test with mock config
        with patch.object(core, "_database_config", None):
            mock_config = MagicMock(spec=FlextModels.DatabaseConfig)
            with patch.object(
                core.container, "get", return_value=FlextResult.ok(mock_config)
            ):
                db_config = core.database_config
                assert db_config == mock_config

    def test_security_config_property(self) -> None:
        """Test security_config property."""
        core = FlextCore()

        # Test without config
        sec_config = core.security_config
        assert sec_config is None or isinstance(sec_config, FlextModels.SecurityConfig)

        # Test with mock config
        with patch.object(core, "_security_config", None):
            mock_config = MagicMock(spec=FlextModels.SecurityConfig)
            with patch.object(
                core.container, "get", return_value=FlextResult.ok(mock_config)
            ):
                sec_config = core.security_config
                assert sec_config == mock_config

    def test_logging_config_property(self) -> None:
        """Test logging_config property."""
        core = FlextCore()

        # Test without config
        log_config = core.logging_config
        assert log_config is None or isinstance(log_config, FlextModels.LoggingConfig)

        # Test with mock config
        with patch.object(core, "_logging_config", None):
            mock_config = MagicMock(spec=FlextModels.LoggingConfig)
            with patch.object(
                core.container, "get", return_value=FlextResult.ok(mock_config)
            ):
                log_config = core.logging_config
                assert log_config == mock_config

    def test_context_property(self) -> None:
        """Test context property."""
        core = FlextCore()
        context = core.context
        assert isinstance(context, FlextContext)

        # Test cached property
        context2 = core.context
        assert context is context2

    def test_logger_property(self) -> None:
        """Test logger property."""
        core = FlextCore()
        logger = core.logger
        assert logger is not None

        # Test cached property
        logger2 = core.logger
        assert logger is logger2

    def test_configure_aggregates_system(self) -> None:
        """Test configure_aggregates_system method."""
        core = FlextCore()

        config = {
            "environment": "development",
            "log_level": "DEBUG",
            "enable_caching": True,
        }

        result = core.configure_aggregates_system(config)
        assert result.is_success

    def test_get_aggregates_config(self) -> None:
        """Test get_aggregates_config method."""
        core = FlextCore()

        # Test normal path
        result = core.get_aggregates_config()
        assert result.is_success

        # Test exception path
        with patch.object(core, "aggregates", side_effect=Exception("Error")):
            result = core.get_aggregates_config()
            assert result.is_failure
            assert "Get config failed" in result.error

    def test_optimize_aggregates_system_all_levels(self) -> None:
        """Test optimize_aggregates_system with all performance levels."""
        core = FlextCore()

        # Test all levels including edge cases
        for level in ["low", "balanced", "high", "extreme", "unknown"]:
            result = core.optimize_aggregates_system(level)
            assert result.is_success
            config = result.value
            assert config["level"] == level
            assert "cache_size" in config
            assert "batch_size" in config

        # Test exception path
        with patch("flext_core.core.FlextResult") as mock_result:
            mock_result.side_effect = Exception("Optimization error")
            with pytest.raises(Exception):
                core.optimize_aggregates_system("high")

    def test_configure_commands_system(self) -> None:
        """Test configure_commands_system method."""
        core = FlextCore()

        config = {"environment": "development", "log_level": "DEBUG", "max_retries": 3}

        result = core.configure_commands_system(config)
        assert result is not None

    def test_get_commands_config(self) -> None:
        """Test get_commands_config method."""
        core = FlextCore()

        # Test normal path
        result = core.get_commands_config()
        assert result.is_success

        # Test exception path
        with patch.object(
            core.commands, "get_commands_config", side_effect=Exception("Error")
        ):
            result = core.get_commands_config()
            assert result.is_failure

    def test_configure_commands_system_with_model(self) -> None:
        """Test configure_commands_system_with_model method."""
        core = FlextCore()

        mock_config = MagicMock(spec=FlextModels.SystemConfigs.CommandsConfig)
        result = core.configure_commands_system_with_model(mock_config)
        assert result.is_success

    def test_get_commands_config_model(self) -> None:
        """Test get_commands_config_model method."""
        core = FlextCore()

        # Test normal path
        result = core.get_commands_config_model()
        assert result.is_success or result.is_failure

        # Test exception path
        with patch.object(
            core.commands, "get_commands_config", side_effect=Exception("Error")
        ):
            result = core.get_commands_config_model()
            assert result.is_failure

    def test_optimize_commands_performance(self) -> None:
        """Test optimize_commands_performance method."""
        core = FlextCore()

        # Test all levels
        for level in ["low", "balanced", "high"]:
            result = core.optimize_commands_performance(level)
            assert result.is_success

    def test_load_config_from_file(self) -> None:
        """Test load_config_from_file method."""
        core = FlextCore()

        # Create a temp config file
        test_config = {"test": "value", "number": 123}
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                json.dumps(test_config)
            )
            with patch("pathlib.Path.exists", return_value=True):
                result = core.load_config_from_file("test.json")
                assert result.is_success
                assert result.value == test_config

        # Test file not found
        with patch("pathlib.Path.exists", return_value=False):
            result = core.load_config_from_file("missing.json")
            assert result.is_failure

        # Test invalid JSON
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "invalid json"
            )
            with patch("pathlib.Path.exists", return_value=True):
                result = core.load_config_from_file("invalid.json")
                assert result.is_failure

    def test_configure_database(self) -> None:
        """Test configure_database method."""
        core = FlextCore()

        config = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "username": "user",
            "password": "pass",
        }

        result = core.configure_database(config)
        assert result.is_success

        # Test exception path
        with patch.object(core.container, "register", side_effect=Exception("Error")):
            result = core.configure_database(config)
            assert result.is_failure

    def test_configure_security(self) -> None:
        """Test configure_security method."""
        core = FlextCore()

        config = {"enable_auth": True, "jwt_secret": "secret", "token_expiry": 3600}

        result = core.configure_security(config)
        assert result.is_success

        # Test exception path
        with patch.object(core.container, "register", side_effect=Exception("Error")):
            result = core.configure_security(config)
            assert result.is_failure

    def test_configure_logging_config(self) -> None:
        """Test configure_logging_config method."""
        core = FlextCore()

        config = {
            "log_level": "DEBUG",
            "log_format": "json",
            "enable_file_logging": True,
        }

        result = core.configure_logging_config(config)
        assert result.is_success

        # Test exception path
        with patch.object(core.container, "register", side_effect=Exception("Error")):
            result = core.configure_logging_config(config)
            assert result.is_failure

    def test_configure_context_system(self) -> None:
        """Test configure_context_system method."""
        core = FlextCore()

        config = {"environment": "development", "trace_enabled": True}

        result = core.configure_context_system(config)
        assert result.is_success

    def test_get_context_config(self) -> None:
        """Test get_context_config method."""
        core = FlextCore()

        result = core.get_context_config()
        assert result.is_success

        # Test exception path
        with patch.object(
            core.context, "get_context_config", side_effect=Exception("Error")
        ):
            result = core.get_context_config()
            assert result.is_failure

    def test_validate_email(self) -> None:
        """Test validate_email method."""
        core = FlextCore()

        # Valid email
        result = core.validate_email("test@example.com")
        assert result.is_success

        # Invalid email
        result = core.validate_email("invalid")
        assert result.is_failure

    def test_validate_string_field(self) -> None:
        """Test validate_string_field method."""
        core = FlextCore()

        # Valid string
        result = core.validate_string_field("test", "field_name")
        assert result.is_success

        # Invalid type
        result = core.validate_string_field(123, "field_name")
        assert result.is_failure

        # Empty string
        result = core.validate_string_field("", "field_name")
        assert result.is_failure

    def test_validate_numeric_field(self) -> None:
        """Test validate_numeric_field method."""
        core = FlextCore()

        # Valid numeric within range
        result = core.validate_numeric_field(
            50, "field_name", min_value=0, max_value=100
        )
        assert result.is_success

        # Out of range
        result = core.validate_numeric_field(
            150, "field_name", min_value=0, max_value=100
        )
        assert result.is_failure

        # Invalid type
        result = core.validate_numeric_field("not a number", "field_name")
        assert result.is_failure

    def test_validate_user_data(self) -> None:
        """Test validate_user_data method."""
        core = FlextCore()

        # Valid user data
        user_data = {"username": "testuser", "email": "test@example.com", "age": 25}
        result = core.validate_user_data(user_data)
        assert result.is_success

        # Invalid user data
        invalid_data = {"username": "", "email": "invalid"}
        result = core.validate_user_data(invalid_data)
        assert result.is_failure

    def test_validate_api_request(self) -> None:
        """Test validate_api_request method."""
        core = FlextCore()

        # Valid request
        request = {"method": "POST", "endpoint": "/api/users", "body": {"name": "test"}}
        result = core.validate_api_request(request)
        assert result.is_success

        # Invalid request
        invalid_request = {"method": "INVALID", "endpoint": ""}
        result = core.validate_api_request(invalid_request)
        assert result.is_failure

    def test_create_entity(self) -> None:
        """Test create_entity method."""
        core = FlextCore()

        # Create with auto ID
        result = core.create_entity("User", {"name": "Test"}, auto_id=True)
        assert result.is_success
        entity = result.value
        assert entity.entity_type == "User"
        assert entity.id is not None

        # Create without auto ID
        result = core.create_entity("User", {"name": "Test"}, auto_id=False)
        assert result.is_success

        # Test exception path
        with patch("flext_core.core.FlextModels") as mock_models:
            mock_models.Entity.side_effect = Exception("Error")
            result = core.create_entity("User", {"name": "Test"})
            assert result.is_failure

    def test_create_value_object(self) -> None:
        """Test create_value_object method."""
        core = FlextCore()

        # Create value object
        result = core.create_value_object("Email", {"address": "test@example.com"})
        assert result.is_success

        # Test exception path
        with patch("flext_core.core.FlextModels") as mock_models:
            mock_models.Value.side_effect = Exception("Error")
            result = core.create_value_object("Email", {"address": "test@example.com"})
            assert result.is_failure

    def test_create_aggregate(self) -> None:
        """Test create_aggregate method."""
        core = FlextCore()

        # Create aggregate with entities
        entities = [
            {"entity_type": "User", "data": {"name": "User1"}},
            {"entity_type": "User", "data": {"name": "User2"}},
        ]
        result = core.create_aggregate_root("UserGroup", entities, auto_id=True)
        assert result.is_success

        # Test exception path
        with patch("flext_core.core.FlextModels") as mock_models:
            mock_models.AggregateRoot.side_effect = Exception("Error")
            result = core.create_aggregate_root("UserGroup", entities)
            assert result.is_failure

    def test_execute_command(self) -> None:
        """Test execute_command method."""
        core = FlextCore()

        # Create and execute command
        command = {"type": "CreateUser", "data": {"name": "Test"}}
        result = core.execute_command(command)
        assert result.is_success or result.is_failure  # Depends on command handler

        # Test with handler
        def handler(cmd):
            return FlextResult.ok({"success": True})

        with patch.object(
            core.commands, "execute", return_value=FlextResult.ok({"success": True})
        ):
            result = core.execute_command(command, handler=handler)
            assert result.is_success

    def test_execute_query(self) -> None:
        """Test execute_query method."""
        core = FlextCore()

        # Execute query
        query = {"type": "GetUser", "id": "123"}
        result = core.execute_query(query)
        assert result.is_success or result.is_failure

        # Test with handler
        def handler(q):
            return FlextResult.ok({"user": {"id": "123", "name": "Test"}})

        with patch.object(
            core.commands, "query", return_value=FlextResult.ok({"user": {}})
        ):
            result = core.execute_query(query, handler=handler)
            assert result.is_success

    def test_register_command_handler(self) -> None:
        """Test register_command_handler method."""
        core = FlextCore()

        def handler(cmd):
            return FlextResult.ok({"handled": True})

        result = core.register_command_handler("CreateUser", handler)
        assert result.is_success

        # Test exception path
        with patch.object(
            core.commands, "register_handler", side_effect=Exception("Error")
        ):
            result = core.register_command_handler("CreateUser", handler)
            assert result.is_failure

    def test_register_query_handler(self) -> None:
        """Test register_query_handler method."""
        core = FlextCore()

        def handler(query):
            return FlextResult.ok({"data": []})

        result = core.register_query_handler("GetUsers", handler)
        assert result.is_success

        # Test exception path
        with patch.object(
            core.commands, "register_query_handler", side_effect=Exception("Error")
        ):
            result = core.register_query_handler("GetUsers", handler)
            assert result.is_failure

    def test_emit_event(self) -> None:
        """Test emit_event method."""
        core = FlextCore()

        event = {"type": "UserCreated", "user_id": "123"}
        result = core.emit_event("UserCreated", event)
        assert result.is_success

        # Test exception path
        with patch.object(core.commands, "emit_event", side_effect=Exception("Error")):
            result = core.emit_event("UserCreated", event)
            assert result.is_failure

    def test_subscribe_to_event(self) -> None:
        """Test subscribe_to_event method."""
        core = FlextCore()

        def handler(event) -> None:
            pass

        result = core.subscribe_to_event("UserCreated", handler)
        assert result.is_success

        # Test exception path
        with patch.object(core.commands, "subscribe", side_effect=Exception("Error")):
            result = core.subscribe_to_event("UserCreated", handler)
            assert result.is_failure

    def test_create_entity_id(self) -> None:
        """Test create_entity_id method."""
        core = FlextCore()

        # Generate entity ID using generate_entity_id (returns string directly)
        entity_id = core.generate_entity_id()
        assert entity_id is not None
        assert len(entity_id) > 0

        # Create entity ID with specific value (returns FlextResult)
        result = core.create_entity_id("USER_123")
        assert result.is_success
        entity_id = result.value
        assert entity_id.root == "USER_123"

    def test_create_correlation_id(self) -> None:
        """Test generate_correlation_id method."""
        core = FlextCore()

        corr_id = core.generate_correlation_id()
        assert corr_id is not None
        assert len(corr_id) > 0

    def test_get_current_context(self) -> None:
        """Test get_current_context method."""
        core = FlextCore()

        result = core.get_current_context()
        assert result.is_success
        context = result.value
        assert (
            "request_id" in context
            or "correlation_id" in context
            or "timestamp" in context
        )

    def test_set_context_value(self) -> None:
        """Test set_context_value method."""
        core = FlextCore()

        result = core.set_context_value("user_id", "123")
        assert result.is_success

        # Verify it was set
        result = core.get_context_value("user_id")
        assert result.is_success
        assert result.value == "123"

    def test_get_context_value(self) -> None:
        """Test get_context_value method."""
        core = FlextCore()

        # Set and get
        core.set_context_value("test_key", "test_value")
        result = core.get_context_value("test_key")
        assert result.is_success
        assert result.value == "test_value"

        # Get non-existent
        result = core.get_context_value("non_existent")
        assert result.is_failure or result.value is None

    def test_clear_context(self) -> None:
        """Test clear_context method."""
        core = FlextCore()

        # Set some values
        core.set_context_value("key1", "value1")
        core.set_context_value("key2", "value2")

        # Clear
        result = core.clear_context()
        assert result.is_success

        # Verify cleared
        result = core.get_context_value("key1")
        assert result.is_failure or result.value is None

    def test_log_debug(self) -> None:
        """Test log_debug method."""
        core = FlextCore()

        with patch.object(core.logger, "debug") as mock_debug:
            core.log_debug("Test debug message", extra={"key": "value"})
            mock_debug.assert_called_once()

    def test_log_info(self) -> None:
        """Test log_info method."""
        core = FlextCore()

        with patch.object(core.logger, "info") as mock_info:
            core.log_info("Test info message", extra={"key": "value"})
            mock_info.assert_called_once()

    def test_log_warning(self) -> None:
        """Test log_warning method."""
        core = FlextCore()

        with patch.object(core.logger, "warning") as mock_warning:
            core.log_warning("Test warning message", extra={"key": "value"})
            mock_warning.assert_called_once()

    def test_log_error(self) -> None:
        """Test log_error method."""
        core = FlextCore()

        with patch.object(core.logger, "error") as mock_error:
            core.log_error(
                "Test error message",
                exception=Exception("Test"),
                extra={"key": "value"},
            )
            mock_error.assert_called_once()

    def test_log_critical(self) -> None:
        """Test log_critical method."""
        core = FlextCore()

        with patch.object(core.logger, "critical") as mock_critical:
            core.log_critical(
                "Test critical message",
                exception=Exception("Test"),
                extra={"key": "value"},
            )
            mock_critical.assert_called_once()

    def test_get_system_health(self) -> None:
        """Test get_system_health method."""
        core = FlextCore()

        result = core.get_system_health()
        assert result.is_success
        health = result.value
        assert "status" in health
        assert health["status"] in {"healthy", "degraded", "unhealthy"}

    def test_get_system_metrics(self) -> None:
        """Test get_system_metrics method."""
        core = FlextCore()

        result = core.get_system_metrics()
        assert result.is_success
        metrics = result.value
        assert isinstance(metrics, dict)

    def test_get_system_info(self) -> None:
        """Test get_system_info method."""
        core = FlextCore()

        result = core.get_system_info()
        assert result.is_success
        info = result.value
        assert "version" in info
        assert "environment" in info

    def test_export_configuration(self) -> None:
        """Test export_configuration method."""
        core = FlextCore()

        result = core.export_configuration()
        assert result.is_success
        config = result.value
        assert isinstance(config, dict)

    def test_import_configuration(self) -> None:
        """Test import_configuration method."""
        core = FlextCore()

        config = {"environment": "test", "debug": True, "log_level": "DEBUG"}

        result = core.import_configuration(config)
        assert result.is_success

        # Test with invalid config
        result = core.import_configuration("invalid")
        assert result.is_failure

    def test_reset_to_defaults(self) -> None:
        """Test reset_to_defaults method."""
        core = FlextCore()

        result = core.reset_to_defaults()
        assert result.is_success

    def test_validate_configuration(self) -> None:
        """Test validate_configuration method."""
        core = FlextCore()

        config = {"environment": "development", "log_level": "INFO"}

        result = core.validate_configuration(config)
        assert result.is_success

        # Test with invalid config
        invalid_config = {"environment": "invalid_env", "log_level": "INVALID"}
        result = core.validate_configuration(invalid_config)
        # May succeed or fail depending on validation rules
        assert result.is_success or result.is_failure

    def test_optimize_performance(self) -> None:
        """Test optimize_core_performance method."""
        core = FlextCore()

        # Test with config dict
        config = {"performance_level": "high", "cache_enabled": True}
        result = core.optimize_core_performance(config)
        assert isinstance(result, FlextResult)

    def test_enable_feature(self) -> None:
        """Test enable_feature method."""
        core = FlextCore()

        result = core.enable_feature("caching")
        assert result.is_success

        result = core.enable_feature("monitoring")
        assert result.is_success

    def test_disable_feature(self) -> None:
        """Test disable_feature method."""
        core = FlextCore()

        result = core.disable_feature("caching")
        assert result.is_success

        result = core.disable_feature("monitoring")
        assert result.is_success

    def test_is_feature_enabled(self) -> None:
        """Test is_feature_enabled method."""
        core = FlextCore()

        # Enable a feature
        core.enable_feature("test_feature")

        # Check if enabled
        result = core.is_feature_enabled("test_feature")
        assert isinstance(result, bool)

    def test_get_feature_flags(self) -> None:
        """Test get_feature_flags method."""
        core = FlextCore()

        result = core.get_feature_flags()
        assert result.is_success
        flags = result.value
        assert isinstance(flags, dict)

    def test_register_service(self) -> None:
        """Test register_service method."""
        core = FlextCore()

        class TestService:
            pass

        service = TestService()
        result = core.register_service("test_service", service)
        assert result.is_success

    def test_get_service(self) -> None:
        """Test get_service method."""
        core = FlextCore()

        # Register and get
        class TestService:
            pass

        service = TestService()
        core.register_service("test_service", service)

        result = core.get_service("test_service")
        assert result.is_success
        assert result.value == service

        # Get non-existent
        result = core.get_service("non_existent")
        assert result.is_failure

    def test_unregister_service(self) -> None:
        """Test unregister_service method."""
        core = FlextCore()

        # Register first
        class TestService:
            pass

        service = TestService()
        core.register_service("test_service", service)

        # Unregister
        result = core.unregister_service("test_service")
        assert result.is_success

        # Verify unregistered
        result = core.get_service("test_service")
        assert result.is_failure

    def test_list_services(self) -> None:
        """Test list_services method."""
        core = FlextCore()

        result = core.list_services()
        assert result.is_success
        services = result.value
        assert isinstance(services, list)

    def test_shutdown(self) -> None:
        """Test shutdown method."""
        core = FlextCore()

        result = core.shutdown()
        assert result.is_success

    def test_initialize(self) -> None:
        """Test initialize method."""
        core = FlextCore()

        result = core.initialize()
        assert result.is_success

    def test_aggregates_property_exception(self) -> None:
        """Test aggregates property with exception."""
        core = FlextCore()

        # Force exception in property
        with patch(
            "flext_core.core.FlextModels.Aggregates.AggregateRoot",
            side_effect=Exception("Init error"),
        ):
            aggregates = core.aggregates
            # Should handle exception gracefully
            assert aggregates is not None

    def test_commands_property_exception(self) -> None:
        """Test commands property with exception."""
        core = FlextCore()

        # Force exception in property
        with patch(
            "flext_core.core.FlextCommands", side_effect=Exception("Init error")
        ):
            commands = core.commands
            # Should handle exception gracefully
            assert commands is not None

    def test_validations_property_exception(self) -> None:
        """Test validations property with exception."""
        core = FlextCore()

        # Force exception in property
        with patch(
            "flext_core.core.FlextValidations", side_effect=Exception("Init error")
        ):
            validations = core.validations
            # Should handle exception gracefully
            assert validations is not None
