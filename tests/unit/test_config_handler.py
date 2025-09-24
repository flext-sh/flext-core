"""Test suite for FlextConfig.HandlerConfiguration companion module.

Extracted during FlextHandlers refactoring to ensure 100% coverage
of configuration resolution logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from flext_core import FlextConfig, FlextConstants, FlextModels


class TestHandlerConfiguration:
    """Test suite for FlextConfig.HandlerConfiguration companion module."""

    def test_resolve_handler_mode_with_explicit_command_mode(self) -> None:
        """Test resolving handler mode with explicit command mode."""
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode="command", handler_config=None
        )
        assert result == "command"

    def test_resolve_handler_mode_with_explicit_query_mode(self) -> None:
        """Test resolving handler mode with explicit query mode."""
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode="query", handler_config=None
        )
        assert result == "query"

    def test_resolve_handler_mode_from_config_object_attribute(self) -> None:
        """Test resolving handler mode from config object handler_type attribute."""
        # Create a mock config object with handler_type

        @dataclass
        class MockConfig:
            handler_type: str

        config = MockConfig("command")
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=config
        )
        assert result == "command"

        config = MockConfig("query")
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=config
        )
        assert result == "query"

    def test_resolve_handler_mode_from_config_dict(self) -> None:
        """Test resolving handler mode from config dictionary."""
        config_dict = {"handler_type": "command"}
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=config_dict
        )
        assert result == "command"

        config_dict = {"handler_type": "query"}
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=config_dict
        )
        assert result == "query"

    def test_resolve_handler_mode_defaults_to_command(self) -> None:
        """Test that handler mode defaults to command when no valid mode found."""
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=None
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

        # Test with invalid mode
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode="invalid", handler_config=None
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    def test_resolve_handler_mode_invalid_config_object(self) -> None:
        """Test resolving handler mode with config object having invalid handler_type."""

        class MockConfig:
            def __init__(self) -> None:
                self.handler_type = "invalid_mode"

        config = MockConfig()
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=config
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    def test_resolve_handler_mode_config_object_without_handler_type(self) -> None:
        """Test resolving handler mode with config object without handler_type."""

        class MockConfig:
            def __init__(self) -> None:
                self.some_other_attr = "value"

        config = MockConfig()
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=config
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    def test_resolve_handler_mode_config_dict_invalid_type(self) -> None:
        """Test resolving handler mode with config dict having invalid handler_type."""
        config_dict = {"handler_type": "invalid_mode"}
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=config_dict
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    def test_resolve_handler_mode_config_dict_without_handler_type(self) -> None:
        """Test resolving handler mode with config dict without handler_type."""
        config_dict = {"some_other_key": "value"}
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=config_dict
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    def test_resolve_handler_mode_explicit_takes_precedence(self) -> None:
        """Test that explicit handler_mode takes precedence over config."""
        config_dict = {"handler_type": "query"}
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode="command", handler_config=config_dict
        )
        assert result == "command"

    def test_create_handler_config_minimal(self) -> None:
        """Test creating handler config with minimal parameters."""
        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "abcd1234" * 4  # 32 chars

            result = FlextConfig.HandlerConfiguration.create_handler_config()

            assert isinstance(result, dict)
            config_dict = result
            assert config_dict["handler_id"] == "command_handler_abcd1234"
            assert config_dict["handler_name"] == "Command Handler"
            assert config_dict["handler_type"] == "command"
            assert config_dict["handler_mode"] == "command"
            assert config_dict["command_timeout"] == 0
            assert config_dict["max_command_retries"] == 0
            assert config_dict["metadata"] == {}

    def test_create_handler_config_with_all_parameters(self) -> None:
        """Test creating handler config with all parameters specified."""
        result = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_mode="query",
            handler_name="CustomHandler",
            handler_id="custom_handler_123",
            handler_config={"extra_param": "value"},
            command_timeout=5000,
            max_command_retries=3,
        )

        assert isinstance(result, dict)
        config_dict = result
        assert config_dict["handler_id"] == "custom_handler_123"
        assert config_dict["handler_name"] == "CustomHandler"
        assert config_dict["handler_type"] == "query"
        assert config_dict["handler_mode"] == "query"
        assert config_dict["command_timeout"] == 5000
        assert config_dict["max_command_retries"] == 3
        assert config_dict["extra_param"] == "value"
        assert isinstance(config_dict["metadata"], dict)

    def test_create_handler_config_query_handler_mode(self) -> None:
        """Test creating query handler config with correct handler_mode mapping."""
        result = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_mode="query"
        )

        config_dict = result
        assert config_dict["handler_type"] == "query"
        assert config_dict["handler_mode"] == "query"

    def test_create_handler_config_command_handler_mode(self) -> None:
        """Test creating command handler config with correct handler_mode mapping."""
        result = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_mode="command"
        )

        config_dict = result
        assert config_dict["handler_type"] == "command"
        assert config_dict["handler_mode"] == "command"

    def test_create_handler_config_merges_additional_config(self) -> None:
        """Test that additional config is properly merged."""
        additional_config = {
            "custom_setting": "value",
            "another_setting": 42,
            "metadata": {"existing": "value"},
        }

        result = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_config=additional_config
        )

        config_dict = result
        assert config_dict["custom_setting"] == "value"
        assert config_dict["another_setting"] == 42
        # Additional config should override base metadata
        assert config_dict["metadata"] == {"existing": "value"}

    def test_create_handler_config_additional_config_overrides_defaults(self) -> None:
        """Test that additional config can override default values."""
        additional_config = {
            "handler_name": "OverrideHandler",
            "command_timeout": 9999,
        }

        result = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_name="OriginalHandler",
            command_timeout=1000,
            handler_config=additional_config,
        )

        config_dict = result
        # Additional config should override the provided parameters
        assert config_dict["handler_name"] == "OverrideHandler"
        assert config_dict["command_timeout"] == 9999

    def test_create_handler_config_generates_unique_ids(self) -> None:
        """Test that unique IDs are generated when not provided."""
        result1 = FlextConfig.HandlerConfiguration.create_handler_config()
        result2 = FlextConfig.HandlerConfiguration.create_handler_config()

        config1 = result1
        config2 = result2

        # Handler IDs should be different
        assert config1["handler_id"] != config2["handler_id"]

        # Both should follow the pattern
        assert str(config1["handler_id"]).startswith("command_handler_")
        assert str(config2["handler_id"]).startswith("command_handler_")

    def test_create_handler_config_none_handler_config(self) -> None:
        """Test creating handler config with None handler_config parameter."""
        result = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_mode="query", handler_config=None
        )

        config_dict = result
        assert config_dict["handler_type"] == "query"
        assert "metadata" in config_dict

    def test_integration_with_flext_models(self) -> None:
        """Test that created config can be validated by FlextModels."""
        config_data = FlextConfig.HandlerConfiguration.create_handler_config(
            handler_mode="command",
            handler_name="TestHandler",
            command_timeout=5000,
            max_command_retries=3,
        )

        # This should not raise an exception
        validated_config = FlextModels.CqrsConfig.Handler.model_validate(config_data)

        assert validated_config.handler_name == "TestHandler"
        assert validated_config.handler_type == "command"
        assert validated_config.command_timeout == 5000
        assert validated_config.max_command_retries == 3


class TestHandlerConfigurationEdgeCases:
    """Test edge cases and error conditions for HandlerConfiguration."""

    def test_resolve_handler_mode_with_none_values(self) -> None:
        """Test resolve_handler_mode handles None values gracefully."""
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=None
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    def test_resolve_handler_mode_with_empty_dict(self) -> None:
        """Test resolve_handler_mode handles empty dictionary."""
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config={}
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    def test_resolve_handler_mode_with_non_dict_non_object(self) -> None:
        """Test resolve_handler_mode with invalid config types."""
        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config="invalid"
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

        result = FlextConfig.HandlerConfiguration.resolve_handler_mode(
            handler_mode=None, handler_config=123
        )
        assert result == FlextConstants.Cqrs.DEFAULT_HANDLER_TYPE

    def test_create_handler_config_empty_strings(self) -> None:
        """Test create_handler_config handles empty string parameters."""
        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "test1234" * 4

            result = FlextConfig.HandlerConfiguration.create_handler_config(
                handler_name="", handler_id=""
            )

            config_dict = result
            # Empty strings should be replaced with defaults
            assert config_dict["handler_name"] == "Command Handler"
            assert config_dict["handler_id"] == "command_handler_test1234"

    def test_create_handler_config_none_strings(self) -> None:
        """Test create_handler_config handles None string parameters."""
        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "none1234" * 4

            result = FlextConfig.HandlerConfiguration.create_handler_config(
                handler_name=None, handler_id=None
            )

            config_dict = result
            assert config_dict["handler_name"] == "Command Handler"
            assert config_dict["handler_id"] == "command_handler_none1234"
