"""Tests for backward compatibility of Commands migration to Pydantic configs.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

import warnings
from typing import Literal

import pytest

from flext_core import FlextCommands, FlextModels


class TestCommandsCompatibility:
    """Test backward compatibility during Commands migration to Pydantic."""

    def test_configure_commands_system_with_config_model(self) -> None:
        """Test new API with CommandsConfig model."""
        config = FlextModels.SystemConfigs.CommandsConfig(
            environment="development",
            log_level="DEBUG",
            enable_handler_discovery=True,
            max_concurrent_commands=50,
            command_timeout_seconds=120,
        )

        result = FlextCommands.configure_commands_system(config)

        assert result.success
        assert isinstance(result.value, FlextModels.SystemConfigs.CommandsConfig)
        assert result.value.environment == "development"
        assert result.value.log_level == "DEBUG"
        assert result.value.max_concurrent_commands == 50
        assert result.value.command_timeout_seconds == 120

    def test_configure_commands_system_with_dict_legacy(self) -> None:
        """Test legacy API with dict config (should show deprecation warning)."""
        dict_config = {
            "environment": "production",
            "log_level": "INFO",
            "enable_handler_discovery": False,
            "max_concurrent_commands": 75,
            "command_timeout_seconds": 60,
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextCommands.configure_commands_system(dict_config)

            # Verify deprecation warning was shown
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Using dict for configure_commands_system is deprecated" in str(
                w[0].message
            )

        # Verify functionality still works
        assert result.success
        assert isinstance(result.value, dict)
        assert result.value["environment"] == "production"
        assert result.value["log_level"] == "INFO"
        assert not result.value["enable_handler_discovery"]
        assert result.value["max_concurrent_commands"] == 75
        assert result.value["command_timeout_seconds"] == 60

    def test_get_commands_system_config_default_behavior(self) -> None:
        """Test get_commands_system_config default behavior (returns dict for compatibility)."""
        result = FlextCommands.get_commands_system_config()

        assert result.success
        # Default returns dict for backward compatibility
        assert isinstance(result.value, dict)

    def test_get_commands_system_config_with_model_true(self) -> None:
        """Test get_commands_system_config with explicit return_model=True."""
        result = FlextCommands.get_commands_system_config(return_model=True)

        assert result.success
        assert isinstance(result.value, FlextModels.SystemConfigs.CommandsConfig)

    def test_get_commands_system_config_with_model_false_legacy(self) -> None:
        """Test get_commands_system_config with return_model=False (legacy dict)."""
        result = FlextCommands.get_commands_system_config(return_model=False)

        assert result.success
        assert isinstance(result.value, dict)
        assert "environment" in result.value
        assert "log_level" in result.value

    def test_bidirectional_compatibility(self) -> None:
        """Test that configs work bidirectionally between dict and model."""
        # Start with dict config
        dict_config = {
            "environment": "staging",
            "log_level": "WARNING",
            "enable_handler_discovery": True,
            "max_concurrent_commands": 25,
            "command_timeout_seconds": 30,
        }

        # Configure with dict (legacy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress deprecation for test
            config_result = FlextCommands.configure_commands_system(dict_config)

        assert config_result.success
        assert isinstance(config_result.value, dict)

        # Now configure with model
        model_config = FlextModels.SystemConfigs.CommandsConfig(
            environment="test",
            log_level="ERROR",
            enable_handler_discovery=False,
            max_concurrent_commands=10,
            command_timeout_seconds=180,
        )

        model_result = FlextCommands.configure_commands_system(model_config)

        assert model_result.success
        assert isinstance(model_result.value, FlextModels.SystemConfigs.CommandsConfig)
        assert model_result.value.environment == "test"

    def test_config_validation_works_for_both_approaches(self) -> None:
        """Test that validation works for both dict and model approaches."""
        # Invalid dict config should fail
        invalid_dict = {
            "environment": "invalid_env",  # Invalid value
            "log_level": "DEBUG",
            "enable_handler_discovery": True,
            "max_concurrent_commands": 50,
            "command_timeout_seconds": 60,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = FlextCommands.configure_commands_system(invalid_dict)

        assert not result.success
        assert "environment" in result.error

        # Invalid model config should also fail
        with pytest.raises(ValueError):
            FlextModels.SystemConfigs.CommandsConfig(
                environment="invalid_env",
                log_level="DEBUG",
                enable_handler_discovery=True,
                max_concurrent_commands=50,
                command_timeout_seconds=60,
            )

    def test_type_safety_with_overloads(self) -> None:
        """Test that type hints work correctly with overloads."""
        # This test mainly verifies type checking works correctly
        # Model input -> Model output
        config = FlextModels.SystemConfigs.CommandsConfig(
            environment="development", log_level="DEBUG"
        )
        result = FlextCommands.configure_commands_system(config)
        assert isinstance(result.value, FlextModels.SystemConfigs.CommandsConfig)

        # Dict input -> Dict output (with warning suppressed)
        dict_config = {"environment": "production", "log_level": "INFO"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = FlextCommands.configure_commands_system(dict_config)
        assert isinstance(result.value, dict)

    def test_create_environment_commands_config_with_model(self) -> None:
        """Test create_environment_commands_config with return_model=True."""
        result = FlextCommands.create_environment_commands_config(
            "production", return_model=True
        )

        assert result.success
        assert isinstance(result.value, FlextModels.SystemConfigs.CommandsConfig)
        assert result.value.environment == "production"
        assert result.value.log_level == "WARNING"  # Production default

    def test_create_environment_commands_config_with_dict(self) -> None:
        """Test create_environment_commands_config with return_model=False (legacy)."""
        result = FlextCommands.create_environment_commands_config(
            "development", return_model=False
        )

        assert result.success
        assert isinstance(result.value, dict)
        assert result.value["environment"] == "development"
        assert result.value["log_level"] == "DEBUG"  # Development default

    def test_create_environment_commands_config_default(self) -> None:
        """Test create_environment_commands_config default behavior (returns dict)."""
        result = FlextCommands.create_environment_commands_config("test")

        assert result.success
        # Default returns dict for backward compatibility
        assert isinstance(result.value, dict)
        assert result.value["environment"] == "test"
        assert result.value["log_level"] == "ERROR"  # Test default

    def test_optimize_commands_performance_with_model(self) -> None:
        """Test optimize_commands_performance with CommandsConfig model."""
        config = FlextModels.SystemConfigs.CommandsConfig(
            environment="production", log_level="INFO", max_concurrent_commands=50
        )

        result = FlextCommands.optimize_commands_performance(
            config, performance_level="high"
        )

        assert result.success
        assert isinstance(result.value, FlextModels.SystemConfigs.CommandsConfig)
        # Should have performance optimizations applied
        assert result.value.environment == "production"

    def test_optimize_commands_performance_with_dict_legacy(self) -> None:
        """Test optimize_commands_performance with dict (should show deprecation warning)."""
        dict_config = {
            "environment": "development",
            "log_level": "DEBUG",
            "max_concurrent_commands": 100,
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = FlextCommands.optimize_commands_performance(
                dict_config, performance_level="low"
            )

            # Verify deprecation warning was shown
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Using dict for optimize_commands_performance is deprecated" in str(
                w[0].message
            )

        # Verify functionality still works
        assert result.success
        assert isinstance(result.value, dict)
        assert result.value["environment"] == "development"

    def test_optimize_commands_performance_levels(self) -> None:
        """Test optimize_commands_performance with different performance levels."""
        config = FlextModels.SystemConfigs.CommandsConfig(
            environment="staging", log_level="INFO"
        )

        # Test all performance levels
        levels: list[Literal["low", "medium", "high", "extreme"]] = [
            "low",
            "medium",
            "high",
            "extreme",
        ]
        for level in levels:
            result = FlextCommands.optimize_commands_performance(
                config, performance_level=level
            )
            assert result.success
            assert isinstance(result.value, FlextModels.SystemConfigs.CommandsConfig)

    def test_all_methods_support_dual_signatures(self) -> None:
        """Test that all migrated methods support both dict and model signatures."""
        # Test configure_commands_system
        model_config = FlextModels.SystemConfigs.CommandsConfig(
            environment="development", log_level="DEBUG"
        )
        dict_config = {"environment": "production", "log_level": "INFO"}

        result1 = FlextCommands.configure_commands_system(model_config)
        assert result1.success
        assert isinstance(result1.value, FlextModels.SystemConfigs.CommandsConfig)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result2 = FlextCommands.configure_commands_system(dict_config)
        assert result2.success
        assert isinstance(result2.value, dict)

        # Test get_commands_system_config
        result3 = FlextCommands.get_commands_system_config(return_model=True)
        assert result3.success
        assert isinstance(result3.value, FlextModels.SystemConfigs.CommandsConfig)

        result4 = FlextCommands.get_commands_system_config(return_model=False)
        assert result4.success
        assert isinstance(result4.value, dict)

        # Test create_environment_commands_config
        result5 = FlextCommands.create_environment_commands_config(
            "test", return_model=True
        )
        assert result5.success
        assert isinstance(result5.value, FlextModels.SystemConfigs.CommandsConfig)

        result6 = FlextCommands.create_environment_commands_config(
            "test", return_model=False
        )
        assert result6.success
        assert isinstance(result6.value, dict)

        # Test optimize_commands_performance
        result7 = FlextCommands.optimize_commands_performance(model_config, "high")
        assert result7.success
        assert isinstance(result7.value, FlextModels.SystemConfigs.CommandsConfig)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result8 = FlextCommands.optimize_commands_performance(dict_config, "low")
        assert result8.success
        assert isinstance(result8.value, dict)
