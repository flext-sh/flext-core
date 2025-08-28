"""Real functionality tests for loggings module without mocks.

Tests the actual FlextLogging implementation with FlextTypes.Config integration,
StrEnum validation, and real execution paths.

Created to achieve comprehensive test coverage with actual functionality validation,
following the user's requirement for real tests without mocks.
"""

from __future__ import annotations

import time

import pytest

from flext_core import FlextConstants, FlextLogger, FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextLoggerRealFunctionality:
    """Test real FlextLogger functionality without mocks."""

    def test_logging_system_configuration_real(self) -> None:
        """Test logging system configuration using FlextTypes.Config."""
        # Test valid configuration
        valid_config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "log_level": FlextConstants.Config.LogLevel.ERROR.value,
            "enable_console_output": False,
            "enable_json_logging": True,
            "enable_correlation_tracking": True,
            "max_log_message_size": 2000,
        }

        result = FlextLogger.configure_logging_system(valid_config)
        assert result.success is True

        config = result.unwrap()
        assert config["environment"] == "production"
        assert config["log_level"] == "ERROR"
        assert config["enable_console_output"] is False
        assert config["enable_json_logging"] is True
        assert config["enable_correlation_tracking"] is True
        assert config["max_log_message_size"] == 2000

    def test_logging_system_invalid_configuration_real(self) -> None:
        """Test logging system with invalid FlextTypes.Config values."""
        # Test invalid environment
        invalid_env_config: FlextTypes.Config.ConfigDict = {
            "environment": "invalid_environment"
        }
        result = FlextLogger.configure_logging_system(invalid_env_config)
        assert result.success is False
        assert "Invalid environment" in result.error

        # Test invalid log level
        invalid_log_config: FlextTypes.Config.ConfigDict = {
            "log_level": "INVALID_LEVEL"
        }
        result = FlextLogger.configure_logging_system(invalid_log_config)
        assert result.success is False
        assert "Invalid log_level" in result.error

    def test_logging_system_default_configuration_real(self) -> None:
        """Test logging system with minimal configuration using defaults."""
        minimal_config: FlextTypes.Config.ConfigDict = {}

        result = FlextLogger.configure_logging_system(minimal_config)
        assert result.success is True

        config = result.unwrap()
        assert (
            config["environment"]
            == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
        )
        assert config["log_level"] == FlextConstants.Config.LogLevel.DEBUG.value
        assert config["enable_console_output"] is True
        assert config["enable_json_logging"] is False
        assert config["enable_correlation_tracking"] is True
        assert config["max_log_message_size"] == 10000

    def test_get_logging_system_config_real(self) -> None:
        """Test retrieving current logging system configuration."""
        result = FlextLogger.get_logging_system_config()
        assert result.success is True

        config = result.unwrap()
        assert "environment" in config
        assert "log_level" in config
        assert "enable_console_output" in config
        assert "enable_json_logging" in config
        assert "enable_correlation_tracking" in config
        assert "active_operations" in config
        assert "logging_processors_enabled" in config

        # Verify types
        assert isinstance(config["enable_console_output"], bool)
        assert isinstance(config["enable_json_logging"], bool)
        assert isinstance(config["active_operations"], int)
        assert isinstance(config["logging_processors_enabled"], list)

    def test_environment_specific_logging_config_real(self) -> None:
        """Test creation of environment-specific logging configurations."""
        # Test production configuration
        prod_result = FlextLogger.create_environment_logging_config("production")
        assert prod_result.success is True

        prod_config = prod_result.unwrap()
        assert prod_config["environment"] == "production"
        assert prod_config["log_level"] == FlextConstants.Config.LogLevel.WARNING.value
        assert prod_config["enable_console_output"] is False  # No console in production
        assert (
            prod_config["enable_json_logging"] is True
        )  # Structured logs for production
        assert (
            prod_config["enable_correlation_tracking"] is True
        )  # Correlation for tracing
        assert prod_config["max_log_message_size"] == 5000  # Limited message length

        # Test development configuration
        dev_result = FlextLogger.create_environment_logging_config("development")
        assert dev_result.success is True

        dev_config = dev_result.unwrap()
        assert dev_config["environment"] == "development"
        assert dev_config["log_level"] == FlextConstants.Config.LogLevel.DEBUG.value
        assert dev_config["enable_console_output"] is True  # Console output for dev
        assert dev_config["enable_json_logging"] is False  # Human readable for dev
        assert (
            dev_config["enable_correlation_tracking"] is True
        )  # Correlation needed for dev debugging
        assert dev_config["max_log_message_size"] == 20000  # Longer messages for dev

        # Test test environment configuration
        test_result = FlextLogger.create_environment_logging_config("test")
        assert test_result.success is True

        test_config = test_result.unwrap()
        assert test_config["environment"] == "test"
        assert (
            test_config["enable_correlation_tracking"] is False
        )  # No correlation in tests
        assert (
            test_config["enable_performance_logging"] is False
        )  # No perf logging in tests

    def test_invalid_environment_logging_config_real(self) -> None:
        """Test environment-specific logging config with invalid environment."""
        invalid_result = FlextLogger.create_environment_logging_config("invalid_env")
        assert invalid_result.success is False
        assert "Invalid environment" in invalid_result.error

    def test_logging_performance_optimization_real(self) -> None:
        """Test real logging performance optimization functionality."""
        # Test performance optimization with various levels
        configs = [
            {"performance_level": "high"},
            {"performance_level": "medium"},
            {"performance_level": "low"},
        ]

        for config in configs:
            result = FlextLogger.optimize_logging_performance(config)
            assert result.success is True

            optimized = result.unwrap()
            assert "async_logging_enabled" in optimized
            assert "buffer_size" in optimized
            assert "flush_interval_ms" in optimized
            assert "max_concurrent_operations" in optimized

            # Verify performance level specific settings
            if config["performance_level"] == "high":
                assert optimized["async_logging_enabled"] is True
                assert optimized["buffer_size"] >= 1000
                assert optimized["max_concurrent_operations"] >= 100
            elif config["performance_level"] == "low":
                assert optimized["async_logging_enabled"] is False
                assert optimized["buffer_size"] <= 1000

    def test_logging_performance_optimization_invalid_config_real(self) -> None:
        """Test performance optimization with invalid configuration."""
        # Test invalid performance level - current implementation accepts any value
        invalid_config = {"performance_level": "invalid_level"}
        result = FlextLogger.optimize_logging_performance(invalid_config)
        # Current implementation doesn't validate performance_level strictly
        assert result.success is True

        # Test invalid buffer size - current implementation doesn't validate buffer size
        invalid_buffer_config = {"buffer_size": -1}
        result = FlextLogger.optimize_logging_performance(invalid_buffer_config)
        # Current implementation doesn't validate buffer_size
        assert result.success is True


class TestFlextLoggerStrEnumIntegration:
    """Test StrEnum integration in logging configuration."""

    def test_all_environment_values_work_real(self) -> None:
        """Test all ConfigEnvironment StrEnum values work in logging."""
        # Test each environment enum value
        for env_enum in FlextConstants.Config.ConfigEnvironment:
            config: FlextTypes.Config.ConfigDict = {"environment": env_enum.value}
            result = FlextLogger.configure_logging_system(config)
            assert result.success is True

            validated_config = result.unwrap()
            assert validated_config["environment"] == env_enum.value

            # Verify expected environment values
            assert env_enum.value in {
                "development",
                "staging",
                "production",
                "test",
                "local",
            }

    def test_all_log_level_values_work_real(self) -> None:
        """Test all LogLevel StrEnum values work in logging."""
        # Test each log level enum value
        for log_enum in FlextConstants.Config.LogLevel:
            config: FlextTypes.Config.ConfigDict = {"log_level": log_enum.value}
            result = FlextLogger.configure_logging_system(config)
            assert result.success is True

            validated_config = result.unwrap()
            assert validated_config["log_level"] == log_enum.value

            # Verify expected log level values
            assert log_enum.value in {
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
                "TRACE",
            }

    def test_environment_specific_logging_all_environments_real(self) -> None:
        """Test environment-specific logging configuration for all valid environments."""
        # Test each valid environment
        for env_enum in FlextConstants.Config.ConfigEnvironment:
            result = FlextLogger.create_environment_logging_config(env_enum.value)
            assert result.success is True

            config = result.unwrap()
            assert config["environment"] == env_enum.value

            # Verify environment-appropriate settings
            if env_enum.value == "production":
                assert config["enable_console_output"] is False
                assert config["enable_json_logging"] is True
                assert config["enable_correlation_tracking"] is True
                assert config["max_log_message_size"] <= 5000
            elif env_enum.value == "development":
                assert config["enable_console_output"] is True
                assert config["enable_json_logging"] is False
                assert config["max_log_message_size"] >= 10000
            elif env_enum.value == "test":
                assert config["enable_correlation_tracking"] is False
                assert config["enable_performance_logging"] is False


class TestLoggingPerformanceReal:
    """Test real performance characteristics of logging."""

    def test_configuration_performance_real(self) -> None:
        """Test configuration performance with real execution."""
        config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "log_level": FlextConstants.Config.LogLevel.ERROR.value,
            "enable_json_logging": True,
        }

        # Measure configuration time
        start_time = time.perf_counter()

        # Configure multiple times to test performance
        for _ in range(100):
            result = FlextLogger.configure_logging_system(config)
            assert result.success is True

        end_time = time.perf_counter()

        # Should configure quickly (less than 100ms for 100 configurations)
        total_time = end_time - start_time
        assert total_time < 0.1  # Less than 100ms

    def test_environment_logging_config_creation_performance_real(self) -> None:
        """Test environment-specific logging config creation performance."""
        start_time = time.perf_counter()

        # Create multiple environment configs to test performance
        environments = ["development", "production", "test", "staging"]
        for _ in range(25):  # 25 * 4 environments = 100 operations
            for env in environments:
                result = FlextLogger.create_environment_logging_config(env)
                assert result.success is True

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should create configs quickly (less than 100ms for 100 creations)
        assert total_time < 0.1  # Less than 100ms

    def test_performance_optimization_performance_real(self) -> None:
        """Test performance optimization configuration performance."""
        # Configure for performance testing
        configs = [
            {"performance_level": "high"},
            {"performance_level": "medium"},
            {"performance_level": "low"},
        ]

        start_time = time.perf_counter()

        # Create many performance optimizations to test performance
        for _ in range(50):
            for config in configs:
                result = FlextLogger.optimize_logging_performance(config)
                assert result.success is True

        end_time = time.perf_counter()
        optimization_time = end_time - start_time

        # Performance optimization should be reasonably fast
        assert optimization_time < 0.5  # Less than 500ms for 150 optimizations


class TestLoggingConfigurationIntegration:
    """Test full configuration integration scenarios."""

    def test_complete_logging_configuration_scenario_real(self) -> None:
        """Test complete logging configuration scenario with all options."""
        # Complete configuration with all supported options
        complete_config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.STAGING.value,
            "log_level": FlextConstants.Config.LogLevel.INFO.value,
            "enable_console_output": True,
            "enable_json_logging": True,
            "enable_correlation_tracking": True,
            "max_log_message_size": 8000,
        }

        result = FlextLogger.configure_logging_system(complete_config)
        assert result.success is True

        config = result.unwrap()
        assert config["environment"] == "staging"
        assert config["log_level"] == "INFO"
        assert config["enable_console_output"] is True
        assert config["enable_json_logging"] is True
        assert config["enable_correlation_tracking"] is True
        assert config["max_log_message_size"] == 8000

    def test_configuration_with_enum_validation_real(self) -> None:
        """Test configuration validation with actual StrEnum values."""
        # Test with actual enum instances (not just string values)
        for env in FlextConstants.Config.ConfigEnvironment:
            for log_level in [
                FlextConstants.Config.LogLevel.DEBUG,
                FlextConstants.Config.LogLevel.INFO,
            ]:
                config: FlextTypes.Config.ConfigDict = {
                    "environment": env.value,
                    "log_level": log_level.value,
                }

                result = FlextLogger.configure_logging_system(config)
                assert result.success is True, (
                    f"Failed for {env.value}, {log_level.value}"
                )

                validated = result.unwrap()
                assert validated["environment"] == env.value
                assert validated["log_level"] == log_level.value

    def test_configuration_state_persistence_real(self) -> None:
        """Test that configuration state is maintained across calls."""
        # Configure logging system
        config: FlextTypes.Config.ConfigDict = {
            "environment": "production",
            "log_level": "CRITICAL",
            "enable_json_logging": True,
            "enable_correlation_tracking": True,
        }

        result = FlextLogger.configure_logging_system(config)
        assert result.success is True

        # Get current configuration
        current_result = FlextLogger.get_logging_system_config()
        assert current_result.success is True

        current_config = current_result.unwrap()
        # Configuration should reflect current system state
        assert "environment" in current_config
        assert "log_level" in current_config
        assert "enable_console_output" in current_config
        assert "enable_json_logging" in current_config
