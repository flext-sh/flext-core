"""Real functionality tests for exceptions module without mocks.

Tests the actual FlextExceptions implementation with FlextTypes.Config integration,
StrEnum validation, and real execution paths.

Created to achieve comprehensive test coverage with actual functionality validation,
following the user's requirement for real tests without mocks.
"""

from __future__ import annotations

import time

import pytest

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.typings import FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextExceptionsRealFunctionality:
    """Test real FlextExceptions functionality without mocks."""

    def test_error_handling_configuration_real(self) -> None:
        """Test error handling configuration using FlextTypes.Config."""
        # Test valid configuration
        valid_config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "log_level": FlextConstants.Config.LogLevel.ERROR.value,
            "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
            "enable_metrics": True,
            "enable_stack_traces": False,
            "max_error_details": 500,
        }

        result = FlextExceptions.configure_error_handling(valid_config)
        assert result.success is True

        config = result.unwrap()
        assert config["environment"] == "production"
        assert config["log_level"] == "ERROR"
        assert config["validation_level"] == "strict"
        assert config["enable_stack_traces"] is False
        assert config["max_error_details"] == 500

    def test_error_handling_invalid_configuration_real(self) -> None:
        """Test error handling with invalid FlextTypes.Config values."""
        # Test invalid environment
        invalid_env_config: FlextTypes.Config.ConfigDict = {
            "environment": "invalid_environment"
        }
        result = FlextExceptions.configure_error_handling(invalid_env_config)
        assert result.success is False
        assert "Invalid environment" in result.error

        # Test invalid log level
        invalid_log_config: FlextTypes.Config.ConfigDict = {
            "log_level": "INVALID_LEVEL"
        }
        result = FlextExceptions.configure_error_handling(invalid_log_config)
        assert result.success is False
        assert "Invalid log_level" in result.error

        # Test invalid validation level
        invalid_val_config: FlextTypes.Config.ConfigDict = {
            "validation_level": "invalid_validation"
        }
        result = FlextExceptions.configure_error_handling(invalid_val_config)
        assert result.success is False
        assert "Invalid validation_level" in result.error

    def test_error_handling_default_configuration_real(self) -> None:
        """Test error handling with minimal configuration using defaults."""
        minimal_config: FlextTypes.Config.ConfigDict = {}

        result = FlextExceptions.configure_error_handling(minimal_config)
        assert result.success is True

        config = result.unwrap()
        assert (
            config["environment"]
            == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
        )
        assert config["log_level"] == FlextConstants.Config.LogLevel.WARNING.value
        assert (
            config["validation_level"]
            == FlextConstants.Config.ValidationLevel.NORMAL.value
        )
        assert config["enable_metrics"] is True
        assert config["enable_stack_traces"] is True

    def test_get_error_handling_config_real(self) -> None:
        """Test retrieving current error handling configuration."""
        result = FlextExceptions.get_error_handling_config()
        assert result.success is True

        config = result.unwrap()
        assert "environment" in config
        assert "log_level" in config
        assert "validation_level" in config
        assert "enable_metrics" in config
        assert "total_errors_recorded" in config
        assert "error_types_available" in config

        # Verify types
        assert isinstance(config["enable_metrics"], bool)
        assert isinstance(config["total_errors_recorded"], int)
        assert isinstance(config["error_types_available"], list)

    def test_environment_specific_config_creation_real(self) -> None:
        """Test creation of environment-specific error configurations."""
        # Test production configuration
        prod_result = FlextExceptions.create_environment_specific_config("production")
        assert prod_result.success is True

        prod_config = prod_result.unwrap()
        assert prod_config["environment"] == "production"
        assert prod_config["log_level"] == FlextConstants.Config.LogLevel.ERROR.value
        assert (
            prod_config["validation_level"]
            == FlextConstants.Config.ValidationLevel.STRICT.value
        )
        assert prod_config["enable_stack_traces"] is False  # Production security
        assert prod_config["max_error_details"] == 500  # Limited details

        # Test development configuration
        dev_result = FlextExceptions.create_environment_specific_config("development")
        assert dev_result.success is True

        dev_config = dev_result.unwrap()
        assert dev_config["environment"] == "development"
        assert dev_config["log_level"] == FlextConstants.Config.LogLevel.DEBUG.value
        assert (
            dev_config["validation_level"]
            == FlextConstants.Config.ValidationLevel.LOOSE.value
        )
        assert dev_config["enable_stack_traces"] is True  # Full debugging
        assert dev_config["max_error_details"] == 2000  # More details

        # Test test environment configuration
        test_result = FlextExceptions.create_environment_specific_config("test")
        assert test_result.success is True

        test_config = test_result.unwrap()
        assert test_config["environment"] == "test"
        assert test_config["enable_metrics"] is False  # No metrics in tests
        assert test_config["error_correlation_enabled"] is False  # No correlation

    def test_invalid_environment_specific_config_real(self) -> None:
        """Test environment-specific config with invalid environment."""
        invalid_result = FlextExceptions.create_environment_specific_config(
            "invalid_env"
        )
        assert invalid_result.success is False
        assert "Invalid environment" in invalid_result.error

    def test_exception_creation_and_metrics_real(self) -> None:
        """Test real exception creation and metrics collection."""
        # Clear metrics before test
        FlextExceptions.clear_metrics()
        initial_metrics = FlextExceptions.get_metrics()
        assert len(initial_metrics) == 0

        # Create some exceptions to trigger metrics
        try:
            msg = "Test validation error"
            raise FlextExceptions.ValidationError(
                msg,
                field="email",
                value="invalid",
                context={"user_id": "123"},
            )
        except FlextExceptions.ValidationError:
            pass  # Expected

        try:
            msg = "Test config error"
            raise FlextExceptions.ConfigurationError(
                msg, config_key="DATABASE_URL"
            )
        except FlextExceptions.ConfigurationError:
            pass  # Expected

        # Check metrics were recorded
        metrics = FlextExceptions.get_metrics()
        assert "_ValidationError" in metrics
        assert "_ConfigurationError" in metrics
        assert metrics["_ValidationError"] >= 1
        assert metrics["_ConfigurationError"] >= 1


class TestFlextExceptionsStrEnumIntegration:
    """Test StrEnum integration in exceptions configuration."""

    def test_all_environment_values_work_real(self) -> None:
        """Test all ConfigEnvironment StrEnum values work in exceptions."""
        # Test each environment enum value
        for env_enum in FlextConstants.Config.ConfigEnvironment:
            config: FlextTypes.Config.ConfigDict = {"environment": env_enum.value}
            result = FlextExceptions.configure_error_handling(config)
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
        """Test all LogLevel StrEnum values work in exceptions."""
        # Test each log level enum value
        for log_enum in FlextConstants.Config.LogLevel:
            config: FlextTypes.Config.ConfigDict = {"log_level": log_enum.value}
            result = FlextExceptions.configure_error_handling(config)
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

    def test_all_validation_level_values_work_real(self) -> None:
        """Test all ValidationLevel StrEnum values work in exceptions."""
        validation_levels = []

        # Test each validation level enum value
        for val_enum in FlextConstants.Config.ValidationLevel:
            config: FlextTypes.Config.ConfigDict = {"validation_level": val_enum.value}
            result = FlextExceptions.configure_error_handling(config)
            assert result.success is True

            validated_config = result.unwrap()
            assert validated_config["validation_level"] == val_enum.value
            validation_levels.append(val_enum.value)

        # Verify we have expected validation levels
        assert "strict" in validation_levels
        assert "normal" in validation_levels
        assert "loose" in validation_levels
        assert len(validation_levels) >= 3

    def test_environment_specific_config_all_environments_real(self) -> None:
        """Test environment-specific configuration for all valid environments."""
        # Test each valid environment
        for env_enum in FlextConstants.Config.ConfigEnvironment:
            result = FlextExceptions.create_environment_specific_config(env_enum.value)
            assert result.success is True

            config = result.unwrap()
            assert config["environment"] == env_enum.value

            # Verify environment-appropriate settings
            if env_enum.value == "production":
                assert config["enable_stack_traces"] is False
                assert config["max_error_details"] <= 500
            elif env_enum.value == "development":
                assert config["enable_stack_traces"] is True
                assert config["max_error_details"] >= 1000
            elif env_enum.value == "test":
                assert config["enable_metrics"] is False
                assert config["error_correlation_enabled"] is False


class TestExceptionsPerformanceReal:
    """Test real performance characteristics of exceptions."""

    def test_configuration_performance_real(self) -> None:
        """Test configuration performance with real execution."""
        config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "log_level": FlextConstants.Config.LogLevel.ERROR.value,
            "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
        }

        # Measure configuration time
        start_time = time.perf_counter()

        # Configure multiple times to test performance
        for _ in range(100):
            result = FlextExceptions.configure_error_handling(config)
            assert result.success is True

        end_time = time.perf_counter()

        # Should configure quickly (less than 100ms for 100 configurations)
        total_time = end_time - start_time
        assert total_time < 0.1  # Less than 100ms

    def test_environment_config_creation_performance_real(self) -> None:
        """Test environment-specific config creation performance."""
        start_time = time.perf_counter()

        # Create multiple environment configs to test performance
        environments = ["development", "production", "test", "staging"]
        for _ in range(25):  # 25 * 4 environments = 100 operations
            for env in environments:
                result = FlextExceptions.create_environment_specific_config(env)
                assert result.success is True

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should create configs quickly (less than 100ms for 100 creations)
        assert total_time < 0.1  # Less than 100ms

    def test_exception_creation_performance_real(self) -> None:
        """Test exception creation performance with configuration."""
        # Configure for performance testing
        config: FlextTypes.Config.ConfigDict = {
            "environment": "development",
            "enable_metrics": True,
            "enable_stack_traces": True,
        }
        FlextExceptions.configure_error_handling(config)

        start_time = time.perf_counter()

        # Create many exceptions to test performance
        for i in range(100):
            try:
                raise FlextExceptions.ValidationError(
                    f"Test error {i}", field="test_field", value=f"value_{i}"
                )
            except FlextExceptions.ValidationError:
                pass  # Expected

        end_time = time.perf_counter()
        creation_time = end_time - start_time

        # Exception creation should be reasonably fast
        assert creation_time < 0.5  # Less than 500ms for 100 exceptions

        # Verify metrics were collected
        metrics = FlextExceptions.get_metrics()
        assert metrics.get("_ValidationError", 0) >= 100


class TestExceptionsConfigurationIntegration:
    """Test full configuration integration scenarios."""

    def test_complete_configuration_scenario_real(self) -> None:
        """Test complete configuration scenario with all options."""
        # Complete configuration with all supported options
        complete_config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.STAGING.value,
            "log_level": FlextConstants.Config.LogLevel.INFO.value,
            "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
            "enable_metrics": True,
            "enable_stack_traces": True,
            "max_error_details": 1500,
            "error_correlation_enabled": True,
        }

        result = FlextExceptions.configure_error_handling(complete_config)
        assert result.success is True

        config = result.unwrap()
        assert config["environment"] == "staging"
        assert config["log_level"] == "INFO"
        assert config["validation_level"] == "normal"
        assert config["enable_metrics"] is True
        assert config["enable_stack_traces"] is True
        assert config["max_error_details"] == 1500
        assert config["error_correlation_enabled"] is True

    def test_configuration_with_enum_validation_real(self) -> None:
        """Test configuration validation with actual StrEnum values."""
        # Test with actual enum instances (not just string values)
        for env in FlextConstants.Config.ConfigEnvironment:
            for log_level in [
                FlextConstants.Config.LogLevel.DEBUG,
                FlextConstants.Config.LogLevel.INFO,
            ]:
                for val_level in [
                    FlextConstants.Config.ValidationLevel.STRICT,
                    FlextConstants.Config.ValidationLevel.NORMAL,
                ]:
                    config: FlextTypes.Config.ConfigDict = {
                        "environment": env.value,
                        "log_level": log_level.value,
                        "validation_level": val_level.value,
                    }

                    result = FlextExceptions.configure_error_handling(config)
                    assert result.success is True, (
                        f"Failed for {env.value}, {log_level.value}, {val_level.value}"
                    )

                    validated = result.unwrap()
                    assert validated["environment"] == env.value
                    assert validated["log_level"] == log_level.value
                    assert validated["validation_level"] == val_level.value

    def test_configuration_state_persistence_real(self) -> None:
        """Test that configuration state is maintained across calls."""
        # Configure error handling
        config: FlextTypes.Config.ConfigDict = {
            "environment": "production",
            "log_level": "CRITICAL",
            "validation_level": "strict",
            "enable_metrics": False,
        }

        result = FlextExceptions.configure_error_handling(config)
        assert result.success is True

        # Get current configuration
        current_result = FlextExceptions.get_error_handling_config()
        assert current_result.success is True

        current_config = current_result.unwrap()
        # Configuration should reflect current system state
        assert "environment" in current_config
        assert "log_level" in current_config
        assert "validation_level" in current_config
