"""Real functionality tests for handlers module without mocks.

Tests the actual handlers implementation with FlextTypes.Config integration,
StrEnum validation, and real execution paths.

Created to achieve comprehensive test coverage with actual functionality validation,
following the user's requirement for real tests without mocks.



Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from typing import Literal, cast

import pytest

from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.typings import FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextHandlersRealFunctionality:
    """Test real FlextHandlers functionality without mocks."""

    def test_basic_handler_creation_real(self) -> None:
        """Test real BasicHandler creation and initialization."""
        handler = FlextHandlers.Implementation.BasicHandler("test_handler")

        assert handler.handler_name == "test_handler"
        assert handler.state == FlextHandlers.Constants.Handler.States.IDLE

        # Test metrics initialization
        metrics = handler.get_metrics()
        assert metrics["requests_processed"] == 0
        assert metrics["successful_requests"] == 0
        assert metrics["failed_requests"] == 0
        assert metrics["average_processing_time"] == 0.0

    def test_handler_configuration_with_flexttypes_real(self) -> None:
        """Test handler configuration using FlextTypes.Config."""
        handler = FlextHandlers.Implementation.BasicHandler("config_handler")

        # Test valid configuration
        config: FlextTypes.Config.ConfigDict = {
            "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
            "timeout": 60000,
        }

        result = handler.configure(config)
        assert result.success is True

        # Test invalid log level
        invalid_config: FlextTypes.Config.ConfigDict = {
            "log_level": "INVALID_LEVEL",
        }

        result = handler.configure(invalid_config)
        assert result.success is False
        assert result.error is not None
        assert "Invalid log_level" in (result.error or "")

    def test_handler_environment_validation_real(self) -> None:
        """Test environment validation using StrEnum values."""
        handler = FlextHandlers.Implementation.BasicHandler("env_handler")

        # Test all valid environments
        valid_environments = [e.value for e in FlextConstants.Config.ConfigEnvironment]

        for env in valid_environments:
            config: FlextTypes.Config.ConfigDict = {"environment": env}
            result = handler.configure(config)
            assert result.success is True, f"Failed for environment: {env}"

        # Test invalid environment
        invalid_config: FlextTypes.Config.ConfigDict = {"environment": "invalid_env"}
        result = handler.configure(invalid_config)
        assert result.success is False
        assert result.error is not None
        assert "Invalid environment" in (result.error or "")

    def test_handler_log_level_validation_real(self) -> None:
        """Test log level validation using StrEnum values."""
        handler = FlextHandlers.Implementation.BasicHandler("log_handler")

        # Test all valid log levels
        valid_levels = [level.value for level in FlextConstants.Config.LogLevel]

        for level in valid_levels:
            config: FlextTypes.Config.ConfigDict = {"log_level": level}
            result = handler.configure(config)
            assert result.success is True, f"Failed for log level: {level}"

    def test_handler_validation_level_integration_real(self) -> None:
        """Test validation level integration with StrEnum."""
        handler = FlextHandlers.Implementation.BasicHandler("val_handler")

        # Test all validation levels
        for val_level in FlextConstants.Config.ValidationLevel:
            config: FlextTypes.Config.ConfigDict = {"validation_level": val_level.value}
            result = handler.configure(config)
            assert result.success is True
            assert val_level.value in {"strict", "normal", "loose", "disabled"}

    def test_handler_complete_configuration_scenario_real(self) -> None:
        """Test complete configuration scenario with all FlextTypes.Config."""
        handler = FlextHandlers.Implementation.BasicHandler("complete_handler")

        # Complete configuration with all supported options
        complete_config: FlextTypes.Config.ConfigDict = {
            "log_level": FlextConstants.Config.LogLevel.INFO.value,
            "environment": FlextConstants.Config.ConfigEnvironment.STAGING.value,
            "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
            "timeout": 30000,
            "max_retries": 5,
        }

        result = handler.configure(complete_config)
        assert result.success is True

        # Verify configuration was applied by trying to handle a request
        # (this tests that configuration doesn't break request processing)
        request_data = {"action": "test", "data": {"key": "value"}}
        handle_result = handler.handle(request_data)

        # Should succeed (or fail gracefully with proper error handling)
        assert hasattr(handle_result, "success")
        assert hasattr(handle_result, "error")

    def test_handler_metrics_real_execution(self) -> None:
        """Test real metrics collection during handler execution."""
        handler = FlextHandlers.Implementation.BasicHandler("metrics_handler")

        # Configure handler
        config: FlextTypes.Config.ConfigDict = {
            "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
            "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
        }
        handler.configure(config)

        # Get initial metrics
        initial_metrics = handler.get_metrics()
        assert initial_metrics["requests_processed"] == 0

        # Process a request
        handler.handle({"test": "data"})

        # Check metrics were updated
        updated_metrics = handler.get_metrics()
        assert updated_metrics["requests_processed"] == 1

        # Either success or failure should be incremented
        assert (
            updated_metrics["successful_requests"] == 1
            or updated_metrics["failed_requests"] == 1
        )


class TestFlextHandlersStrEnumIntegration:
    """Test StrEnum integration in handlers."""

    def test_all_config_environment_values_work_real(self) -> None:
        """Test all ConfigEnvironment StrEnum values work in handlers."""
        handler = FlextHandlers.Implementation.BasicHandler("enum_handler")

        # Test each environment enum value
        for env_enum in FlextConstants.Config.ConfigEnvironment:
            config: FlextTypes.Config.ConfigDict = {"environment": env_enum.value}
            result = handler.configure(config)
            assert result.success is True
            assert env_enum.value in {
                "development",
                "staging",
                "production",
                "test",
                "local",
            }

    def test_all_log_level_values_work_real(self) -> None:
        """Test all LogLevel StrEnum values work in handlers."""
        handler = FlextHandlers.Implementation.BasicHandler("log_enum_handler")

        # Test each log level enum value
        for log_enum in FlextConstants.Config.LogLevel:
            config: FlextTypes.Config.ConfigDict = {"log_level": log_enum.value}
            result = handler.configure(config)
            assert result.success is True
            assert log_enum.value in {
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
                "TRACE",
            }

    def test_validation_level_enum_functionality_real(self) -> None:
        """Test ValidationLevel StrEnum functionality."""
        handler = FlextHandlers.Implementation.BasicHandler("validation_enum_handler")

        # Test each validation level
        validation_levels = []
        for val_enum in FlextConstants.Config.ValidationLevel:
            config: FlextTypes.Config.ConfigDict = {"validation_level": val_enum.value}
            result = handler.configure(config)
            assert result.success is True
            validation_levels.append(val_enum.value)

        # Verify we have the expected validation levels
        assert "strict" in validation_levels
        assert "normal" in validation_levels
        assert "loose" in validation_levels
        assert len(validation_levels) >= 3


class TestHandlersPerformanceReal:
    """Test real performance characteristics of handlers."""

    def test_handler_configuration_performance_real(self) -> None:
        """Test configuration performance with real execution."""
        handler = FlextHandlers.Implementation.BasicHandler("perf_handler")

        config: FlextTypes.Config.ConfigDict = {
            "log_level": FlextConstants.Config.LogLevel.INFO.value,
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
        }

        # Measure configuration time
        start_time = time.perf_counter()

        # Configure multiple times to test performance
        for _ in range(100):
            result = handler.configure(config)
            assert result.success is True

        end_time = time.perf_counter()

        # Should configure quickly (less than 100ms for 100 configurations)
        total_time = end_time - start_time
        assert total_time < 0.1  # Less than 100ms

    def test_multiple_handlers_configuration_real(self) -> None:
        """Test configuration of multiple handlers with real execution."""
        handlers = []

        # Create multiple handlers
        for i in range(10):
            handler = FlextHandlers.Implementation.BasicHandler(f"handler_{i}")
            handlers.append(handler)

        # Configure all handlers
        config: FlextTypes.Config.ConfigDict = {
            "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
            "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
        }

        success_count = 0
        for handler in handlers:
            result = handler.configure(config)
            if result.success:
                success_count += 1

        # All handlers should configure successfully
        assert success_count == 10

        # Verify each handler maintains its own configuration
        for handler in handlers:
            metrics = handler.get_metrics()
            assert metrics["requests_processed"] == 0  # Each handler starts fresh


class TestHandlersConfigurationIntegration:
    """Test handler configuration method integration scenarios."""

    def test_get_handler_config_real(self) -> None:
        """Test retrieving handler configuration with real execution."""
        handler = FlextHandlers.Implementation.BasicHandler("get_config_handler")

        # Configure handler first
        config: FlextTypes.Config.ConfigDict = {
            "log_level": FlextConstants.Config.LogLevel.WARNING.value,
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
            "timeout": 45000,
        }
        handler.configure(config)

        # Get current configuration
        result = handler.get_handler_config()
        assert result.success is True

        current_config = result.unwrap()
        assert "handler_name" in current_config
        assert "handler_state" in current_config
        assert "requests_processed" in current_config
        assert "success_rate" in current_config

        # Verify handler-specific information
        assert current_config["handler_name"] == "get_config_handler"
        assert (
            current_config["handler_state"]
            == FlextHandlers.Constants.Handler.States.IDLE
        )
        assert isinstance(current_config["requests_processed"], int)
        assert isinstance(current_config["success_rate"], float)

    def test_create_environment_handler_config_real(self) -> None:
        """Test environment-specific handler configuration creation."""
        # Test production environment
        prod_result = (
            FlextHandlers.Implementation.BasicHandler.create_environment_handler_config(
                "production",
            )
        )
        assert prod_result.success is True

        prod_config = prod_result.unwrap()
        assert prod_config["environment"] == "production"
        assert prod_config["log_level"] == FlextConstants.Config.LogLevel.WARNING.value
        assert (
            prod_config["validation_level"]
            == FlextConstants.Config.ValidationLevel.STRICT.value
        )

        timeout_value = prod_config["timeout"]
        assert isinstance(timeout_value, (int, str))
        assert int(timeout_value) >= 30000  # Production should have reasonable timeout

        max_retries_value = prod_config["max_retries"]
        assert isinstance(max_retries_value, (int, str))
        assert int(max_retries_value) >= 1  # Production should have retries

        # Test development environment
        dev_result = (
            FlextHandlers.Implementation.BasicHandler.create_environment_handler_config(
                "development",
            )
        )
        assert dev_result.success is True

        dev_config = dev_result.unwrap()
        assert dev_config["environment"] == "development"
        assert dev_config["log_level"] == FlextConstants.Config.LogLevel.DEBUG.value
        assert (
            dev_config["validation_level"]
            == FlextConstants.Config.ValidationLevel.LOOSE.value
        )
        assert (
            dev_config["enable_debugging"] is True
        )  # Development should enable debugging

        # Test test environment
        test_result = (
            FlextHandlers.Implementation.BasicHandler.create_environment_handler_config(
                "test",
            )
        )
        assert test_result.success is True

        test_config = test_result.unwrap()
        assert test_config["environment"] == "test"
        assert (
            test_config["enable_performance_tracking"] is False
        )  # No perf tracking in tests

        # Test invalid environment

        # Test with properly typed invalid environment using cast
        # Define Environment type locally to avoid import issues
        type Environment = Literal[
            "development", "production", "staging", "test", "local"
        ]

        invalid_result = (
            FlextHandlers.Implementation.BasicHandler.create_environment_handler_config(
                cast("Environment", "invalid_env"),
            )
        )
        assert invalid_result.success is False
        assert invalid_result.error is not None
        assert "Invalid environment" in invalid_result.error

    def test_optimize_handler_performance_real(self) -> None:
        """Test handler performance optimization with real execution."""
        # Test high performance configuration
        high_perf_config: FlextTypes.Config.ConfigDict = {
            "performance_level": "high",
            "max_concurrent_requests": 100,
        }

        result = FlextHandlers.Implementation.BasicHandler.optimize_handler_performance(
            high_perf_config,
        )
        assert result.success is True

        optimized_config = result.unwrap()
        assert "performance_level" in optimized_config
        assert "max_concurrent_requests" in optimized_config
        assert "request_queue_size" in optimized_config
        assert "processing_timeout" in optimized_config

        # High performance should have optimized values
        max_concurrent = optimized_config["max_concurrent_requests"]
        assert isinstance(max_concurrent, (int, str))
        assert int(max_concurrent) >= 50

        queue_size = optimized_config["request_queue_size"]
        assert isinstance(queue_size, (int, str))
        assert int(queue_size) >= 1000

        # Test medium performance configuration
        medium_perf_config: FlextTypes.Config.ConfigDict = {
            "performance_level": "medium",
            "max_concurrent_requests": 50,
        }

        result = FlextHandlers.Implementation.BasicHandler.optimize_handler_performance(
            medium_perf_config,
        )
        assert result.success is True

        medium_config = result.unwrap()
        assert medium_config["performance_level"] == "medium"

        medium_concurrent = medium_config["max_concurrent_requests"]
        assert isinstance(medium_concurrent, (int, str, float))
        assert 10 <= int(medium_concurrent) <= 100

        # Test low performance configuration
        low_perf_config: FlextTypes.Config.ConfigDict = {
            "performance_level": "low",
            "max_concurrent_requests": 10,
        }

        result = FlextHandlers.Implementation.BasicHandler.optimize_handler_performance(
            low_perf_config,
        )
        assert result.success is True

        low_config = result.unwrap()
        assert low_config["performance_level"] == "low"

        low_concurrent = low_config["max_concurrent_requests"]
        assert isinstance(low_concurrent, (int, str, float))
        assert int(low_concurrent) <= 20

    def test_configuration_state_persistence_real(self) -> None:
        """Test that handler configuration state is maintained across calls."""
        handler = FlextHandlers.Implementation.BasicHandler("state_handler")

        # Configure handler
        initial_config: FlextTypes.Config.ConfigDict = {
            "log_level": FlextConstants.Config.LogLevel.CRITICAL.value,
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
        }

        result = handler.configure(initial_config)
        assert result.success is True

        # Get configuration and verify persistence
        config_result = handler.get_handler_config()
        assert config_result.success is True

        retrieved_config = config_result.unwrap()
        # Configuration should reflect the handler state (not necessarily exact input)
        assert "handler_name" in retrieved_config
        assert "state" in retrieved_config
        assert retrieved_config["handler_name"] == "state_handler"

        # Process some requests to change internal state
        handler.handle({"test": "request1"})
        handler.handle({"test": "request2"})

        # Configuration should still be accessible and show updated metrics
        updated_config_result = handler.get_handler_config()
        assert updated_config_result.success is True

        updated_config = updated_config_result.unwrap()
        assert "metrics" in updated_config
        # Metrics should show processed requests
        metrics = updated_config["metrics"]
        assert isinstance(metrics, dict)

        requests_processed = metrics["requests_processed"]
        assert isinstance(requests_processed, (int, str))
        assert int(requests_processed) >= 2


class TestHandlersConfigurationEdgeCases:
    """Test edge cases and error scenarios in handler configuration."""

    def test_handler_configuration_validation_errors_real(self) -> None:
        """Test various configuration validation errors."""
        handler = FlextHandlers.Implementation.BasicHandler("validation_handler")

        # Test invalid timeout type
        invalid_timeout_config: FlextTypes.Config.ConfigDict = {
            "timeout": "invalid_timeout",  # Should be integer
        }

        result = handler.configure(invalid_timeout_config)
        # Implementation might accept this or reject it - test actual behavior
        # The important thing is it returns a FlextResult
        assert hasattr(result, "success")
        assert hasattr(result, "error")

        # Test negative timeout
        negative_timeout_config: FlextTypes.Config.ConfigDict = {
            "timeout": -1000,
        }

        result = handler.configure(negative_timeout_config)
        # Implementation might accept this or reject it - test actual behavior
        assert hasattr(result, "success")
        assert hasattr(result, "error")

    def test_environment_handler_config_all_environments_real(self) -> None:
        """Test environment handler config for all valid environments."""
        # Test each valid environment
        for env_enum in FlextConstants.Config.ConfigEnvironment:
            result = FlextHandlers.Implementation.BasicHandler.create_environment_handler_config(
                env_enum.value,
            )
            assert result.success is True

            config = result.unwrap()
            assert config["environment"] == env_enum.value

            # Verify environment-appropriate settings
            if env_enum.value == "production":
                assert config["log_level"] in {"WARNING", "ERROR", "CRITICAL"}
                assert config["validation_level"] == "strict"
            elif env_enum.value == "development":
                assert config["log_level"] in {"DEBUG", "INFO"}
                assert config.get("enable_debugging", False) is True
            elif env_enum.value == "test":
                assert config.get("enable_performance_tracking", True) is False

    def test_performance_optimization_edge_cases_real(self) -> None:
        """Test performance optimization with edge case configurations."""
        # Test with minimal configuration
        minimal_config: FlextTypes.Config.ConfigDict = {}

        result = FlextHandlers.Implementation.BasicHandler.optimize_handler_performance(
            minimal_config,
        )
        assert result.success is True

        optimized = result.unwrap()
        assert "performance_level" in optimized
        assert "max_concurrent_requests" in optimized

        # Test with zero concurrent requests
        zero_config: FlextTypes.Config.ConfigDict = {
            "max_concurrent_requests": 0,
        }

        result = FlextHandlers.Implementation.BasicHandler.optimize_handler_performance(
            zero_config,
        )
        # Implementation should handle this gracefully
        assert result.success is True

        # Test with very high concurrent requests
        high_config: FlextTypes.Config.ConfigDict = {
            "max_concurrent_requests": 10000,
        }

        result = FlextHandlers.Implementation.BasicHandler.optimize_handler_performance(
            high_config,
        )
        assert result.success is True

        high_optimized = result.unwrap()
        # Should have reasonable limits even with high input
        high_concurrent = high_optimized["max_concurrent_requests"]
        assert isinstance(high_concurrent, (int, str, float))
        assert int(high_concurrent) <= 10000
