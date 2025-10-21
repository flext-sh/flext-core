"""Final coverage push - targeted tests for specific uncovered lines.

This file targets SPECIFIC uncovered lines to reach 75% threshold.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import os

import pytest

from flext_core import FlextConfig, FlextExceptions


class TestCoverageTargetedUncovered:
    """Tests targeting specific uncovered line ranges in config.py."""

    def test_config_env_variable_trace_debug(self) -> None:
        """Test trace mode requires debug mode validation."""
        # This should raise because trace=True but debug=False
        with pytest.raises(FlextExceptions.ValidationError):
            FlextConfig(debug=False, trace=True)

    def test_config_debug_only(self) -> None:
        """Test debug mode without trace."""
        config = FlextConfig(debug=True, trace=False)
        assert config.debug is True
        assert config.trace is False

    def test_config_trace_with_debug(self) -> None:
        """Test trace mode with debug mode enabled."""
        config = FlextConfig(debug=True, trace=True)
        assert config.debug is True
        assert config.trace is True

    def test_validate_runtime_requirements_success_path(self) -> None:
        """Test validate_runtime_requirements succeeds with valid config."""
        config = FlextConfig(debug=True, trace=True)
        result = config.validate_runtime_requirements()
        assert result.is_success
        assert result.value is None

    def test_config_di_provider_initialization(self) -> None:
        """Test DI provider initialization and singleton behavior."""
        FlextConfig.reset_global_instance()
        # First call initializes
        provider1 = FlextConfig.get_di_config_provider()
        assert provider1 is not None
        # Second call returns same
        provider2 = FlextConfig.get_di_config_provider()
        assert provider1 is provider2

    def test_config_di_provider_with_instance(self) -> None:
        """Test DI provider uses global instance configuration."""
        FlextConfig.reset_global_instance()
        config = FlextConfig(app_name="di_test", debug=True)
        provider = FlextConfig.get_di_config_provider()
        assert provider is not None

    def test_config_timeout_validation_all_fields(self) -> None:
        """Test timeout fields all use float coercion."""
        config = FlextConfig(
            retry_delay=10,  # type: ignore
            rate_limit_window_seconds=20,  # type: ignore
            timeout_seconds=30,  # type: ignore
            dispatcher_timeout_seconds=40,  # type: ignore
        )
        assert isinstance(config.retry_delay, float)
        assert isinstance(config.rate_limit_window_seconds, float)
        assert isinstance(config.timeout_seconds, float)
        assert isinstance(config.dispatcher_timeout_seconds, float)

    def test_config_retry_attempts_int_coercion(self) -> None:
        """Test retry attempts uses int coercion."""
        config = FlextConfig(
            max_retry_attempts=5,
        )
        assert isinstance(config.max_retry_attempts, int)
        assert config.max_retry_attempts == 5

    def test_config_field_validator_order(self) -> None:
        """Test field validators execute in correct order."""
        # This tests the validator that processes multiple fields
        config = FlextConfig(
            retry_delay=1.5,
            rate_limit_window_seconds=2.5,
            timeout_seconds=3.5,
            dispatcher_timeout_seconds=4.5,
        )
        assert config.retry_delay == 1.5
        assert config.rate_limit_window_seconds == 2.5
        assert config.timeout_seconds == 3.5
        assert config.dispatcher_timeout_seconds == 4.5

    def test_config_with_environment_variables(self) -> None:
        """Test config respects environment variables."""
        saved_env = os.environ.copy()
        try:
            # Set environment variables
            os.environ["FLEXT_DEBUG"] = "true"
            os.environ["FLEXT_LOG_LEVEL"] = "WARNING"

            config = FlextConfig()
            assert config.debug is True
            assert config.log_level == "WARNING"
        finally:
            # Restore environment
            for key in ["FLEXT_DEBUG", "FLEXT_LOG_LEVEL"]:
                if key in saved_env:
                    os.environ[key] = saved_env[key]
                else:
                    os.environ.pop(key, None)

    def test_config_instance_type_checking(self) -> None:
        """Test config instance type validation."""
        FlextConfig.reset_global_instance()

        instance1 = FlextConfig.get_global_instance()
        assert isinstance(instance1, FlextConfig)

        instance2 = FlextConfig.get_global_instance()
        assert isinstance(instance2, FlextConfig)
        assert instance1 is instance2

    def test_config_set_then_get_global(self) -> None:
        """Test setting and getting global instance."""
        FlextConfig.reset_global_instance()

        custom_config = FlextConfig(app_name="custom")
        FlextConfig.set_global_instance(custom_config)

        retrieved = FlextConfig.get_global_instance()
        assert retrieved is custom_config
        assert retrieved.app_name == "custom"

    def test_config_business_rules_with_multiple_checks(self) -> None:
        """Test validate_business_rules multiple times."""
        config = FlextConfig()

        result1 = config.validate_business_rules()
        assert result1.is_success

        result2 = config.validate_business_rules()
        assert result2.is_success

        # Results should be independent
        assert result1 is not result2


__all__ = ["TestCoverageTargetedUncovered"]
