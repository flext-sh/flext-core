"""Targeted coverage tests for FlextConfig validators - Edge cases and error paths.

This module focuses on achieving 75% coverage by testing validator error paths
and edge cases that are not covered by integration tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import pytest

from flext_core import FlextConfig, FlextExceptions


class TestConfigValidatorErrorPaths:
    """Test error paths in FlextConfig validators."""

    def test_validate_debug_trace_consistency_valid(self) -> None:
        """Test validate_debug_trace_consistency with valid config."""
        config = FlextConfig(debug=True, trace=True)
        assert config.trace is True
        assert config.debug is True

    def test_validate_debug_trace_consistency_debug_only(self) -> None:
        """Test validate_debug_trace_consistency with debug only."""
        config = FlextConfig(debug=True, trace=False)
        assert config.debug is True
        assert config.trace is False

    def test_validate_debug_trace_consistency_invalid(self) -> None:
        """Test validate_debug_trace_consistency rejects trace without debug."""
        with pytest.raises(
            FlextExceptions.ValidationError,
            match="Trace mode requires debug mode",
        ):
            FlextConfig(debug=False, trace=True)

    def test_validate_runtime_requirements_valid(self) -> None:
        """Test validate_runtime_requirements with valid config."""
        config = FlextConfig(debug=True, trace=True)
        result = config.validate_runtime_requirements()
        assert result.is_success

    def test_validate_runtime_requirements_directly(self) -> None:
        """Test validate_runtime_requirements method directly.

        The model_validator prevents invalid states, so we test the method logic.
        """
        config = FlextConfig(debug=False, trace=False)
        result = config.validate_runtime_requirements()
        assert result.is_success

    def test_get_global_instance_singleton(self) -> None:
        """Test get_global_instance returns same instance."""
        FlextConfig.reset_global_instance()

        instance1 = FlextConfig.get_global_instance()
        instance2 = FlextConfig.get_global_instance()

        assert instance1 is instance2

    def test_set_global_instance(self) -> None:
        """Test set_global_instance replaces singleton."""
        FlextConfig.reset_global_instance()

        config1 = FlextConfig(app_name="app1")
        FlextConfig.set_global_instance(config1)

        instance = FlextConfig.get_global_instance()
        assert instance is config1
        assert instance.app_name == "app1"

    def test_reset_global_instance(self) -> None:
        """Test reset_global_instance clears singleton."""
        FlextConfig.reset_global_instance()

        instance1 = FlextConfig.get_global_instance()
        FlextConfig.reset_global_instance()
        instance2 = FlextConfig.get_global_instance()

        assert instance1 is not instance2

    def test_get_global_instance_nested_check(self) -> None:
        """Test get_global_instance nested isinstance check (coverage path)."""
        FlextConfig.reset_global_instance()

        # First call establishes instance
        instance1 = FlextConfig.get_global_instance()
        assert instance1 is not None

        # Second call should return same instance
        instance2 = FlextConfig.get_global_instance()
        assert instance2 is instance1

    def test_get_global_instance_thread_safe(self) -> None:
        """Test get_global_instance is thread-safe."""
        import threading

        FlextConfig.reset_global_instance()

        instances: list[FlextConfig] = []
        lock = threading.Lock()

        def get_instance() -> None:
            """Get instance in thread."""
            instance = FlextConfig.get_global_instance()
            with lock:
                instances.append(instance)

        threads = [threading.Thread(target=get_instance) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All instances should be the same
        assert len({id(inst) for inst in instances}) == 1


class TestConfigValidatorEdgeCases:
    """Test edge cases in config validators."""

    def test_config_with_timeout_seconds_at_boundary(self) -> None:
        """Test TimeoutSeconds at boundary values."""
        # Min boundary: >= 0.1 (per config.py constraint)
        config_min = FlextConfig(timeout_seconds=0.1)
        assert config_min.timeout_seconds >= 0.1

        # Max boundary: <= 300 (per config.py constraint)
        config_max = FlextConfig(timeout_seconds=300.0)
        assert config_max.timeout_seconds == 300.0

    def test_config_with_retry_count_at_boundary(self) -> None:
        """Test RetryCount at boundary values."""
        # Min boundary: >= 0
        config_min = FlextConfig(max_retry_attempts=0)
        assert config_min.max_retry_attempts == 0

        # Max boundary: <= 10
        config_max = FlextConfig(max_retry_attempts=10)
        assert config_max.max_retry_attempts == 10

    def test_config_timeout_seconds_coercion_from_int(self) -> None:
        """Test timeout_seconds can be coerced from int."""
        config = FlextConfig(timeout_seconds=60)
        assert isinstance(config.timeout_seconds, float)
        assert config.timeout_seconds == 60.0

    def test_config_retry_attempts_coercion_from_string(self) -> None:
        """Test retry_attempts can be coerced from string."""
        config = FlextConfig(max_retry_attempts="5")  # type: ignore[arg-type]
        assert config.max_retry_attempts == 5

    def test_config_float_field_dispatcher_timeout(self) -> None:
        """Test dispatcher_timeout_seconds coercion."""
        config = FlextConfig(dispatcher_timeout_seconds=15.5)
        assert config.dispatcher_timeout_seconds == 15.5

    def test_config_float_field_rate_limit_window(self) -> None:
        """Test rate_limit_window_seconds coercion."""
        config = FlextConfig(rate_limit_window_seconds=60)
        assert config.rate_limit_window_seconds == 60.0

    def test_config_float_field_retry_delay(self) -> None:
        """Test retry_delay coercion."""
        config = FlextConfig(retry_delay=1.5)
        assert config.retry_delay == 1.5


class TestConfigValidationMethods:
    """Test config validation methods."""

    def test_validate_business_rules_default_config(self) -> None:
        """Test validate_business_rules with default config."""
        config = FlextConfig()
        result = config.validate_business_rules()
        assert result.is_success

    def test_get_di_config_provider_creation(self) -> None:
        """Test get_di_config_provider creates provider."""
        FlextConfig.reset_global_instance()
        provider = FlextConfig.get_di_config_provider()
        assert provider is not None

    def test_get_di_config_provider_singleton(self) -> None:
        """Test get_di_config_provider returns same instance."""
        FlextConfig.reset_global_instance()
        provider1 = FlextConfig.get_di_config_provider()
        provider2 = FlextConfig.get_di_config_provider()
        assert provider1 is provider2

    def test_config_with_timeout_and_retry_coercion(self) -> None:
        """Test both timeout and retry coercion together."""
        config = FlextConfig(
            timeout_seconds=60,
            max_retry_attempts="3",  # type: ignore[arg-type]
            dispatcher_timeout_seconds=90,
        )
        assert config.timeout_seconds == 60.0
        assert config.max_retry_attempts == 3
        assert config.dispatcher_timeout_seconds == 90.0

    def test_config_float_coercion_multiple_fields(self) -> None:
        """Test float coercion across multiple fields."""
        config = FlextConfig(
            rate_limit_window_seconds=30,
            retry_delay=1.5,
        )
        assert config.rate_limit_window_seconds == 30.0
        assert config.retry_delay == 1.5

    def test_config_model_dump_preserves_types(self) -> None:
        """Test model_dump preserves field types."""
        config = FlextConfig(
            max_retry_attempts=5,
            timeout_seconds=45,
        )
        dumped = config.model_dump()
        assert dumped["max_retry_attempts"] == 5
        assert dumped["timeout_seconds"] == 45.0

    def test_config_field_validator_on_multiple_fields(self) -> None:
        """Test field validator applies to all specified fields."""
        config = FlextConfig(
            retry_delay=10,
            rate_limit_window_seconds=60,
            timeout_seconds=30,
            dispatcher_timeout_seconds=15,
        )
        assert isinstance(config.retry_delay, float)
        assert isinstance(config.rate_limit_window_seconds, float)
        assert isinstance(config.timeout_seconds, float)
        assert isinstance(config.dispatcher_timeout_seconds, float)


class TestConfigBusinessLogicValidation:
    """Test config business logic validation rules."""

    def test_validate_business_rules_returns_success(self) -> None:
        """Test validate_business_rules returns success result."""
        config = FlextConfig()
        result = config.validate_business_rules()
        assert result.is_success is True
        assert result.is_failure is False

    def test_validate_business_rules_multiple_times(self) -> None:
        """Test validate_business_rules can be called multiple times."""
        config = FlextConfig()
        result1 = config.validate_business_rules()
        result2 = config.validate_business_rules()
        assert result1.is_success is True
        assert result2.is_success is True

    def test_config_equality_with_same_values(self) -> None:
        """Test config equality with identical values."""
        config1 = FlextConfig(app_name="test", debug=True)
        config2 = FlextConfig(app_name="test", debug=True)
        # Configs should be equal if they have same values
        assert config1.app_name == config2.app_name
        assert config1.debug == config2.debug

    def test_config_dict_representation(self) -> None:
        """Test config dict representation and round-trip."""
        original_config = FlextConfig(app_name="myapp", debug=False)
        config_dict = original_config.model_dump()
        assert config_dict["app_name"] == "myapp"
        assert config_dict["debug"] is False
        # Should be able to recreate from dict
        new_config = FlextConfig(**config_dict)
        assert new_config.app_name == original_config.app_name
        assert new_config.debug == original_config.debug


__all__ = [
    "TestConfigBusinessLogicValidation",
    "TestConfigValidationMethods",
    "TestConfigValidatorEdgeCases",
    "TestConfigValidatorErrorPaths",
]
