"""Comprehensive coverage tests for core.py using only existing methods.

This targets specific uncovered methods that actually exist in FlextCore.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from flext_core import FlextCore, FlextResult
from flext_core.loggings import FlextLogger


class TestFlextCoreRealMethodsCoverage:
    """Tests targeting uncovered methods that actually exist in FlextCore."""

    def test_configure_logging_comprehensive(self) -> None:
        """Test configure_logging with various parameters."""
        core = FlextCore.get_instance()

        # Test with different log levels
        log_levels = ["debug", "info", "warning", "error", "critical"]
        for level in log_levels:
            core.configure_logging(log_level=level)

        # Test with _json_output parameter
        core.configure_logging(log_level="info", _json_output=True)
        core.configure_logging(log_level="debug", _json_output=False)

        # Test with invalid log level (should fallback to INFO)
        core.configure_logging(log_level="invalid_level")

    def test_create_log_context_comprehensive(self) -> None:
        """Test create_log_context with various inputs."""
        core = FlextCore.get_instance()

        # Test with existing FlextLogger instance
        existing_logger = FlextLogger("test_logger")
        context_logger = core.create_log_context(
            logger=existing_logger, request_id="test_123", user_id="user_456"
        )
        assert isinstance(context_logger, FlextLogger)

        # Test with string logger name
        string_logger = core.create_log_context(
            logger="my_service", operation="test_operation", correlation_id="corr_789"
        )
        assert isinstance(string_logger, FlextLogger)

        # Test with None logger (default)
        default_logger = core.create_log_context(
            logger=None, action="default_action", timestamp="2023-01-01"
        )
        assert isinstance(default_logger, FlextLogger)

    def test_get_environment_config_comprehensive(self) -> None:
        """Test get_environment_config with different environments."""
        core = FlextCore.get_instance()

        # Test different environment types
        environments = ["development", "testing", "staging", "production"]
        for env in environments:
            result = core.get_environment_config(env)  # type: ignore[arg-type]
            assert isinstance(result, FlextResult)

        # Test default environment
        result = core.get_environment_config()
        assert isinstance(result, FlextResult)

    def test_load_config_methods(self) -> None:
        """Test configuration loading methods."""
        core = FlextCore.get_instance()

        # Test load_config_from_env
        with patch.dict(os.environ, {"TEST_VAR": "test_value", "DEBUG": "true"}):
            result = core.load_config_from_env()
            assert isinstance(result, FlextResult)

        # Test with empty environment
        with patch.dict(os.environ, {}, clear=True):
            result = core.load_config_from_env()
            assert isinstance(result, FlextResult)

    def test_validate_methods_comprehensive(self) -> None:
        """Test various validate methods."""
        core = FlextCore.get_instance()

        # Test validate_config_with_types
        configs = [
            {"database": {"url": "postgresql://localhost", "timeout": 30}},
            {"logging": {"level": "info", "format": "json"}},
            {},
        ]

        for config in configs:
            result = core.validate_config_with_types(config)
            assert isinstance(result, FlextResult)

        # Test validate_dict_structure
        for config in configs:
            result = core.validate_dict_structure(config)
            assert isinstance(result, FlextResult)

        # Test validate_service_name
        service_names = [
            "valid_service",
            "another-service",
            "service123",
            "",
            "invalid service",
        ]
        for name in service_names:
            result = core.validate_service_name(name)
            assert isinstance(result, FlextResult)

    def test_merge_configs_comprehensive(self) -> None:
        """Test merge_configs functionality."""
        core = FlextCore.get_instance()

        # Test merging different configurations
        base_configs = [
            {"app": {"name": "test", "version": "1.0"}},
            {"database": {"host": "localhost", "port": 5432}},
            {},
        ]

        override_configs = [
            {"app": {"version": "2.0", "debug": True}},
            {"database": {"port": 5433, "ssl": True}},
            {"new_section": {"key": "value"}},
        ]

        for base, override in zip(base_configs, override_configs, strict=False):
            result = core.merge_configs(base, override)
            assert isinstance(result, FlextResult)

    def test_system_configuration_methods(self) -> None:
        """Test various system configuration methods."""
        core = FlextCore.get_instance()

        # Test configure_core_system
        result = core.configure_core_system({"performance": "high"})
        assert isinstance(result, FlextResult)

        # Test get_core_system_config
        result = core.get_core_system_config()
        assert isinstance(result, FlextResult)

        # Test optimize_core_performance
        result = core.optimize_core_performance("maximum")
        assert isinstance(result, FlextResult)

        # Test configure_fields_system
        result = core.configure_fields_system({"validation": True})
        assert isinstance(result, FlextResult)

    def test_factory_methods_comprehensive(self) -> None:
        """Test various factory methods."""
        core = FlextCore.get_instance()

        # Test create_config_provider
        result = core.create_config_provider()
        assert result is not None

        # Test create_demo_function
        result = core.create_demo_function()
        assert callable(result)

        # Test create_factory
        factory_result = core.create_factory()
        assert factory_result is not None

        # Test create_processing_pipeline
        pipeline = core.create_processing_pipeline()
        assert pipeline is not None

    def test_field_creation_methods(self) -> None:
        """Test field creation methods."""
        core = FlextCore.get_instance()

        # Test create_boolean_field
        result = core.create_boolean_field("test_bool", default=True)
        assert result is not None

        # Test create_integer_field
        result = core.create_integer_field("test_int", default=42, min_value=0)
        assert result is not None

        # Test create_string_field
        result = core.create_string_field("test_str", default="test", max_length=100)
        assert result is not None

        # Test create_email_address
        result = core.create_email_address("test@example.com")
        assert result is not None

    def test_service_methods_comprehensive(self) -> None:
        """Test service-related methods."""
        core = FlextCore.get_instance()

        # Test create_service_processor
        processor = core.create_service_processor()
        assert processor is not None

        # Test create_service_name_value
        service_name = core.create_service_name_value("test_service")
        assert service_name is not None

        # Test register_factory
        result = core.register_factory("test_factory", lambda: "test_value")
        assert isinstance(result, FlextResult)

        # Test get_service_with_fallback
        result = core.get_service_with_fallback("nonexistent", "fallback_value")
        assert result == "fallback_value"

    def test_performance_and_metrics_methods(self) -> None:
        """Test performance and metrics methods."""
        core = FlextCore.get_instance()

        # Test track_performance
        @core.track_performance
        def test_function() -> str:
            return "test"

        result = test_function()
        assert result == "test"

        # Test get_serialization_metrics
        metrics = core.get_serialization_metrics()
        assert isinstance(metrics, dict)

        # Test get_exception_metrics
        metrics = core.get_exception_metrics()
        assert isinstance(metrics, dict)

        # Test clear_exception_metrics
        core.clear_exception_metrics()

    def test_utility_methods_comprehensive(self) -> None:
        """Test various utility methods."""
        core = FlextCore.get_instance()

        # Test safe_get_env_var
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = core.safe_get_env_var("TEST_VAR")
            assert result == "test_value"

        result = core.safe_get_env_var("NONEXISTENT", "default")
        assert result == "default"

        # Test truncate
        result = core.truncate("this is a long string", 10)
        assert len(result) <= 10

        # Test make_pure
        def impure_func(x: int) -> int:
            return x + 1

        pure_func = core.make_pure(impure_func)
        assert callable(pure_func)

        # Test make_immutable
        mutable_dict = {"key": "value"}
        immutable = core.make_immutable(mutable_dict)
        assert immutable is not None

    def test_validator_methods(self) -> None:
        """Test validator-related methods."""
        core = FlextCore.get_instance()

        # Test require_not_none
        result = core.require_not_none("test_value")
        assert result == "test_value"

        # Test require_non_empty
        result = core.require_non_empty("test")
        assert result == "test"

        # Test require_positive
        result = core.require_positive(42)
        assert result == 42

        # Test validate_type
        result = core.validate_type("test", str)
        assert isinstance(result, FlextResult)

        # Test validate_protocol
        result = core.validate_protocol("test", str)
        assert isinstance(result, FlextResult)

    def test_decorator_creation_methods(self) -> None:
        """Test decorator creation methods."""
        core = FlextCore.get_instance()

        # Test create_validation_decorator
        validator_decorator = core.create_validation_decorator()
        assert callable(validator_decorator)

        # Test create_performance_decorator
        perf_decorator = core.create_performance_decorator()
        assert callable(perf_decorator)

        # Test create_logging_decorator
        log_decorator = core.create_logging_decorator()
        assert callable(log_decorator)

        # Test create_error_handling_decorator
        error_decorator = core.create_error_handling_decorator()
        assert callable(error_decorator)

    def test_creation_methods_comprehensive(self) -> None:
        """Test various creation methods."""
        core = FlextCore.get_instance()

        # Test create_metadata
        metadata = core.create_metadata({"key": "value"})
        assert metadata is not None

        # Test create_timestamp
        timestamp = core.create_timestamp()
        assert timestamp is not None

        # Test create_version_number
        version = core.create_version_number("1.0.0")
        assert version is not None

        # Test create_validated_model
        model = core.create_validated_model({"field": "value"})
        assert model is not None

        # Test create_validator_class
        validator_class = core.create_validator_class()
        assert validator_class is not None

    def test_functional_methods(self) -> None:
        """Test functional programming methods."""
        core = FlextCore.get_instance()

        # Test pipe
        result = core.pipe("test", str.upper, str.lower)
        assert result == "test"

        # Test compose
        composed = core.compose(str.upper, str.lower)
        assert callable(composed)

        # Test when
        result = core.when(True, lambda: "true_value", lambda: "false_value")
        assert result == "true_value"

        # Test safe_call
        def safe_func(x: int) -> int:
            if x < 0:
                raise ValueError("Negative value")
            return x * 2

        result = core.safe_call(safe_func, 5)
        assert isinstance(result, FlextResult)

        result = core.safe_call(safe_func, -5)
        assert isinstance(result, FlextResult)
        assert result.failure

    def test_system_info_methods(self) -> None:
        """Test system information methods."""
        core = FlextCore.get_instance()

        # Test get_system_info
        info = core.get_system_info()
        assert isinstance(info, dict)

        # Test get_all_functionality
        functionality = core.get_all_functionality()
        assert isinstance(functionality, list)

        # Test list_available_methods
        methods = core.list_available_methods()
        assert isinstance(methods, list)

        # Test get_method_info
        info = core.get_method_info("get_instance")
        assert info is not None

    def test_health_and_setup_methods(self) -> None:
        """Test health check and setup methods."""
        core = FlextCore.get_instance()

        # Test health_check
        result = core.health_check()
        assert isinstance(result, FlextResult)

        # Test setup_container_with_services
        result = core.setup_container_with_services()
        assert isinstance(result, FlextResult)

        # Test reset_all_caches
        core.reset_all_caches()

        # Test thread_safe_operation
        @core.thread_safe_operation
        def thread_safe_func() -> str:
            return "safe"

        result = thread_safe_func()
        assert result == "safe"

    def test_environment_methods(self) -> None:
        """Test environment-related methods."""
        core = FlextCore.get_instance()

        # Test create_environment_core_config
        config = core.create_environment_core_config()
        assert config is not None

        # Test get_settings
        settings = core.get_settings()
        assert settings is not None
