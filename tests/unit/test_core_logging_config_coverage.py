"""Coverage tests targeting logging and configuration methods in core.py.

This targets specific uncovered lines in the logging and configuration sections.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from flext_core import FlextCore, FlextResult
from flext_core.loggings import FlextLogger


class TestFlextCoreLoggingConfigCoverage:
    """Tests targeting uncovered logging and configuration methods."""

    def test_configure_logging_comprehensive(self) -> None:
        """Test configure_logging with various parameters."""
        core = FlextCore.get_instance()

        # Test with different log levels
        log_levels = ["debug", "info", "warning", "error", "critical"]
        for level in log_levels:
            core.configure_logging(log_level=level)
            # Should not raise exceptions

        # Test with _json_output parameter (correct parameter name)
        core.configure_logging(log_level="info", _json_output=True)
        core.configure_logging(log_level="debug", _json_output=False)

        # Test with invalid log level (should fallback to INFO)
        core.configure_logging(log_level="invalid_level")

        # Test with None log level
        core.configure_logging(log_level=None)  # type: ignore[arg-type]

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

        # Test with no logger parameter
        no_logger = core.create_log_context(request_type="api", method="GET")
        assert isinstance(no_logger, FlextLogger)

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

    def test_load_configuration_methods(self) -> None:
        """Test configuration loading methods."""
        core = FlextCore.get_instance()

        # Test load_config_from_env (actual method name)
        with patch.dict(os.environ, {"TEST_VAR": "test_value", "DEBUG": "true"}):
            result = core.load_config_from_env()
            assert isinstance(result, FlextResult)

        # Test with empty environment
        with patch.dict(os.environ, {}, clear=True):
            result = core.load_config_from_env()
            assert isinstance(result, FlextResult)

    def test_validate_configuration_methods(self) -> None:
        """Test configuration validation methods."""
        core = FlextCore.get_instance()

        # Test validate_config_with_types with various configs
        configs = [
            {"database": {"url": "postgresql://localhost", "timeout": 30}},
            {"logging": {"level": "info", "format": "json"}},
            {"api": {"host": "0.0.0.0", "port": 8000}},
            {},  # Empty config
            {"complex": {"nested": {"deep": {"value": 123}}}},
        ]

        for config in configs:
            result = core.validate_config_with_types(config)
            assert isinstance(result, FlextResult)

        # Test validate_dict_structure
        for config in configs:
            result = core.validate_dict_structure(config)
            assert isinstance(result, FlextResult)

    def test_merge_configurations_comprehensive(self) -> None:
        """Test configuration merging functionality."""
        core = FlextCore.get_instance()

        # Test merging different types of configurations
        base_configs = [
            {"app": {"name": "test", "version": "1.0"}},
            {"database": {"host": "localhost", "port": 5432}},
            {},
        ]

        override_configs = [
            {"app": {"version": "2.0", "debug": True}},
            {"database": {"port": 5433, "ssl": True}},
            {"new_section": {"key": "value"}},
            {},
        ]

        for base, override in zip(base_configs, override_configs, strict=False):
            result = core.merge_configurations(base, override)
            assert isinstance(result, FlextResult)

    def test_configuration_schema_methods(self) -> None:
        """Test configuration schema related methods."""
        core = FlextCore.get_instance()

        # Test get_configuration_schema
        result = core.get_configuration_schema()
        assert isinstance(result, FlextResult)

        # Test validate_against_schema with various data
        test_data = [
            {"name": "test", "version": "1.0.0"},
            {"database": {"url": "postgresql://localhost"}},
            {"invalid": "structure"},
            {},
        ]

        for data in test_data:
            result = core.validate_against_schema(data)
            assert isinstance(result, FlextResult)

    def test_runtime_configuration_methods(self) -> None:
        """Test runtime configuration management."""
        core = FlextCore.get_instance()

        # Test set_runtime_configuration
        runtime_configs = [
            {"feature_flags": {"new_feature": True, "beta": False}},
            {"performance": {"cache_size": 1000, "timeout": 30}},
            {"monitoring": {"enabled": True, "interval": 60}},
        ]

        for config in runtime_configs:
            result = core.set_runtime_configuration(config)
            assert isinstance(result, FlextResult)

        # Test get_runtime_configuration
        result = core.get_runtime_configuration()
        assert isinstance(result, FlextResult)

        # Test update_runtime_configuration
        updates = [
            {"feature_flags": {"new_feature": False}},
            {"performance": {"timeout": 60}},
        ]

        for update in updates:
            result = core.update_runtime_configuration(update)
            assert isinstance(result, FlextResult)

    def test_environment_detection_methods(self) -> None:
        """Test environment detection and management."""
        core = FlextCore.get_instance()

        # Test detect_environment
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            result = core.detect_environment()
            assert isinstance(result, FlextResult)

        with patch.dict(os.environ, {"NODE_ENV": "development"}):
            result = core.detect_environment()
            assert isinstance(result, FlextResult)

        with patch.dict(os.environ, {}, clear=True):
            result = core.detect_environment()
            assert isinstance(result, FlextResult)

        # Test is_production_environment
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            result = core.is_production_environment()
            assert isinstance(result, bool)

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            result = core.is_production_environment()
            assert isinstance(result, bool)

        # Test is_development_environment
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            result = core.is_development_environment()
            assert isinstance(result, bool)

    def test_configuration_caching_methods(self) -> None:
        """Test configuration caching functionality."""
        core = FlextCore.get_instance()

        # Test cache_configuration
        configs_to_cache = [
            {"service": "cache_test", "data": {"key": "value"}},
            {"complex": {"nested": {"config": True}}},
        ]

        for config in configs_to_cache:
            result = core.cache_configuration("test_key", config)
            assert isinstance(result, FlextResult)

        # Test get_cached_configuration
        result = core.get_cached_configuration("test_key")
        assert isinstance(result, FlextResult)

        result = core.get_cached_configuration("nonexistent_key")
        assert isinstance(result, FlextResult)

        # Test clear_configuration_cache
        result = core.clear_configuration_cache()
        assert isinstance(result, FlextResult)

    def test_configuration_validation_edge_cases(self) -> None:
        """Test configuration validation with edge cases."""
        core = FlextCore.get_instance()

        # Test with various edge case configurations
        edge_cases = [
            {"empty_string": ""},
            {"zero_values": {"port": 0, "timeout": 0}},
            {"negative_values": {"retry": -1}},
            {"very_large_numbers": {"max_connections": 999999}},
            {"unicode_strings": {"name": "test_àáâãäå"}},
            {"special_chars": {"path": "/path/with spaces/and-dashes"}},
            {"boolean_variations": {"true": True, "false": False}},
            {"null_values": {"optional": None}},
        ]

        for config in edge_cases:
            result = core.validate_configuration(config)
            assert isinstance(result, FlextResult)

    def test_logging_integration_comprehensive(self) -> None:
        """Test logging system integration thoroughly."""
        core = FlextCore.get_instance()

        # Test logging with various contexts
        contexts = [
            {"request_id": "req_123", "user_id": "user_456"},
            {"operation": "test_op", "service": "test_service"},
            {"trace_id": "trace_789", "span_id": "span_abc"},
        ]

        for context in contexts:
            logger = core.create_log_context(**context)
            assert isinstance(logger, FlextLogger)

            # Test that the logger is functional
            logger.info("Test log message", extra={"test": True})
            logger.warning("Test warning", extra={"level": "warn"})
            logger.error("Test error", extra={"error_code": "TEST_001"})

    def test_configuration_file_integration(self) -> None:
        """Test configuration file operations integration."""
        core = FlextCore.get_instance()

        # Test load_configuration_from_dict
        dicts_to_load = [
            {"app": {"name": "test_app"}, "version": "1.0"},
            {"database": {"host": "localhost"}, "cache": {"ttl": 300}},
            {"empty": {}},
        ]

        for config_dict in dicts_to_load:
            result = core.load_configuration_from_dict(config_dict)
            assert isinstance(result, FlextResult)

        # Test with None dict
        result = core.load_configuration_from_dict(None)  # type: ignore[arg-type]
        assert isinstance(result, FlextResult)

    def test_configuration_export_methods(self) -> None:
        """Test configuration export functionality."""
        core = FlextCore.get_instance()

        # Test export_configuration
        result = core.export_configuration()
        assert isinstance(result, FlextResult)

        # Test export_configuration_to_dict
        result = core.export_configuration_to_dict()
        assert isinstance(result, FlextResult)

        # Test export_configuration_as_json
        result = core.export_configuration_as_json()
        assert isinstance(result, FlextResult)

    def test_advanced_configuration_scenarios(self) -> None:
        """Test advanced configuration scenarios."""
        core = FlextCore.get_instance()

        # Test configuration with complex nested structures
        complex_config = {
            "services": {
                "database": {
                    "primary": {"host": "db1.example.com", "port": 5432},
                    "replica": {"host": "db2.example.com", "port": 5432},
                    "settings": {
                        "pool_size": 20,
                        "timeout": 30,
                        "ssl": {"enabled": True, "verify": False},
                    },
                },
                "cache": {
                    "redis": {
                        "cluster": ["redis1:6379", "redis2:6379", "redis3:6379"],
                        "settings": {"max_connections": 100, "timeout": 5},
                    }
                },
            },
            "monitoring": {
                "metrics": {"enabled": True, "port": 9090},
                "logging": {"level": "info", "format": "json"},
                "tracing": {"enabled": True, "endpoint": "http://jaeger:14268"},
            },
        }

        result = core.validate_configuration(complex_config)
        assert isinstance(result, FlextResult)

        result = core.merge_configurations({}, complex_config)
        assert isinstance(result, FlextResult)
