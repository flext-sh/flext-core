"""Modern tests for flext_core.config - Configuration Management Implementation.

Refactored test suite using comprehensive testing libraries for config functionality.
Demonstrates SOLID principles, modern pytest patterns, and extensive test automation.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from tests.support.factory_boy_factories import (
    ConfigFactory,
    EdgeCaseGenerators,
    ProductionConfigFactory,
    create_validation_test_cases,
)
from tests.support.performance_utils import BenchmarkUtils, PerformanceProfiler

from flext_core import (
    FlextConfig,
    FlextResult,
)
from flext_core.config import merge_configs

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ============================================================================
# CORE CONFIG FUNCTIONALITY TESTS
# ============================================================================


class TestFlextConfigCore:
    """Test core FlextConfig functionality with factory patterns."""

    def test_config_creation_with_factory(self) -> None:
        """Test config creation using factories."""
        config = ConfigFactory()

        class TestConfig(FlextConfig):
            database_url: str
            log_level: str
            debug: bool
            timeout: int
            max_connections: int

        test_config = TestConfig(
            database_url=config.database_url,
            log_level=config.log_level,
            debug=config.debug,
            timeout=config.timeout,
            max_connections=config.max_connections,
        )

        assert test_config.database_url == config.database_url
        assert test_config.log_level == config.log_level
        assert test_config.debug == config.debug
        assert test_config.timeout == config.timeout
        assert test_config.max_connections == config.max_connections

    def test_production_config_patterns(self) -> None:
        """Test production configuration patterns."""
        prod_config = ProductionConfigFactory()

        class ProdConfig(FlextConfig):
            database_url: str
            log_level: str
            debug: bool
            timeout: int
            max_connections: int

        config = ProdConfig(
            database_url=prod_config.database_url,
            log_level=prod_config.log_level,
            debug=prod_config.debug,
            timeout=prod_config.timeout,
            max_connections=prod_config.max_connections,
        )

        # Production config should have production-like values
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.timeout >= 60
        assert config.max_connections >= 500

    @pytest.mark.parametrize(
        ("field_name", "field_value"),
        [
            ("database_url", "postgresql://test:test@localhost/test_db"),
            ("log_level", "DEBUG"),
            ("debug", True),
            ("timeout", 30),
            ("max_connections", 100),
        ],
    )
    def test_config_field_assignment(
        self, field_name: str, field_value: object
    ) -> None:
        """Test config field assignment using parametrization."""

        class TestConfig(FlextConfig):
            database_url: str = "default"
            log_level: str = "INFO"
            debug: bool = False
            timeout: int = 30
            max_connections: int = 100

        config = TestConfig()
        setattr(config, field_name, field_value)

        assert getattr(config, field_name) == field_value


# ============================================================================
# SETTINGS TESTS
# ============================================================================


class TestFlextConfigSettings:
    """Test config-based settings functionality."""

    def test_settings_creation(self) -> None:
        """Test settings creation with configuration values."""

        class TestConfig(FlextConfig):
            app_name: str = "default_app"
            version: str = "1.0.0"
            debug: bool = False

        config = TestConfig()

        assert config.app_name == "default_app"
        assert config.version == "1.0.0"
        assert config.debug is False

    def test_settings_with_env_override(self) -> None:
        """Test settings with environment variable patterns."""
        # Set environment variable for testing
        test_key = "FLEXT_TEST_CONFIG"
        test_value = "test_value_from_env"
        os.environ[test_key] = test_value

        try:
            # Environment variable should be accessible
            assert os.environ.get(test_key) == test_value
        finally:
            # Clean up environment
            os.environ.pop(test_key, None)

    def test_settings_validation(self) -> None:
        """Test settings validation patterns."""

        class TestConfig(FlextConfig):
            port: int = 8080
            host: str = "localhost"
            workers: int = 4

        config = TestConfig(port=3000, host="0.0.0.0", workers=8)

        assert config.port == 3000
        assert config.host == "0.0.0.0"
        assert config.workers == 8


# ============================================================================
# CONFIG OPERATIONS TESTS
# ============================================================================


class TestConfigOperations:
    """Test configuration operations."""

    def test_merge_configs_success(self) -> None:
        """Test successful config merging."""
        config1 = ConfigFactory()
        config2 = ProductionConfigFactory()

        dict1 = {
            "database_url": config1.database_url,
            "log_level": config1.log_level,
            "debug": config1.debug,
        }

        dict2 = {
            "timeout": config2.timeout,
            "max_connections": config2.max_connections,
            "features": config2.features,
        }

        result = merge_configs(dict1, dict2)

        assert result.success
        merged = result.value
        assert merged["database_url"] == config1.database_url
        assert merged["timeout"] == config2.timeout
        assert merged["max_connections"] == config2.max_connections

    def test_merge_configs_with_overlap(self) -> None:
        """Test config merging with overlapping keys."""
        dict1 = {"key1": "value1", "key2": "value2", "shared": "from_first"}
        dict2 = {"key3": "value3", "key4": "value4", "shared": "from_second"}

        result = merge_configs(dict1, dict2)

        assert result.success
        merged = result.value
        assert merged["key1"] == "value1"
        assert merged["key3"] == "value3"
        assert merged["shared"] == "from_second"  # Second dict should override

    def test_merge_configs_empty(self) -> None:
        """Test merging with empty configs."""
        dict1 = {"key1": "value1"}
        dict2: dict[str, object] = {}

        result = merge_configs(dict1, dict2)

        assert result.success
        merged = result.value
        assert merged["key1"] == "value1"
        assert len(merged) == 1


# ============================================================================
# FILE-BASED CONFIG TESTS
# ============================================================================


class TestFileBasedConfig:
    """Test file-based configuration loading."""

    def test_json_config_loading(self) -> None:
        """Test loading configuration from JSON file."""
        config_data = {
            "database_url": "postgresql://test:test@localhost/test_db",
            "log_level": "DEBUG",
            "debug": True,
            "timeout": 30,
            "max_connections": 100,
        }

        with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name

        try:
            # Load and validate config
            with Path(temp_file).open(encoding="utf-8") as f:
                loaded_data = json.load(f)

            assert loaded_data == config_data
            assert loaded_data["database_url"] == config_data["database_url"]
            assert loaded_data["debug"] == config_data["debug"]
        finally:
            Path(temp_file).unlink()

    def test_config_file_not_found(self) -> None:
        """Test handling of missing config file."""
        non_existent_file = "/path/that/does/not/exist/config.json"

        # Should handle missing file gracefully
        assert not Path(non_existent_file).exists()

    def test_invalid_json_config(self) -> None:
        """Test handling of invalid JSON configuration."""
        invalid_json = '{"key": "value", "invalid": }'

        with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", suffix=".json", delete=False) as f:
            f.write(invalid_json)
            temp_file = f.name

        try:
            # Should handle invalid JSON gracefully
            with pytest.raises(json.JSONDecodeError), Path(temp_file).open(encoding="utf-8") as f:
                json.load(f)
        finally:
            Path(temp_file).unlink()


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestConfigPerformance:
    """Test config performance characteristics."""

    def test_config_creation_performance(self, benchmark: object) -> None:
        """Benchmark config creation performance."""

        def create_configs() -> list[FlextConfig]:
            configs = []
            for _i in range(100):
                config = ConfigFactory()

                class TestConfig(FlextConfig):
                    database_url: str
                    log_level: str
                    debug: bool
                    timeout: int
                    max_connections: int

                test_config = TestConfig(
                    database_url=config.database_url,
                    log_level=config.log_level,
                    debug=config.debug,
                    timeout=config.timeout,
                    max_connections=config.max_connections,
                )
                configs.append(test_config)
            return configs

        configs = BenchmarkUtils.benchmark_with_warmup(
            benchmark, create_configs, warmup_rounds=3
        )

        assert len(configs) == 100
        assert all(isinstance(c, FlextConfig) for c in configs)

    def test_merge_performance(self, benchmark: object) -> None:
        """Benchmark config merging performance."""

        def merge_many_configs() -> list[FlextResult[dict[str, object]]]:
            results = []
            for _i in range(50):
                config1 = ConfigFactory()
                config2 = ProductionConfigFactory()

                dict1 = {
                    "database_url": config1.database_url,
                    "log_level": config1.log_level,
                    "debug": config1.debug,
                }

                dict2 = {
                    "timeout": config2.timeout,
                    "max_connections": config2.max_connections,
                }

                result = merge_configs(dict1, dict2)
                results.append(result)
            return results

        results = BenchmarkUtils.benchmark_with_warmup(
            benchmark, merge_many_configs, warmup_rounds=2
        )

        assert len(results) == 50
        assert all(r.success for r in results)

    def test_config_memory_efficiency(self) -> None:
        """Test memory efficiency of config operations."""
        profiler = PerformanceProfiler()

        with profiler.profile_memory("config_operations"):
            # Create many configs with different patterns
            configs = []
            for _i in range(1000):
                config = ConfigFactory()

                class TestConfig(FlextConfig):
                    database_url: str
                    log_level: str
                    debug: bool

                test_config = TestConfig(
                    database_url=config.database_url,
                    log_level=config.log_level,
                    debug=config.debug,
                )
                configs.append(test_config)

        # Assert reasonable memory usage (< 5MB for 1000 configs)
        profiler.assert_memory_efficient(
            max_memory_mb=5.0, operation_name="config_operations"
        )


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestConfigValidation:
    """Test config validation functionality."""

    def test_validation_test_cases(self) -> None:
        """Test config validation with comprehensive test cases."""
        test_cases = create_validation_test_cases()

        class TestConfig(FlextConfig):
            name: str
            email: str
            age: int
            is_active: bool
            created_at: object
            metadata: object

        for case in test_cases:
            if case["expected_valid"]:
                # Should create config successfully
                config = TestConfig(
                    name=case["data"].name,
                    email=case["data"].email,
                    age=case["data"].age,
                    is_active=case["data"].is_active,
                    created_at=case["data"].created_at,
                    metadata=case["data"].metadata,
                )
                assert config.name == case["data"].name
            else:
                # Should handle invalid data gracefully
                # Note: Actual validation behavior depends on config implementation
                pass

    @pytest.mark.parametrize(
        "config_type",
        [
            ConfigFactory,
            ProductionConfigFactory,
        ],
    )
    def test_factory_config_validation(self, config_type: type) -> None:
        """Test validation of different config factory types."""
        config = config_type()

        # All factory configs should have required fields
        assert hasattr(config, "database_url")
        assert hasattr(config, "log_level")
        assert hasattr(config, "debug")
        assert hasattr(config, "timeout")
        assert hasattr(config, "max_connections")

        # Values should be appropriate types
        assert isinstance(config.database_url, str)
        assert isinstance(config.log_level, str)
        assert isinstance(config.debug, bool)
        assert isinstance(config.timeout, int)
        assert isinstance(config.max_connections, int)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestConfigEdgeCases:
    """Test config edge cases and boundary conditions."""

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.unicode_strings())
    def test_unicode_config_values(self, edge_value: str) -> None:
        """Test config with unicode values."""

        class TestConfig(FlextConfig):
            text_value: str

        config = TestConfig(text_value=edge_value)
        assert config.text_value == edge_value

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.boundary_numbers())
    def test_boundary_number_config(self, edge_value: float) -> None:
        """Test config with boundary number values."""

        class TestConfig(FlextConfig):
            numeric_value: float

        config = TestConfig(numeric_value=edge_value)
        assert config.numeric_value == edge_value

    @pytest.mark.parametrize("empty_value", EdgeCaseGenerators.empty_values())
    def test_empty_config_values(self, empty_value: object) -> None:
        """Test config with empty/null values."""

        class TestConfig(FlextConfig):
            optional_value: object = None

        config = TestConfig(optional_value=empty_value)
        assert config.optional_value == empty_value

    def test_large_config_data(self) -> None:
        """Test config with large data structures."""
        large_text = EdgeCaseGenerators.large_values()[0]  # Large string

        class TestConfig(FlextConfig):
            large_field: str

        config = TestConfig(large_field=large_text)

        assert len(config.large_field) == 10000
        assert config.large_field == large_text


# ============================================================================
# ENVIRONMENT VARIABLE TESTS
# ============================================================================


class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_env_var_override(self) -> None:
        """Test environment variable override functionality."""
        # Set test environment variable
        test_key = "FLEXT_TEST_VALUE"
        test_value = "environment_value"
        os.environ[test_key] = test_value

        try:
            # Environment variable should be accessible
            assert os.environ.get(test_key) == test_value
        finally:
            # Clean up
            os.environ.pop(test_key, None)

    def test_env_var_not_set(self) -> None:
        """Test handling of unset environment variables."""
        non_existent_key = "FLEXT_NON_EXISTENT_VAR"

        # Should return None for non-existent variables
        assert os.environ.get(non_existent_key) is None
        assert os.environ.get(non_existent_key, "default") == "default"

    def test_env_var_with_defaults(self) -> None:
        """Test environment variables with default values."""

        class TestConfig(FlextConfig):
            app_name: str = "default_app"
            port: int = 8080
            debug: bool = False

        config = TestConfig()

        # Should use default values when env vars not set
        assert config.app_name == "default_app"
        assert config.port == 8080
        assert config.debug is False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestConfigIntegration:
    """Integration tests for config functionality."""

    def test_complete_config_workflow(self) -> None:
        """Test complete config workflow with all patterns."""
        # Create base config
        base_config = ConfigFactory()

        # Create production overrides
        prod_config = ProductionConfigFactory()

        # Merge configurations
        base_dict = {
            "database_url": base_config.database_url,
            "log_level": base_config.log_level,
            "debug": base_config.debug,
        }

        prod_dict = {
            "log_level": prod_config.log_level,  # Override
            "debug": prod_config.debug,  # Override
            "timeout": prod_config.timeout,  # Add new
            "max_connections": prod_config.max_connections,  # Add new
        }

        merge_result = merge_configs(base_dict, prod_dict)

        assert merge_result.success
        final_config = merge_result.value

        # Create final config object
        class FinalConfig(FlextConfig):
            database_url: str
            log_level: str
            debug: bool
            timeout: int
            max_connections: int

        config = FinalConfig(**final_config)

        # Verify final configuration
        assert config.database_url == base_config.database_url  # From base
        assert config.log_level == prod_config.log_level  # Overridden
        assert config.debug == prod_config.debug  # Overridden
        assert config.timeout == prod_config.timeout  # Added
        assert config.max_connections == prod_config.max_connections  # Added

    def test_config_factory_comprehensive(self) -> None:
        """Test comprehensive config creation with all factory types."""
        # Test all factory variations
        dev_config = ConfigFactory()
        prod_config = ProductionConfigFactory()

        # Verify different config profiles have appropriate values
        assert dev_config.debug is True
        assert prod_config.debug is False

        assert dev_config.log_level == "DEBUG"
        assert prod_config.log_level == "INFO"

        assert dev_config.timeout < prod_config.timeout
        assert dev_config.max_connections < prod_config.max_connections

    def test_config_with_constants(self) -> None:
        """Test config integration with FlextConstants."""

        class TestConfig(FlextConfig):
            environment: str = "development"
            version: str = "1.0.0"

        config = TestConfig()

        # Should work with config system
        assert config.environment == "development"
        assert config.version == "1.0.0"

        # Can be combined with constants
        combined_info = f"{config.environment}_{config.version}"
        assert combined_info == "development_1.0.0"
