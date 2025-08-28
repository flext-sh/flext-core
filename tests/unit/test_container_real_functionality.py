"""Real functionality tests for container module without mocks.

Tests the actual FlextContainer implementation with FlextTypes.Config integration,
StrEnum validation, and real execution paths.

Created to achieve comprehensive test coverage with actual functionality validation,
following the user's requirement for real tests without mocks.
"""

from __future__ import annotations

import time

import pytest

from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer, get_flext_container
from flext_core.typings import FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextContainerRealFunctionality:
    """Test real FlextContainer functionality without mocks."""

    def test_container_initialization_with_config_real(self) -> None:
        """Test container initialization with FlextTypes.Config."""
        config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "log_level": FlextConstants.Config.LogLevel.ERROR.value,
            "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
            "config_source": FlextConstants.Config.ConfigSource.FILE.value,
            "max_services": 500,
            "service_timeout": 60000,
            "enable_auto_wire": False,
            "enable_factory_cache": False,
        }

        container = FlextContainer(config=config)

        # Verify container was created
        assert container is not None
        assert container.get_service_count() == 0

        # Verify configuration was applied by getting config summary
        summary_result = container.get_configuration_summary()
        assert summary_result.success is True

        summary = summary_result.unwrap()
        assert summary["container_config"]["environment"] == "production"
        assert summary["container_config"]["log_level"] == "ERROR"
        assert summary["container_config"]["validation_level"] == "strict"

    def test_container_configuration_validation_real(self) -> None:
        """Test container configuration with StrEnum validation."""
        container = FlextContainer()

        # Test valid configuration
        valid_config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.STAGING.value,
            "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
            "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
        }

        result = container.configure_container(valid_config)
        assert result.success is True

        # Verify configuration was applied
        config_result = container.get_container_config()
        assert config_result.success is True
        config = config_result.unwrap()
        assert config["environment"] == "staging"
        assert config["log_level"] == "DEBUG"
        assert config["validation_level"] == "loose"

    def test_container_configuration_invalid_enum_values_real(self) -> None:
        """Test container configuration with invalid StrEnum values."""
        container = FlextContainer()

        # Test invalid environment
        invalid_env_config: FlextTypes.Config.ConfigDict = {
            "environment": "invalid_environment"
        }
        result = container.configure_container(invalid_env_config)
        assert result.success is False
        assert "Invalid environment" in result.error

        # Test invalid log level
        invalid_log_config: FlextTypes.Config.ConfigDict = {
            "log_level": "INVALID_LEVEL"
        }
        result = container.configure_container(invalid_log_config)
        assert result.success is False
        assert "Invalid log_level" in result.error

        # Test invalid validation level
        invalid_val_config: FlextTypes.Config.ConfigDict = {
            "validation_level": "invalid_validation"
        }
        result = container.configure_container(invalid_val_config)
        assert result.success is False
        assert "Invalid validation_level" in result.error

    def test_environment_scoped_container_real(self) -> None:
        """Test creation of environment-scoped containers."""
        base_container = FlextContainer()

        # Test all valid environments
        environments = [
            FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
            FlextConstants.Config.ConfigEnvironment.STAGING.value,
            FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            FlextConstants.Config.ConfigEnvironment.TEST.value,
        ]

        for env in environments:
            scoped_result = base_container.create_environment_scoped_container(env)
            assert scoped_result.success is True

            scoped_container = scoped_result.unwrap()
            config_result = scoped_container.get_container_config()
            assert config_result.success is True

            config = config_result.unwrap()
            assert config["environment"] == env

            # Production should have stricter settings
            if env == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value:
                assert (
                    config["validation_level"]
                    == FlextConstants.Config.ValidationLevel.STRICT.value
                )
                assert config["service_timeout"] == 60000
            else:
                assert (
                    config["validation_level"]
                    == FlextConstants.Config.ValidationLevel.NORMAL.value
                )
                assert config["service_timeout"] == 30000

    def test_configuration_summary_real_execution(self) -> None:
        """Test configuration summary generation with real execution."""
        container = FlextContainer()

        # Register some test services
        container.register("test_service", {"data": "test"})
        container.register_factory("test_factory", lambda: {"factory": "data"})

        summary_result = container.get_configuration_summary()
        assert summary_result.success is True

        summary = summary_result.unwrap()

        # Verify structure
        assert "container_config" in summary
        assert "service_statistics" in summary
        assert "environment_info" in summary
        assert "performance_settings" in summary
        assert "available_enum_values" in summary

        # Verify service statistics
        stats = summary["service_statistics"]
        assert stats["total_services"] == 2
        assert "test_service" in stats["service_names"]
        assert "test_factory" in stats["service_names"]

        # Verify enum values are available
        enums = summary["available_enum_values"]
        assert "development" in enums["environments"]
        assert "staging" in enums["environments"]
        assert "production" in enums["environments"]
        assert "INFO" in enums["log_levels"]
        assert "DEBUG" in enums["log_levels"]
        assert "ERROR" in enums["log_levels"]

    def test_service_registration_with_config_validation_real(self) -> None:
        """Test service registration respects container configuration."""
        config: FlextTypes.Config.ConfigDict = {
            "max_services": 2,
            "service_timeout": 5000,
        }
        container = FlextContainer(config=config)

        # Register services within limit
        result1 = container.register("service1", {"data": 1})
        assert result1.success is True

        result2 = container.register("service2", {"data": 2})
        assert result2.success is True

        # Verify services were registered
        assert container.get_service_count() == 2

        # Get services back
        service1_result = container.get("service1")
        assert service1_result.success is True
        assert service1_result.unwrap()["data"] == 1

        service2_result = container.get("service2")
        assert service2_result.success is True
        assert service2_result.unwrap()["data"] == 2


class TestFlextContainerStrEnumIntegration:
    """Test StrEnum integration in container configuration."""

    def test_all_config_environment_values_work_real(self) -> None:
        """Test all ConfigEnvironment StrEnum values work in container."""
        container = FlextContainer()

        # Test each environment enum value
        for env_enum in FlextConstants.Config.ConfigEnvironment:
            config: FlextTypes.Config.ConfigDict = {"environment": env_enum.value}
            result = container.configure_container(config)
            assert result.success is True

            # Verify expected environment values
            assert env_enum.value in {
                "development",
                "staging",
                "production",
                "test",
                "local",
            }

    def test_all_log_level_values_work_real(self) -> None:
        """Test all LogLevel StrEnum values work in container."""
        container = FlextContainer()

        # Test each log level enum value
        for log_enum in FlextConstants.Config.LogLevel:
            config: FlextTypes.Config.ConfigDict = {"log_level": log_enum.value}
            result = container.configure_container(config)
            assert result.success is True

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
        """Test all ValidationLevel StrEnum values work in container."""
        container = FlextContainer()

        # Test each validation level enum value
        validation_levels = []
        for val_enum in FlextConstants.Config.ValidationLevel:
            config: FlextTypes.Config.ConfigDict = {"validation_level": val_enum.value}
            result = container.configure_container(config)
            assert result.success is True
            validation_levels.append(val_enum.value)

        # Verify we have expected validation levels
        assert "strict" in validation_levels
        assert "normal" in validation_levels
        assert "loose" in validation_levels
        assert "disabled" in validation_levels  # Include all possible values

    def test_config_source_enum_integration_real(self) -> None:
        """Test ConfigSource StrEnum integration."""
        container = FlextContainer()

        # Test each config source enum value
        for source_enum in FlextConstants.Config.ConfigSource:
            config: FlextTypes.Config.ConfigDict = {"config_source": source_enum.value}
            result = container.configure_container(config)
            assert result.success is True

            # Verify the config source was set
            config_result = container.get_container_config()
            assert config_result.success is True
            current_config = config_result.unwrap()
            assert current_config["config_source"] == source_enum.value


class TestContainerPerformanceReal:
    """Test real performance characteristics of container."""

    def test_container_configuration_performance_real(self) -> None:
        """Test configuration performance with real execution."""
        container = FlextContainer()

        config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
            "log_level": FlextConstants.Config.LogLevel.INFO.value,
            "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
        }

        # Measure configuration time
        start_time = time.perf_counter()

        # Configure multiple times to test performance
        for _ in range(100):
            result = container.configure_container(config)
            assert result.success is True

        end_time = time.perf_counter()

        # Should configure quickly (less than 100ms for 100 configurations)
        total_time = end_time - start_time
        assert total_time < 0.1  # Less than 100ms

    def test_service_registration_retrieval_performance_real(self) -> None:
        """Test service registration and retrieval performance."""
        container = FlextContainer()

        # Register many services
        start_time = time.perf_counter()

        for i in range(100):
            service = {"id": i, "data": f"service_{i}"}
            result = container.register(f"service_{i}", service)
            assert result.success is True

        registration_time = time.perf_counter() - start_time

        # Retrieve all services
        start_time = time.perf_counter()

        for i in range(100):
            result = container.get(f"service_{i}")
            assert result.success is True
            service = result.unwrap()
            assert service["id"] == i

        retrieval_time = time.perf_counter() - start_time

        # Performance should be reasonable
        assert registration_time < 0.1  # Less than 100ms
        assert retrieval_time < 0.05  # Less than 50ms

    def test_environment_scoped_container_performance_real(self) -> None:
        """Test environment scoped container creation performance."""
        base_container = FlextContainer()

        start_time = time.perf_counter()

        # Create multiple environment scoped containers
        for _ in range(50):
            result = base_container.create_environment_scoped_container(
                FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
            )
            assert result.success is True

            scoped_container = result.unwrap()
            assert scoped_container.get_service_count() == 0

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should create containers quickly (less than 50ms for 50 containers)
        assert total_time < 0.05


class TestGlobalContainerIntegration:
    """Test global container integration with FlextTypes.Config."""

    def test_global_container_access_real(self) -> None:
        """Test global container access with real functionality."""
        # Get global container
        container = get_flext_container()
        assert container is not None

        # Register a service globally
        result = container.register("global_test", {"global": True})
        assert result.success is True

        # Retrieve from different global access
        another_container = get_flext_container()
        service_result = another_container.get("global_test")
        assert service_result.success is True

        service = service_result.unwrap()
        assert service["global"] is True

        # Clean up
        container.unregister("global_test")

    def test_global_container_configuration_real(self) -> None:
        """Test global container configuration with FlextTypes.Config."""
        container = get_flext_container()

        # Configure global container
        config: FlextTypes.Config.ConfigDict = {
            "environment": FlextConstants.Config.ConfigEnvironment.TEST.value,
            "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
        }

        result = container.configure_container(config)
        assert result.success is True

        # Verify configuration through global access
        global_container = get_flext_container()
        config_result = global_container.get_container_config()
        assert config_result.success is True

        global_config = config_result.unwrap()
        assert global_config["environment"] == "test"
        assert global_config["log_level"] == "DEBUG"
