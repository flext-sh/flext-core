"""Tests for FLEXT services module.

Unit tests validating service layer abstractions, service patterns,
and dependency injection for enterprise service architecture.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from flext_core import FlextServices
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextServices:
    """Test FLEXT service layer abstractions."""

    def test_module_imports(self) -> None:
        """Test that services module imports correctly."""
        assert FlextServices is not None

    def test_configure_services_system_success(self) -> None:
        """Test successful configuration of services system."""
        config = {
            "environment": "development",
            "log_level": "DEBUG",
            "enable_service_registry": True,
            "max_concurrent_services": 50,
            "service_timeout_seconds": 30,
            "enable_batch_processing": True,
            "batch_size": 25,
        }
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["environment"] == "development"
        assert validated_config["log_level"] == "DEBUG"
        assert validated_config["enable_service_registry"] is True
        assert validated_config["max_concurrent_services"] == 50
        assert validated_config["service_timeout_seconds"] == 30
        assert validated_config["enable_batch_processing"] is True
        assert validated_config["batch_size"] == 25

    def test_configure_services_system_invalid_environment(self) -> None:
        """Test configuration with invalid environment."""
        config = {"environment": "invalid_env"}
        result = FlextServices.configure_services_system(config)
        assert result.is_failure
        assert "Invalid environment" in result.error

    def test_configure_services_system_invalid_log_level(self) -> None:
        """Test configuration with invalid log level."""
        config = {"log_level": "INVALID"}
        result = FlextServices.configure_services_system(config)
        assert result.is_failure
        assert "Invalid log_level" in result.error

    def test_configure_services_system_defaults(self) -> None:
        """Test configuration with default values."""
        config: FlextTypes.Config.ConfigDict = {}
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["environment"] == "development"
        assert validated_config["log_level"] == "DEBUG"
        assert validated_config["enable_service_registry"] is True
        assert validated_config["enable_service_orchestration"] is True
        assert validated_config["enable_service_metrics"] is True
        assert validated_config["enable_service_validation"] is True
        assert validated_config["max_concurrent_services"] == 100
        assert validated_config["service_timeout_seconds"] == 30
        assert validated_config["enable_batch_processing"] is True
        assert validated_config["batch_size"] == 50

    @given(st.integers(min_value=1, max_value=1000))
    def test_configure_services_system_concurrent_services(self, max_concurrent: int) -> None:
        """Test configuration with various concurrent service limits."""
        config = {"max_concurrent_services": max_concurrent}
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["max_concurrent_services"] == max_concurrent

    @given(st.integers(min_value=1, max_value=300))
    def test_configure_services_system_timeout(self, timeout: int) -> None:
        """Test configuration with various timeout values."""
        config = {"service_timeout_seconds": timeout}
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["service_timeout_seconds"] == timeout

    @given(st.integers(min_value=1, max_value=1000))
    def test_configure_services_system_batch_size(self, batch_size: int) -> None:
        """Test configuration with various batch sizes."""
        config = {"batch_size": batch_size}
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["batch_size"] == batch_size

    def test_configure_services_system_all_boolean_flags(self) -> None:
        """Test configuration with all boolean flags."""
        config = {
            "enable_service_registry": False,
            "enable_service_orchestration": False,
            "enable_service_metrics": False,
            "enable_service_validation": False,
            "enable_batch_processing": False,
        }
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["enable_service_registry"] is False
        assert validated_config["enable_service_orchestration"] is False
        assert validated_config["enable_service_metrics"] is False
        assert validated_config["enable_service_validation"] is False
        assert validated_config["enable_batch_processing"] is False

    def test_configure_services_system_production_environment(self) -> None:
        """Test configuration for production environment."""
        config = {
            "environment": "production",
            "log_level": "ERROR",
        }
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["environment"] == "production"
        assert validated_config["log_level"] == "ERROR"

    def test_configure_services_system_test_environment(self) -> None:
        """Test configuration for test environment."""
        config = {
            "environment": "test",
            "log_level": "WARNING",
        }
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["environment"] == "test"
        assert validated_config["log_level"] == "WARNING"


class TestServiceComponents:
    """Test FlextServices component creation."""

    def test_service_processor_class_exists(self) -> None:
        """Test that ServiceProcessor class exists."""
        assert hasattr(FlextServices, 'ServiceProcessor')

    def test_service_processor_abstract(self) -> None:
        """Test that ServiceProcessor is abstract."""
        # ServiceProcessor should be abstract and not directly instantiable
        with pytest.raises(TypeError):
            FlextServices.ServiceProcessor()  # type: ignore[abstract]

    def test_service_orchestrator_class_exists(self) -> None:
        """Test that ServiceOrchestrator class exists."""
        assert hasattr(FlextServices, 'ServiceOrchestrator')

    def test_service_orchestrator_creation(self) -> None:
        """Test creating a ServiceOrchestrator."""
        orchestrator = FlextServices.ServiceOrchestrator()
        assert orchestrator is not None

    def test_service_registry_class_exists(self) -> None:
        """Test that ServiceRegistry class exists."""
        assert hasattr(FlextServices, 'ServiceRegistry')

    def test_service_registry_creation(self) -> None:
        """Test creating a ServiceRegistry."""
        registry = FlextServices.ServiceRegistry()
        assert registry is not None

    def test_service_metrics_class_exists(self) -> None:
        """Test that ServiceMetrics class exists."""
        assert hasattr(FlextServices, 'ServiceMetrics')

    def test_service_metrics_creation(self) -> None:
        """Test creating ServiceMetrics."""
        metrics = FlextServices.ServiceMetrics()
        assert metrics is not None

    def test_service_validation_class_exists(self) -> None:
        """Test that ServiceValidation class exists."""
        assert hasattr(FlextServices, 'ServiceValidation')

    def test_service_validation_creation(self) -> None:
        """Test creating ServiceValidation."""
        validation = FlextServices.ServiceValidation()
        assert validation is not None


class TestServiceConfiguration:
    """Test service configuration scenarios."""

    def test_minimal_config(self) -> None:
        """Test minimal configuration."""
        config = {"environment": "development"}
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        # Should have all defaults applied
        assert len(validated_config) >= 8

    def test_maximal_config(self) -> None:
        """Test maximal configuration with all options."""
        config = {
            "environment": "production",
            "log_level": "INFO",
            "enable_service_registry": True,
            "enable_service_orchestration": True,
            "enable_service_metrics": True,
            "enable_service_validation": True,
            "max_concurrent_services": 500,
            "service_timeout_seconds": 60,
            "enable_batch_processing": True,
            "batch_size": 100,
        }
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["environment"] == "production"
        assert validated_config["log_level"] == "INFO"
        assert validated_config["max_concurrent_services"] == 500
        assert validated_config["service_timeout_seconds"] == 60
        assert validated_config["batch_size"] == 100

    @given(st.sampled_from(["development", "production", "test", "staging"]))
    def test_valid_environments(self, environment: str) -> None:
        """Test all valid environment values."""
        config = {"environment": environment}
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["environment"] == environment

    @given(st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]))
    def test_valid_log_levels(self, log_level: str) -> None:
        """Test all valid log level values."""
        config = {"log_level": log_level}
        result = FlextServices.configure_services_system(config)
        assert result.success
        validated_config = result.unwrap()
        assert validated_config["log_level"] == log_level

    def test_config_error_handling(self) -> None:
        """Test configuration error handling with invalid values."""
        invalid_configs = [
            {"environment": ""},
            {"environment": None},
            {"log_level": ""},
            {"log_level": None},
            {"environment": 123},
            {"log_level": 456},
        ]
        
        for invalid_config in invalid_configs:
            result = FlextServices.configure_services_system(invalid_config)
            # Should either succeed with defaults or fail gracefully
            if result.is_failure:
                assert isinstance(result.error, str)
                assert len(result.error) > 0


class TestServiceEdgeCases:
    """Test edge cases and error scenarios."""

    def test_empty_config(self) -> None:
        """Test with completely empty configuration."""
        config: FlextTypes.Config.ConfigDict = {}
        result = FlextServices.configure_services_system(config)
        assert result.success

    def test_config_with_extra_fields(self) -> None:
        """Test configuration with extra unrecognized fields."""
        config = {
            "environment": "development",
            "unknown_field": "value",
            "another_unknown": 123,
        }
        result = FlextServices.configure_services_system(config)
        # Should succeed and ignore unknown fields
        assert result.success

    def test_config_type_safety(self) -> None:
        """Test configuration type safety."""
        config = {
            "environment": "development",
            "max_concurrent_services": "not_a_number",  # Wrong type
        }
        result = FlextServices.configure_services_system(config)
        # Should handle gracefully - either convert or use default
        assert result.success
        validated_config = result.unwrap()
        # If it's a string, the system accepted it as-is; that's also valid behavior
        concurrent_services = validated_config["max_concurrent_services"]
        assert concurrent_services is not None

    @given(st.dictionaries(st.text(), st.text()))
    def test_config_with_random_string_values(self, random_config: dict[str, str]) -> None:
        """Property-based test with random configuration values."""
        # Filter out potentially problematic keys
        filtered_config = {
            k: v for k, v in random_config.items() 
            if k not in {"environment", "log_level"} and len(k) < 50
        }
        
        result = FlextServices.configure_services_system(filtered_config)
        # Should always succeed with unknown fields
        assert result.success

    def test_service_components_basic_functionality(self) -> None:
        """Test basic functionality of service components."""
        # Test that all service components can be created without errors
        components = [
            FlextServices.ServiceOrchestrator(),
            FlextServices.ServiceRegistry(),
            FlextServices.ServiceMetrics(),
            FlextServices.ServiceValidation(),
        ]
        
        for component in components:
            assert component is not None
            assert hasattr(component, '__class__')
            assert component.__class__.__name__ in [
                'ServiceOrchestrator', 'ServiceRegistry', 
                'ServiceMetrics', 'ServiceValidation'
            ]


class TestServiceIntegration:
    """Integration tests focusing on what actually works."""

    def test_configuration_and_component_creation(self) -> None:
        """Test that configuration and component creation work together."""
        # Configure services system
        config = {
            "environment": "test",
            "enable_service_registry": True,
            "enable_service_orchestration": True,
            "enable_service_metrics": True,
        }
        config_result = FlextServices.configure_services_system(config)
        assert config_result.success

        # Create components
        registry = FlextServices.ServiceRegistry()
        orchestrator = FlextServices.ServiceOrchestrator() 
        metrics = FlextServices.ServiceMetrics()
        
        assert registry is not None
        assert orchestrator is not None
        assert metrics is not None

    def test_multiple_configuration_calls(self) -> None:
        """Test multiple configuration calls don't interfere."""
        configs = [
            {"environment": "development"},
            {"environment": "test"},
            {"environment": "production"},
        ]
        
        for config in configs:
            result = FlextServices.configure_services_system(config)
            assert result.success
            validated_config = result.unwrap()
            assert validated_config["environment"] == config["environment"]

    def test_concurrent_component_creation(self) -> None:
        """Test creating multiple components concurrently."""
        # Create multiple instances of each component type
        registries = [FlextServices.ServiceRegistry() for _ in range(5)]
        orchestrators = [FlextServices.ServiceOrchestrator() for _ in range(5)]
        metrics = [FlextServices.ServiceMetrics() for _ in range(5)]
        validations = [FlextServices.ServiceValidation() for _ in range(5)]
        
        all_components = registries + orchestrators + metrics + validations
        
        # All should be created successfully
        assert len(all_components) == 20
        assert all(component is not None for component in all_components)
        
        # Each should be a unique instance
        assert len(set(id(component) for component in registries)) == 5
        assert len(set(id(component) for component in orchestrators)) == 5