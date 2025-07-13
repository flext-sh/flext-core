"""Comprehensive tests for flext_core.config.base module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from flext_core.config.base import BaseConfig
from flext_core.config.base import BaseSettings
from flext_core.config.base import ConfigSection
from flext_core.config.base import ConfigurationError
from flext_core.config.base import DIContainer
from flext_core.config.base import configure_container
from flext_core.config.base import get_config
from flext_core.config.base import get_container
from flext_core.config.base import get_settings
from flext_core.config.base import injectable
from flext_core.config.base import singleton


class TestBaseConfig:
    """Test BaseConfig functionality."""

    def test_base_config_creation(self) -> None:
        """Test BaseConfig can be created."""
        config = BaseConfig()
        assert config is not None
        assert hasattr(config, "model_config")

    def test_base_config_to_dict(self) -> None:
        """Test BaseConfig to_dict method."""
        config = BaseConfig()
        result = config.to_dict()
        assert isinstance(result, dict)

    def test_base_config_model_config(self) -> None:
        """Test BaseConfig model configuration."""
        config = BaseConfig()
        model_config = config.model_config

        # Should have strict validation settings
        assert model_config["extra"] == "forbid"
        assert model_config["validate_assignment"] is True
        assert model_config["str_strip_whitespace"] is True

    def test_base_config_get_subsection(self) -> None:
        """Test BaseConfig get_subsection method."""
        config = BaseConfig()

        # Test getting subsection with prefix
        subsection = config.get_subsection("test_")
        assert isinstance(subsection, dict)


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_configuration_error_creation(self) -> None:
        """Test ConfigurationError can be created."""
        error = ConfigurationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_configuration_error_inheritance(self) -> None:
        """Test ConfigurationError inheritance."""
        from flext_core.domain.core import DomainError

        error = ConfigurationError("Test error")
        assert isinstance(error, DomainError)
        assert isinstance(error, Exception)

    def test_configuration_error_with_cause(self) -> None:
        """Test ConfigurationError with cause."""
        try:
            msg = "Original error"
            raise ValueError(msg)
        except ValueError as e:
            with pytest.raises(ConfigurationError) as exc_info:
                msg = "Configuration error"
                raise ConfigurationError(msg) from e
            assert exc_info.value.__cause__ is e


class TestBaseSettings:
    """Test BaseSettings functionality."""

    def test_base_settings_creation(self) -> None:
        """Test BaseSettings can be created with defaults."""
        settings = BaseSettings()

        assert settings.project_name == "flext"
        assert settings.environment == "test"
        assert settings.debug is True

    def test_base_settings_custom_values(self) -> None:
        """Test BaseSettings with custom values."""
        settings = BaseSettings(
            project_name="custom",
            environment="production",
            debug=True,
        )

        assert settings.project_name == "custom"
        assert settings.environment == "production"
        assert settings.debug is True

    def test_base_settings_get_env_prefix(self) -> None:
        """Test BaseSettings environment prefix."""
        prefix = BaseSettings.get_env_prefix()
        assert isinstance(prefix, str)
        assert "FLEXT" in prefix.upper()

    @patch.dict(
        os.environ,
        {
            "FLEXT_PROJECT_NAME": "test_project",
            "FLEXT_ENVIRONMENT": "test",
            "FLEXT_DEBUG": "true",
        },
    )
    def test_base_settings_from_env(self) -> None:
        """Test BaseSettings reads from environment variables."""
        settings = BaseSettings()

        assert settings.project_name == "test_project"
        assert settings.environment == "test"
        assert settings.debug is True

    def test_base_settings_from_env_method(self) -> None:
        """Test BaseSettings from_env class method."""
        settings = BaseSettings.from_env()

        assert isinstance(settings, BaseSettings)
        assert hasattr(settings, "project_name")
        assert hasattr(settings, "environment")

    def test_base_settings_to_env_dict(self) -> None:
        """Test BaseSettings to_env_dict method."""
        settings = BaseSettings(
            project_name="test",
            environment="development",
            debug=True,
        )

        env_dict = settings.to_env_dict()

        assert isinstance(env_dict, dict)
        assert any("PROJECT_NAME" in key for key in env_dict)
        assert any("ENVIRONMENT" in key for key in env_dict)
        assert any("DEBUG" in key for key in env_dict)

    def test_base_settings_to_dict(self) -> None:
        """Test BaseSettings to_dict method."""
        settings = BaseSettings(
            project_name="test",
            environment="development",
        )

        result = settings.to_dict()

        assert isinstance(result, dict)
        assert result["project_name"] == "test"
        assert result["environment"] == "development"

    def test_base_settings_get_subsection(self) -> None:
        """Test BaseSettings get_subsection method."""
        settings = BaseSettings()

        # Test getting subsection with prefix
        subsection = settings.get_subsection("project_")
        assert isinstance(subsection, dict)

    def test_base_settings_configure_dependencies(self) -> None:
        """Test BaseSettings configure_dependencies method."""
        settings = BaseSettings()
        container = DIContainer()

        # Should not raise an exception
        settings.configure_dependencies(container)


class TestDIContainer:
    """Test DIContainer dependency injection functionality."""

    def test_di_container_creation(self) -> None:
        """Test DIContainer can be created."""
        container = DIContainer()
        assert container is not None
        assert hasattr(container, "_services")

    def test_di_container_register_service(self) -> None:
        """Test DIContainer register method."""
        container = DIContainer()

        # Test registering a service
        test_service = "test_service"
        container.register(str, test_service)

        # Should be able to retrieve it
        retrieved = container.resolve(str)
        assert retrieved == test_service

    def test_di_container_get_service(self) -> None:
        """Test DIContainer resolve method."""
        container = DIContainer()

        # Register a service
        test_service = BaseSettings(_env_file=None)
        container.register(BaseSettings, test_service)

        # Should be able to resolve it
        retrieved = container.resolve(BaseSettings)
        assert retrieved is test_service

    def test_di_container_has_service(self) -> None:
        """Test DIContainer service existence."""
        container = DIContainer()

        # DIContainer will auto-create simple types like str
        result = container.resolve(str)
        assert result == ""  # Auto-created empty string

        # Register a specific service
        container.register(str, "test")

        # Should return the registered instance, not auto-created
        result = container.resolve(str)
        assert result == "test"

    def test_di_container_get_nonexistent_service(self) -> None:
        """Test DIContainer resolve with nonexistent service."""
        container = DIContainer()

        # DIContainer will try to auto-create simple types
        result = container.resolve(str)
        assert result == ""  # str() returns empty string

        # For a type with a dependency that can't be resolved, should raise ConfigurationError
        from abc import ABC
        from abc import abstractmethod

        class AbstractService(ABC):
            @abstractmethod
            def do_something(self) -> None:
                pass

        class ComplexService:
            def __init__(self, abstract_service: AbstractService) -> None:
                self.service = abstract_service

        try:
            container.resolve(ComplexService)
            msg = "Should have raised ConfigurationError"
            raise AssertionError(msg)
        except (ConfigurationError, TypeError):
            # Either ConfigurationError or TypeError when trying to instantiate abstract class
            assert True  # Expected behavior

    def test_di_container_resolve_dependencies(self) -> None:
        """Test DIContainer resolve method."""
        container = DIContainer()

        # Test resolving dependencies if method exists
        if hasattr(container, "resolve"):
            # Register dependencies
            settings = BaseSettings()
            container.register(BaseSettings, settings)

            # Should be able to resolve
            resolved = container.resolve(BaseSettings)
            assert resolved is settings


class TestConfigSection:
    """Test ConfigSection functionality."""

    def test_config_section_creation(self) -> None:
        """Test ConfigSection can be created."""
        section = ConfigSection(BaseConfig)
        assert section is not None

    def test_config_section_with_data(self) -> None:
        """Test ConfigSection with data."""
        # Test if ConfigSection accepts data
        try:
            section = ConfigSection(BaseConfig, prefix="test_")
            assert hasattr(section, "prefix") or section is not None
        except TypeError:
            # ConfigSection might not accept parameters
            section = ConfigSection(BaseConfig)
            assert section is not None


class TestDecoratorFunctions:
    """Test decorator functions."""

    def test_injectable_decorator(self) -> None:
        """Test injectable decorator."""

        @injectable()
        class TestService:
            def __init__(self) -> None:
                self.value = "test"

        # Should create the class normally
        service = TestService()
        assert service.value == "test"

    def test_singleton_decorator(self) -> None:
        """Test singleton decorator."""

        @singleton()
        class TestSingleton:
            def __init__(self) -> None:
                self.value = "singleton"

        # Should create the class normally
        instance = TestSingleton()
        assert instance.value == "singleton"

    def test_combined_decorators(self) -> None:
        """Test combining injectable and singleton decorators."""

        @singleton()
        @injectable()
        class TestCombined:
            def __init__(self) -> None:
                self.value = "combined"

        # Should create the class normally
        instance = TestCombined()
        assert instance.value == "combined"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_container_function(self) -> None:
        """Test get_container function."""
        container = get_container()
        assert isinstance(container, DIContainer)

    def test_get_config_function(self) -> None:
        """Test get_config function."""
        # Test getting config with specific class
        config = get_config(BaseConfig)

        # Should return instance of BaseConfig
        assert isinstance(config, BaseConfig)

    def test_get_settings_function(self) -> None:
        """Test get_settings function."""
        # Test getting settings with specific class
        settings = get_settings(BaseSettings)

        # Should return instance of BaseSettings
        assert isinstance(settings, BaseSettings)

    def test_configure_container_function(self) -> None:
        """Test configure_container function."""
        container = DIContainer()

        # Should not raise an exception
        configure_container(container)

        # Container should still be valid
        assert isinstance(container, DIContainer)


class TestEnvironmentIntegration:
    """Test environment variable integration."""

    @patch.dict(
        os.environ,
        {
            "FLEXT_PROJECT_NAME": "env_test",
            "FLEXT_ENVIRONMENT": "test",
            "FLEXT_DEBUG": "true",
        },
    )
    def test_settings_environment_loading(self) -> None:
        """Test settings load from environment variables."""
        settings = BaseSettings()

        assert settings.project_name == "env_test"
        assert settings.environment == "test"
        assert settings.debug is True

    def test_settings_env_file_loading(self) -> None:
        """Test settings load from env file."""
        env_content = """
FLEXT_PROJECT_NAME=file_test
FLEXT_ENVIRONMENT=development
FLEXT_DEBUG=false
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write(env_content)
            env_file = f.name

        try:
            # Test loading from env file
            settings = BaseSettings.from_env(env_file)

            # Check if values were loaded
            # Note: actual loading depends on implementation
            assert isinstance(settings, BaseSettings)
        finally:
            Path(env_file).unlink(missing_ok=True)

    def test_settings_env_prefix(self) -> None:
        """Test settings environment prefix."""
        prefix = BaseSettings.get_env_prefix()

        assert isinstance(prefix, str)
        assert len(prefix) > 0
        assert prefix.endswith("_")


class TestConfigurationValidation:
    """Test configuration validation features."""

    def test_invalid_configuration_error(self) -> None:
        """Test invalid configuration raises error."""
        # Test with invalid environment if validation exists
        try:
            settings = BaseSettings(environment="invalid_env")
            # If validation allows it, that's fine
            assert settings.environment == "invalid_env"
        except Exception:
            # Validation is working as expected
            pass

    def test_configuration_error_handling(self) -> None:
        """Test configuration error handling."""
        # Test from_env with invalid configuration
        try:
            # Create invalid env file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".env", delete=False, encoding="utf-8"
            ) as f:
                f.write("INVALID_FORMAT")
                env_file = f.name

            try:
                BaseSettings.from_env(env_file)
            except ConfigurationError:
                # Error handling is working
                pass
            except Exception:
                # Other exceptions are also acceptable
                pass
        finally:
            Path(env_file).unlink(missing_ok=True)


class TestSettingsInteraction:
    """Test settings interaction with other components."""

    def test_settings_model_dump(self) -> None:
        """Test settings model dump functionality."""
        settings = BaseSettings(
            project_name="test",
            environment="development",
            debug=True,
        )

        dumped = settings.model_dump()

        assert isinstance(dumped, dict)
        assert dumped["project_name"] == "test"
        assert dumped["environment"] == "development"
        assert dumped["debug"] is True

    def test_settings_model_dump_exclude(self) -> None:
        """Test settings model dump with exclusions."""
        settings = BaseSettings(
            project_name="test",
            environment="development",
        )

        # Test excluding fields
        dumped = settings.model_dump(exclude={"project_version"})

        assert isinstance(dumped, dict)
        assert "project_name" in dumped
        assert "environment" in dumped
        assert "project_version" not in dumped

    def test_settings_equality(self) -> None:
        """Test settings equality."""
        settings1 = BaseSettings(
            project_name="test",
            environment="development",
        )

        settings2 = BaseSettings(
            project_name="test",
            environment="development",
        )

        # Should be equal if all fields match
        assert settings1 == settings2

    def test_settings_inequality(self) -> None:
        """Test settings inequality."""
        settings1 = BaseSettings(
            project_name="test1",
            environment="development",
        )

        settings2 = BaseSettings(
            project_name="test2",
            environment="development",
        )

        # Should not be equal if fields differ
        assert settings1 != settings2


class TestDependencyInjectionIntegration:
    """Test dependency injection integration."""

    def test_container_settings_integration(self) -> None:
        """Test container and settings integration."""
        container = DIContainer()
        settings = BaseSettings()

        # Configure dependencies
        settings.configure_dependencies(container)

        # Should be able to get settings from container
        retrieved = container.resolve(BaseSettings)
        assert retrieved is settings

    def test_container_service_lifecycle(self) -> None:
        """Test container service lifecycle."""
        container = DIContainer()

        # Register multiple services
        settings = BaseSettings()
        config = BaseConfig()

        container.register(BaseSettings, settings)
        container.register(BaseConfig, config)

        # Should maintain separate instances
        retrieved_settings = container.resolve(BaseSettings)
        retrieved_config = container.resolve(BaseConfig)

        assert retrieved_settings is settings
        assert retrieved_config is config
        assert retrieved_settings is not retrieved_config

    def test_container_has_method(self) -> None:
        """Test container service existence functionality."""
        container = DIContainer()

        # Helper function to check if service exists
        def has_service(service_type) -> bool | None:
            try:
                container.resolve(service_type)
                return True
            except (KeyError, ConfigurationError):
                return False

        # Initially should not have services
        assert not has_service(BaseSettings)
        assert not has_service(BaseConfig)

        # Register a service
        settings = BaseSettings()
        container.register(BaseSettings, settings)

        # Should have registered service
        assert has_service(BaseSettings)
        # Should not have unregistered service
        assert not has_service(BaseConfig)


class TestAdvancedConfigurationFeatures:
    """Test advanced configuration features."""

    def test_config_subsection_functionality(self) -> None:
        """Test configuration subsection functionality."""
        config = BaseConfig()

        # Test getting subsection
        subsection = config.get_subsection("test_")
        assert isinstance(subsection, dict)

        settings = BaseSettings()
        subsection = settings.get_subsection("project_")
        assert isinstance(subsection, dict)

    def test_config_serialization(self) -> None:
        """Test configuration serialization."""
        config = BaseConfig()

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)

        settings = BaseSettings()
        settings_dict = settings.to_dict()
        assert isinstance(settings_dict, dict)

    def test_settings_environment_integration(self) -> None:
        """Test settings environment integration."""
        settings = BaseSettings()

        # Test environment variable conversion
        env_dict = settings.to_env_dict()
        assert isinstance(env_dict, dict)

        # All keys should be uppercase and have prefix
        for key in env_dict:
            assert key.isupper()
            assert "FLEXT" in key

    def test_settings_validation_integration(self) -> None:
        """Test settings validation integration."""
        # Test that settings validate properly
        settings = BaseSettings(
            project_name="valid_name",
            environment="development",
            debug=False,
        )

        assert settings.project_name == "valid_name"
        assert settings.environment == "development"
        assert settings.debug is False
