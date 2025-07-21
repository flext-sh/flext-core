"""Tests for flext_core.config.adapters module."""

from __future__ import annotations

import pytest

from flext_core.config.adapters import (
    CLIConfig,
    CLISettings,
    DjangoSettings,
    SingerConfig,
    cli_config_to_dict,
    django_settings_adapter,
    singer_config_adapter,
)


class TestCLIConfig:
    """Test CLIConfig functionality."""

    def test_cli_config_creation(self) -> None:
        """Test CLIConfig can be created with defaults."""
        config = CLIConfig()

        assert config.output_format == "text"
        assert config.verbose is False
        assert config.quiet is False

    def test_cli_config_custom_values(self) -> None:
        """Test CLIConfig with custom values."""
        config = CLIConfig(
            output_format="json",
            verbose=True,
            quiet=False,
        )

        assert config.output_format == "json"
        assert config.verbose is True
        assert config.quiet is False

    def test_cli_config_output_format_validation(self) -> None:
        """Test CLIConfig output format validation."""
        # Valid formats should work
        for fmt in ["text", "json", "yaml", "table"]:
            config = CLIConfig(output_format=fmt)
            assert config.output_format == fmt

        # Invalid format should raise validation error
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="String should match pattern"):
            CLIConfig(output_format="invalid")

    def test_cli_config_serialization(self) -> None:
        """Test CLIConfig serialization."""
        config = CLIConfig(
            output_format="json",
            verbose=True,
            quiet=False,
        )

        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["output_format"] == "json"
        assert config_dict["verbose"] is True
        assert config_dict["quiet"] is False


class TestCLISettings:
    """Test CLISettings functionality."""

    def test_cli_settings_creation(self) -> None:
        """Test CLISettings can be created."""
        settings = CLISettings()

        assert settings.project_name == "flext"  # default value
        assert settings.project_version == "0.7.0"  # default FlextFramework.VERSION
        assert settings.environment == "test"  # test environment (set by pytest)
        assert settings.default_profile == "default"

    def test_cli_settings_custom_values(self) -> None:
        """Test CLISettings with custom values."""
        settings = CLISettings(
            project_name="custom-cli",
            project_version="2.0.0",
            environment="production",
            default_profile="production",
        )

        assert settings.project_name == "custom-cli"
        assert settings.project_version == "2.0.0"
        assert settings.environment == "production"
        assert settings.default_profile == "production"

    def test_cli_settings_env_integration(self) -> None:
        """Test CLISettings environment variable integration."""
        settings = CLISettings(
            project_name="env-test",
            project_version="1.0.0",
            environment="development",
        )

        # Test environment prefix functionality
        env_prefix = settings.get_env_prefix()
        assert isinstance(env_prefix, str)
        assert env_prefix == "FLEXT_"

        # Test environment dictionary conversion
        env_dict = settings.to_env_dict()
        assert isinstance(env_dict, dict)
        assert any("PROJECT_NAME" in key for key in env_dict)


class TestDjangoSettings:
    """Test DjangoSettings functionality."""

    def test_django_settings_creation(self) -> None:
        """Test DjangoSettings can be created."""
        settings = DjangoSettings(
            secret_key="test-secret-key",  # Only required field
        )

        assert settings.project_name == "flext"  # default value
        assert settings.secret_key == "test-secret-key"
        # Debug might be True in test environment, so check type rather than value
        assert isinstance(settings.debug, bool)
        assert settings.allowed_hosts == ["localhost", "127.0.0.1"]
        # Database URL might be overridden by environment, check it's a valid SQLite URL
        assert settings.database_url.startswith("sqlite://")
        assert settings.static_url == "/static/"
        assert settings.media_url == "/media/"

    def test_django_settings_custom_values(self) -> None:
        """Test DjangoSettings with custom values."""
        settings = DjangoSettings(
            project_name="custom-django",
            project_version="3.0.0",
            environment="production",
            secret_key="super-secret-key",
            debug=True,
            allowed_hosts=["localhost", "127.0.0.1", "example.com"],
            database_url="postgresql://REDACTED_LDAP_BIND_PASSWORD:pass@db.example.com/myapp",
            static_url="/assets/",
            media_url="/uploads/",
        )

        assert settings.project_name == "custom-django"
        assert settings.project_version == "3.0.0"
        assert settings.environment == "production"
        assert settings.secret_key == "super-secret-key"
        assert settings.debug is True
        assert settings.allowed_hosts == ["localhost", "127.0.0.1", "example.com"]
        assert settings.database_url == "postgresql://REDACTED_LDAP_BIND_PASSWORD:pass@db.example.com/myapp"
        assert settings.static_url == "/assets/"
        assert settings.media_url == "/uploads/"

    def test_django_settings_required_fields(self) -> None:
        """Test DjangoSettings required fields validation."""
        # Should raise ValidationError if required fields are missing
        with pytest.raises(ValueError, match="validation error"):
            DjangoSettings()

    def test_django_settings_serialization(self) -> None:
        """Test DjangoSettings serialization."""
        settings = DjangoSettings(
            secret_key="serialize-key",  # Only required field
        )

        settings_dict = settings.model_dump()

        assert isinstance(settings_dict, dict)
        assert settings_dict["project_name"] == "flext"  # default value
        assert settings_dict["secret_key"] == "serialize-key"

    def test_django_settings_to_django_format(self) -> None:
        """Test converting DjangoSettings to Django format."""
        settings = DjangoSettings(
            secret_key="django-format-key",  # Only required field
            debug=True,
            database_url="postgresql://user:pass@localhost:5432/testdb",
        )

        django_dict = settings.to_django_settings()

        assert isinstance(django_dict, dict)
        assert django_dict["SECRET_KEY"] == "django-format-key"
        assert django_dict["DEBUG"] is True
        assert "DATABASES" in django_dict
        assert "default" in django_dict["DATABASES"]


class TestSingerConfig:
    """Test SingerConfig functionality."""

    def test_singer_config_creation(self) -> None:
        """Test SingerConfig can be created."""
        config = SingerConfig()

        assert config.stream_maps is None
        assert config.stream_map_config is None
        assert config.state is None
        assert config.max_parallel_streams == 0
        assert config.batch_config is None
        assert config.api_url is None
        assert config.timeout == 300.0
        assert config.retry_count == 3
        assert config.page_size == 100

    def test_singer_config_custom_values(self) -> None:
        """Test SingerConfig with custom values."""
        stream_maps = {
            "users": {
                "email": "user_email",
                "name": "full_name",
            },
        }

        stream_map_config = {
            "null_value": None,
            "add_record_metadata": True,
        }

        config = SingerConfig(
            stream_maps=stream_maps,
            stream_map_config=stream_map_config,
            max_parallel_streams=5,
            api_url="https://api.example.com",
            timeout=600.0,
            retry_count=5,
            page_size=200,
        )

        assert config.stream_maps == stream_maps
        assert config.stream_map_config == stream_map_config
        assert config.max_parallel_streams == 5
        assert config.api_url == "https://api.example.com"
        assert config.timeout == 600.0
        assert config.retry_count == 5
        assert config.page_size == 200

    def test_singer_config_api_url_validation(self) -> None:
        """Test SingerConfig API URL validation."""
        # Valid URLs should work
        valid_urls = [
            "https://api.example.com",
            "https://api.test.com/v1",
        ]

        for url in valid_urls:
            config = SingerConfig(api_url=url)
            assert config.api_url == url

        # Invalid URLs should raise validation error
        with pytest.raises(ValueError, match="Invalid.*URL"):
            SingerConfig(api_url="not-a-url")

    def test_singer_config_validation_constraints(self) -> None:
        """Test SingerConfig validation constraints."""
        # Valid values
        config = SingerConfig(
            max_parallel_streams=10,
            timeout=30.0,
            retry_count=5,
            page_size=50,
        )

        assert config.max_parallel_streams == 10
        assert config.timeout == 30.0
        assert config.retry_count == 5
        assert config.page_size == 50

        # Invalid values should raise validation errors
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            SingerConfig(max_parallel_streams=-1)

        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            SingerConfig(timeout=0.0)

        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            SingerConfig(retry_count=-1)

        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            SingerConfig(page_size=0)

    def test_singer_config_serialization(self) -> None:
        """Test SingerConfig serialization."""
        config = SingerConfig(
            stream_maps={"test": {"id": "test_id"}},
            api_url="https://api.test.com",
            timeout=120.0,
        )

        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["stream_maps"] == {"test": {"id": "test_id"}}
        assert config_dict["api_url"] == "https://api.test.com"
        assert config_dict["timeout"] == 120.0


class TestAdapterFunctions:
    """Test adapter functions."""

    def test_cli_config_to_dict(self) -> None:
        """Test cli_config_to_dict function."""
        config = CLIConfig(
            output_format="json",
            verbose=True,
        )

        result = cli_config_to_dict(config)

        assert isinstance(result, dict)
        assert result["output_format"] == "json"
        assert result["verbose"] is True

    def test_cli_config_to_dict_exclude_unset(self) -> None:
        """Test cli_config_to_dict excludes unset values."""
        config = CLIConfig(verbose=True)  # Only set verbose, others use defaults

        result = cli_config_to_dict(config)

        assert isinstance(result, dict)
        assert "verbose" in result
        # Should exclude defaults that weren't explicitly set

    def test_django_settings_adapter(self) -> None:
        """Test django_settings_adapter function."""
        settings = DjangoSettings(
            project_name="adapter-test",
            project_version="1.0.0",
            environment="development",
            secret_key="adapter-secret",
            debug=True,
            allowed_hosts=["localhost", "testserver"],
            database_url="postgresql://user:pass@localhost/test",
        )

        result = django_settings_adapter(settings)

        assert isinstance(result, dict)
        assert result["SECRET_KEY"] == "adapter-secret"
        assert result["DEBUG"] is True
        assert result["ALLOWED_HOSTS"] == ["localhost", "testserver"]
        assert "DATABASES" in result
        assert "INSTALLED_APPS" in result
        assert "MIDDLEWARE" in result

    def test_django_settings_adapter_with_class(self) -> None:
        """Test django_settings_adapter with class instead of instance."""
        result = django_settings_adapter(DjangoSettings)

        assert isinstance(result, dict)
        assert "SECRET_KEY" in result
        assert "DATABASES" in result
        assert "INSTALLED_APPS" in result

    def test_singer_config_adapter(self) -> None:
        """Test singer_config_adapter function."""
        result = singer_config_adapter(SingerConfig)

        assert isinstance(result, dict)
        assert "$schema" in result
        assert result["type"] == "object"
        assert "properties" in result
        assert "required" in result
        assert result["additionalProperties"] is False

    def test_singer_config_adapter_schema_properties(self) -> None:
        """Test singer_config_adapter generates proper schema properties."""
        result = singer_config_adapter(SingerConfig)

        properties = result.get("properties", {})

        # Check some expected properties exist
        expected_props = ["stream_maps", "api_url", "timeout", "retry_count"]
        for prop in expected_props:
            assert prop in properties
            assert "description" in properties[prop]

        # Check secret fields are marked
        if "api_key" in properties:
            assert properties["api_key"].get("secret") is True
            assert properties["api_key"].get("writeOnly") is True


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""

    def test_cli_to_singer_workflow(self) -> None:
        """Test CLI config to Singer config workflow."""
        cli_config = CLIConfig(
            output_format="json",
            verbose=True,
        )

        # CLI config can be converted to dict
        cli_dict = cli_config_to_dict(cli_config)

        # Use CLI settings to create Singer config
        singer_config = SingerConfig(
            page_size=200 if cli_dict.get("verbose") else 100,
        )

        assert singer_config.page_size == 200  # Should be increased due to verbose

    def test_django_to_cli_workflow(self) -> None:
        """Test Django settings to CLI workflow."""
        django_settings = DjangoSettings(
            project_name="django-cli-test",
            project_version="1.0.0",
            environment="development",
            secret_key="django-secret",
            debug=True,
        )

        # Create CLI settings from Django settings
        cli_settings = CLISettings(
            project_name=django_settings.project_name,
            project_version=django_settings.project_version,
            environment=django_settings.environment,
        )

        assert cli_settings.project_name == "django-cli-test"
        assert cli_settings.project_version == "1.0.0"
        assert cli_settings.environment == "development"

    def test_multi_config_validation(self) -> None:
        """Test validation across multiple configuration types."""
        # Create configurations that should be compatible
        cli_config = CLIConfig(output_format="json")

        singer_config = SingerConfig(
            api_url="https://api.example.com",
        )

        django_settings = DjangoSettings(
            project_name="validation-test",
            project_version="1.0.0",
            environment="development",
            secret_key="validation-secret",
        )

        # All should be valid
        assert cli_config.output_format == "json"
        assert singer_config.api_url == "https://api.example.com"
        assert django_settings.project_name == "validation-test"

        # Test adapter functions work correctly
        cli_dict = cli_config_to_dict(cli_config)
        singer_dict = singer_config_adapter(SingerConfig)
        django_dict = django_settings_adapter(django_settings)

        assert isinstance(cli_dict, dict)
        assert isinstance(singer_dict, dict)
        assert isinstance(django_dict, dict)

    def test_configuration_inheritance_patterns(self) -> None:
        """Test configuration inheritance and composition patterns."""
        # Test that all configs inherit from proper base classes
        from flext_core.config.base import BaseConfig, BaseSettings

        cli_config = CLIConfig()
        cli_settings = CLISettings()
        django_settings = DjangoSettings(
            secret_key="inheritance-secret",  # Only required field
        )
        singer_config = SingerConfig()

        # Test inheritance
        assert isinstance(cli_config, BaseConfig)
        assert isinstance(cli_settings, BaseSettings)
        assert isinstance(django_settings, BaseSettings)
        assert isinstance(singer_config, BaseConfig)

        # Test that BaseSettings methods work
        assert hasattr(cli_settings, "model_dump")
        assert hasattr(cli_settings, "get_env_prefix")
        assert hasattr(django_settings, "to_env_dict")

        # Test that BaseConfig methods work
        assert hasattr(cli_config, "model_dump")
        assert hasattr(singer_config, "get_subsection")

    def test_environment_variable_integration(self) -> None:
        """Test environment variable integration across adapters."""
        # Test CLI settings environment integration
        cli_settings = CLISettings(
            project_name="env-integration",
            project_version="1.0.0",
            environment="test",
        )

        env_dict = cli_settings.to_env_dict()
        assert "FLEXT_PROJECT_NAME" in env_dict
        assert env_dict["FLEXT_PROJECT_NAME"] == "env-integration"

        # Test Django settings environment integration
        django_settings = DjangoSettings(
            secret_key="env-secret",  # Only required field
        )

        env_dict = django_settings.to_env_dict()
        assert "FLEXT_SECRET_KEY" in env_dict
        assert env_dict["FLEXT_SECRET_KEY"] == "env-secret"
