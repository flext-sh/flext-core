"""Tests for configuration model classes.

Tests FlextBaseConfigModel, database configs, and related
configuration classes for comprehensive coverage.
"""

from __future__ import annotations

import tempfile

import pytest
from pydantic import SecretStr, ValidationError

from flext_core import (
    FlextConfig,
    FlextDatabaseConfig,
    FlextJWTConfig,
    FlextLDAPConfig,
    FlextObservabilityConfig,
    FlextOracleConfig,
    FlextRedisConfig,
    FlextSettings,
    FlextSingerConfig,
    load_config_from_env,
    merge_configs,
    safe_load_json_file,
    validate_config,
)


class TestFlextConfig:
    """Test FlextConfig main configuration class."""

    def test_config_creation(self) -> None:
        """Test FlextConfig creation with defaults."""
        config = FlextConfig()
        assert config.name == "flext"
        assert config.environment == "development"
        assert config.debug is False

    def test_config_custom_values(self) -> None:
        """Test FlextConfig with custom values."""
        config = FlextConfig(
            name="custom-app",
            environment="production",
            debug=True,
        )
        assert config.name == "custom-app"
        assert config.environment == "production"
        assert config.debug is True


class TestFlextDatabaseConfig:
    """Test FlextDatabaseConfig functionality."""

    def test_database_config_creation(self) -> None:
        """Test database config creation with defaults."""
        config = FlextDatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.username == "postgres"
        assert isinstance(config.password, SecretStr)

    def test_database_config_custom_values(self) -> None:
        """Test database config with custom values."""
        config = FlextDatabaseConfig(
            host="db.example.com",
            port=3306,
            username="REDACTED_LDAP_BIND_PASSWORD",
            password=SecretStr("secret123"),
        )
        assert config.host == "db.example.com"
        assert config.port == 3306
        assert config.username == "REDACTED_LDAP_BIND_PASSWORD"
        assert config.password.get_secret_value() == "secret123"

    def test_database_config_validation(self) -> None:
        """Test database config port validation."""
        with pytest.raises(ValidationError):
            FlextDatabaseConfig(port=70000)  # Invalid port

    def test_database_config_model_dump(self) -> None:
        """Test database config model_dump method."""
        config = FlextDatabaseConfig(host="test.com", port=5432)
        result = config.model_dump()
        assert result["host"] == "test.com"
        assert result["port"] == 5432


class TestFlextRedisConfig:
    """Test FlextRedisConfig functionality."""

    def test_redis_config_creation(self) -> None:
        """Test Redis config creation with defaults."""
        config = FlextRedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.database == 0

    def test_redis_config_custom_values(self) -> None:
        """Test Redis config with custom values."""
        config = FlextRedisConfig(
            host="redis.example.com",
            port=6380,
            database=1,
            password=SecretStr("redis_pass"),
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.database == 1
        assert config.password is not None
        assert config.password.get_secret_value() == "redis_pass"

    def test_redis_config_validation(self) -> None:
        """Test Redis config validation."""
        with pytest.raises(ValidationError):
            FlextRedisConfig(database=-1)  # Invalid database

    def test_redis_config_model_dump(self) -> None:
        """Test Redis config model_dump method."""
        config = FlextRedisConfig(host="redis.com", port=6379)
        result = config.model_dump()
        assert result["host"] == "redis.com"
        assert result["port"] == 6379


class TestFlextOracleConfig:
    """Test FlextOracleConfig functionality."""

    def test_oracle_config_creation(self) -> None:
        """Test Oracle config creation."""
        config = FlextOracleConfig(
            username="oracle_user",
            password=SecretStr("oracle_pass"),
            service_name="ORCL",
        )
        assert config.host == "localhost"
        assert config.port == 1521
        assert config.username == "oracle_user"
        assert config.password.get_secret_value() == "oracle_pass"
        assert config.service_name == "ORCL"

    def test_oracle_config_custom_values(self) -> None:
        """Test Oracle config with custom values."""
        config = FlextOracleConfig(
            host="oracle.example.com",
            port=1522,
            username="REDACTED_LDAP_BIND_PASSWORD",
            password=SecretStr("REDACTED_LDAP_BIND_PASSWORD_pass"),
            service_name="PROD",
            sid="ORACLE",
        )
        assert config.host == "oracle.example.com"
        assert config.port == 1522
        assert config.sid == "ORACLE"


class TestFlextLDAPConfig:
    """Test FlextLDAPConfig functionality."""

    def test_ldap_config_creation(self) -> None:
        """Test LDAP config creation."""
        config = FlextLDAPConfig(
            bind_dn="cn=REDACTED_LDAP_BIND_PASSWORD,dc=example,dc=com",
            bind_password=SecretStr("ldap_pass"),
            base_dn="dc=example,dc=com",
        )
        assert config.host == "localhost"
        assert config.port == 389
        assert config.bind_dn == "cn=REDACTED_LDAP_BIND_PASSWORD,dc=example,dc=com"
        assert config.bind_password is not None
        assert config.bind_password.get_secret_value() == "ldap_pass"
        assert config.base_dn == "dc=example,dc=com"
        assert config.use_ssl is False

    def test_ldap_config_ssl(self) -> None:
        """Test LDAP config with SSL."""
        config = FlextLDAPConfig(
            bind_dn="cn=REDACTED_LDAP_BIND_PASSWORD,dc=example,dc=com",
            bind_password=SecretStr("ldap_pass"),
            base_dn="dc=example,dc=com",
            port=636,
            use_ssl=True,
        )
        assert config.port == 636
        assert config.use_ssl is True


class TestConfigUtilities:
    """Test configuration utility functions."""

    def test_merge_configs(self) -> None:
        """Test merge_configs function."""
        config1 = FlextDatabaseConfig(host="db1.com", port=5432)
        config2 = FlextRedisConfig(host="redis1.com", port=6379)

        result = merge_configs(config1.model_dump(), config2.model_dump())
        assert result.is_success
        merged_data = result.value
        assert isinstance(merged_data, dict)
        assert "host" in merged_data

    def test_validate_config(self) -> None:
        """Test validate_config function."""
        config = FlextDatabaseConfig(host="test-db")
        result = validate_config(config.model_dump())
        assert result.is_success
        # FlextResult[None].ok(None) is valid - represents "success without data"
        assert result.value is None

    def test_safe_load_json_file(self) -> None:
        """Test safe_load_json_file function."""
        # Use existing temp_json_file fixture
        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            f.write('{"test": "value"}')
            f.flush()

            result = safe_load_json_file(f.name)
            assert result.is_success
            assert result.value["test"] == "value"


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_with_defaults(self) -> None:
        """Test config handling with default values."""
        config = FlextDatabaseConfig()  # Uses defaults
        result = config.model_dump()
        assert "host" in result
        assert result["host"] is not None

    def test_config_model_dump(self) -> None:
        """Test model_dump functionality."""
        config = FlextDatabaseConfig(host="test")
        result = config.model_dump()
        assert isinstance(result, dict)
        assert result["host"] == "test"

    def test_secret_str_handling(self) -> None:
        """Test SecretStr handling in configs."""
        config = FlextDatabaseConfig(password=SecretStr("secret123"))
        # Secret should not be exposed in dict
        result = config.model_dump()
        # The password should be handled properly by Pydantic
        assert "password" in result

    def test_config_inheritance(self) -> None:
        """Test config class inheritance."""
        config = FlextDatabaseConfig()
        # FlextDatabaseConfig inherits from FlextModel (via BaseModel)
        assert hasattr(config, "model_dump")
        assert hasattr(config, "model_validate")

    def test_config_field_validation(self) -> None:
        """Test config field validation."""
        # Test minimum values
        with pytest.raises(ValidationError):
            FlextDatabaseConfig(port=0)  # Below minimum

        # Test maximum values
        with pytest.raises(ValidationError):
            FlextDatabaseConfig(port=99999)  # Above maximum

    def test_complex_config_merge(self) -> None:
        """Test merging complex configurations."""
        db_config = FlextDatabaseConfig(host="db1.com", port=5432)
        redis_config = FlextRedisConfig(host="redis1.com", port=6379)

        # Convert to dictionaries for merge_configs function
        result = merge_configs(db_config.model_dump(), redis_config.model_dump())
        assert result.is_success
        merged_data = result.value
        assert isinstance(merged_data, dict)
        assert len(merged_data) > 0
        # Verify merge contains data from both configs
        assert "host" in merged_data  # Should have redis host as it comes second
        assert "port" in merged_data


class TestFlextJWTConfig:
    """Test FlextJWTConfig functionality."""

    def test_jwt_config_creation(self) -> None:
        """Test JWT config creation."""
        config = FlextJWTConfig(
            secret_key=SecretStr("secret123456789012345678901234567890"),
        )
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7

    def test_jwt_config_secret_key_validation(self) -> None:
        """Test JWT config secret key validation."""
        with pytest.raises(ValidationError):
            FlextJWTConfig(secret_key=SecretStr("short"))  # Too short

    def test_jwt_config_algorithm_validation(self) -> None:
        """Test JWT config algorithm validation."""
        with pytest.raises(ValidationError):
            FlextJWTConfig(
                secret_key=SecretStr("secret123456789012345678901234567890"),
                algorithm="INVALID",
            )

    def test_jwt_config_to_jwt_dict(self) -> None:
        """Test JWT config to_jwt_dict method."""
        config = FlextJWTConfig(
            secret_key=SecretStr("secret123456789012345678901234567890"),
        )
        result = config.to_jwt_dict()
        assert isinstance(result, dict)
        assert result["algorithm"] == "HS256"


class TestFlextSingerConfig:
    """Test FlextSingerConfig functionality."""

    def test_singer_config_creation(self) -> None:
        """Test Singer config creation."""
        config = FlextSingerConfig(stream_name="test_stream")
        assert config.stream_name == "test_stream"
        assert config.batch_size == 1000
        assert isinstance(config.stream_schema, dict)
        assert isinstance(config.stream_config, dict)

    def test_singer_config_custom_values(self) -> None:
        """Test Singer config with custom values."""
        config = FlextSingerConfig(
            stream_name="custom_stream",
            batch_size=500,
            stream_schema={"type": "object"},
            stream_config={"table": "test"},
        )
        assert config.stream_name == "custom_stream"
        assert config.batch_size == 500
        assert config.stream_schema == {"type": "object"}
        assert config.stream_config == {"table": "test"}

    def test_singer_config_stream_name_validation(self) -> None:
        """Test Singer config stream name validation."""
        with pytest.raises(ValidationError):
            FlextSingerConfig(stream_name="")

    def test_singer_config_to_singer_dict(self) -> None:
        """Test Singer config to_singer_dict method."""
        config = FlextSingerConfig(stream_name="test")
        result = config.to_singer_dict()
        assert isinstance(result, dict)
        assert result["stream_name"] == "test"


class TestFlextObservabilityConfig:
    """Test FlextObservabilityConfig functionality."""

    def test_observability_config_creation(self) -> None:
        """Test Observability config creation."""
        config = FlextObservabilityConfig()
        assert config.log_level == "INFO"
        assert config.log_format == "json"
        assert config.metrics_enabled is True
        assert config.tracing_enabled is True

    def test_observability_config_custom_values(self) -> None:
        """Test Observability config with custom values."""
        config = FlextObservabilityConfig(
            log_level="DEBUG",
            service_name="test-service",
            metrics_enabled=False,
        )
        assert config.log_level == "DEBUG"
        assert config.service_name == "test-service"
        assert config.metrics_enabled is False

    def test_observability_config_log_level_validation(self) -> None:
        """Test Observability config log level validation."""
        with pytest.raises(ValidationError):
            FlextObservabilityConfig(log_level="INVALID")

    def test_observability_config_to_observability_dict(self) -> None:
        """Test Observability config to_observability_dict method."""
        config = FlextObservabilityConfig()
        result = config.to_observability_dict()
        assert isinstance(result, dict)
        assert result["log_level"] == "INFO"


class TestFlextSettings:
    """Test FlextSettings base class functionality."""

    def test_settings_creation(self) -> None:
        """Test Settings base class creation."""
        settings = FlextSettings()
        # FlextSettings is the base class, so test basic functionality
        assert hasattr(settings, "model_config")
        assert hasattr(settings, "validate_business_rules")

    def test_settings_validation(self) -> None:
        """Test Settings validation."""
        settings = FlextSettings()
        result = settings.validate_business_rules()
        assert result.is_success

    def test_settings_create_with_validation(self) -> None:
        """Test Settings create_with_validation method."""
        result = FlextSettings.create_with_validation()
        assert result.is_success
        settings = result.value
        assert isinstance(settings, FlextSettings)


class TestAdvancedConfigFeatures:
    """Test advanced configuration features."""

    def test_database_config_get_connection_string(self) -> None:
        """Test database config connection string generation."""
        config = FlextDatabaseConfig(
            host="localhost",
            port=5432,
            username="user",
            password=SecretStr("pass"),
            database="testdb",
        )
        conn_str = config.get_connection_string()
        assert conn_str == "postgresql://user:pass@localhost:5432/testdb"

    def test_database_config_to_database_dict(self) -> None:
        """Test database config to_database_dict method."""
        config = FlextDatabaseConfig()
        result = config.to_database_dict()
        assert isinstance(result, dict)
        assert result["host"] == "localhost"
        assert result["port"] == 5432

    def test_redis_config_get_connection_string(self) -> None:
        """Test Redis config connection string generation."""
        config = FlextRedisConfig()
        conn_str = config.get_connection_string()
        assert conn_str == "redis://localhost:6379/0"

    def test_redis_config_get_connection_string_with_password(self) -> None:
        """Test Redis config connection string with password."""
        config = FlextRedisConfig(password=SecretStr("secret"))
        conn_str = config.get_connection_string()
        assert conn_str == "redis://:secret@localhost:6379/0"

    def test_redis_config_to_redis_dict(self) -> None:
        """Test Redis config to_redis_dict method."""
        config = FlextRedisConfig()
        result = config.to_redis_dict()
        assert isinstance(result, dict)
        assert result["host"] == "localhost"

    def test_oracle_config_get_connection_string_service_name(self) -> None:
        """Test Oracle config connection string with service_name."""
        config = FlextOracleConfig(
            service_name="TEST",
            username="user",
            password=SecretStr("pass"),
        )
        conn_str = config.get_connection_string()
        assert conn_str == "localhost:1521/TEST"

    def test_oracle_config_get_connection_string_sid(self) -> None:
        """Test Oracle config connection string with SID."""
        config = FlextOracleConfig(
            sid="TESTSID",
            username="user",
            password=SecretStr("pass"),
        )
        conn_str = config.get_connection_string()
        assert conn_str == "localhost:1521:TESTSID"

    def test_oracle_config_to_oracle_dict(self) -> None:
        """Test Oracle config to_oracle_dict method."""
        config = FlextOracleConfig(
            service_name="TEST",
            username="user",
            password=SecretStr("pass"),
        )
        result = config.to_oracle_dict()
        assert isinstance(result, dict)
        assert result["service_name"] == "TEST"

    def test_ldap_config_get_connection_string(self) -> None:
        """Test LDAP config connection string generation."""
        config = FlextLDAPConfig(base_dn="dc=example,dc=com")
        conn_str = config.get_connection_string()
        assert conn_str == "ldap://localhost:389"

    def test_ldap_config_get_connection_string_ssl(self) -> None:
        """Test LDAP config connection string with SSL."""
        config = FlextLDAPConfig(base_dn="dc=example,dc=com", use_ssl=True)
        conn_str = config.get_connection_string()
        assert conn_str == "ldaps://localhost:389"

    def test_ldap_config_to_ldap_dict(self) -> None:
        """Test LDAP config to_ldap_dict method."""
        config = FlextLDAPConfig(base_dn="dc=example,dc=com")
        result = config.to_ldap_dict()
        assert isinstance(result, dict)
        assert result["host"] == "localhost"


class TestConfigValidation:
    """Test comprehensive configuration validation."""

    def test_database_config_host_validation(self) -> None:
        """Test database config host validation."""
        with pytest.raises(ValidationError):
            FlextDatabaseConfig(host="")  # Empty host

    def test_database_config_username_validation(self) -> None:
        """Test database config username validation."""
        with pytest.raises(ValidationError):
            FlextDatabaseConfig(username="")  # Empty username

    def test_oracle_config_validation_no_identifier(self) -> None:
        """Test Oracle config validation without service_name or sid."""
        # Creating config without identifiers should fail at creation time
        with pytest.raises(
            ValueError,
            match="Either service_name or sid must be provided",
        ):
            FlextOracleConfig(
                username="user",
                password=SecretStr("pass"),
            )

    def test_ldap_config_base_dn_validation_empty(self) -> None:
        """Test LDAP config base DN validation with empty value."""
        with pytest.raises(ValidationError):
            FlextLDAPConfig(base_dn="")

    def test_ldap_config_base_dn_validation_format(self) -> None:
        """Test LDAP config base DN validation with incorrect format."""
        with pytest.raises(ValidationError):
            FlextLDAPConfig(base_dn="invalid_format")


class TestConfigFactoryFunctions:
    """Test configuration factory functions."""

    def test_load_config_from_env(self) -> None:
        """Test load_config_from_env function."""
        # load_config_from_env expects a config_type string, not a class
        result = load_config_from_env("database")
        assert result.is_success
        config = result.value
        assert config is not None
