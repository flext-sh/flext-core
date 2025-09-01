"""Tests for config_new module to increase coverage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import cast

import pytest

from flext_core.config_new import FlextConfigCore, FlextConfigSchemaAppConfig
from flext_core.config_new.loader import FlextConfigLoader
from flext_core.config_new.providers import FlextConfigProviders
from flext_core.config_new.schema_app import (
    FlextConfigSchemaAppConfig as SchemaAppConfig,
)
from flext_core.typings import FlextTypes


class TestFlextConfigCore:
    """Test FlextConfigCore facade."""

    def test_from_defaults(self) -> None:
        """Test loading from defaults."""
        config = FlextConfigCore.from_defaults()
        assert isinstance(config, FlextConfigSchemaAppConfig)
        assert config.app_name == "flext-app"
        assert config.debug is False

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading from environment variables."""
        monkeypatch.setenv("FLEXT_APP_NAME", "test-app")
        monkeypatch.setenv("FLEXT_APP_VERSION", "1.0.0")
        monkeypatch.setenv("FLEXT_DEBUG", "true")

        config = FlextConfigCore.from_env()
        assert isinstance(config, FlextConfigSchemaAppConfig)
        assert config.app_name == "test-app"
        assert config.app_version == "1.0.0"
        assert config.debug is True

    def test_from_file_success(self) -> None:
        """Test loading from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "app_name": "file-app",
                    "app_version": "2.0.0",
                    "debug": False,
                    "config_source": "file",
                },
                f,
            )
            temp_path = f.name

        try:
            result = FlextConfigCore.from_file(temp_path)
            assert result.is_success
            config = result.unwrap()
            assert config.app_name == "file-app"
            assert config.app_version == "2.0.0"
        finally:
            Path(temp_path).unlink()

    def test_from_file_failure(self) -> None:
        """Test loading from non-existent file."""
        result = FlextConfigCore.from_file("/non/existent/file.json")
        assert result.is_failure
        assert "not found" in str(result.error).lower()

    def test_merged_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test merged configuration from multiple sources."""
        monkeypatch.setenv("FLEXT_APP_NAME", "env-app")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"app_version": "3.0.0"}, f)
            temp_path = f.name

        try:
            result = FlextConfigCore.merged(
                cli={"debug": True},
                env=True,
                file_path=temp_path,
                include_defaults=True,
            )
            assert result.is_success
            config = result.unwrap()
            assert config.debug is True  # From CLI (highest priority)
            assert config.app_name == "env-app"  # From env
            assert config.app_version == "3.0.0"  # From file
        finally:
            Path(temp_path).unlink()

    def test_validate_dict(self) -> None:
        """Test dictionary validation."""
        valid_dict: FlextTypes.Config.ConfigDict = {
            "app_name": "test",
            "app_version": "1.0.0",
            "debug": True,
            "config_source": "test",
        }
        result = FlextConfigCore.validate_dict(valid_dict)
        assert result.is_success

        invalid_dict: FlextTypes.Config.ConfigDict = {"invalid_key": "value"}
        result = FlextConfigCore.validate_dict(invalid_dict)
        assert result.is_failure

    def test_schema(self) -> None:
        """Test JSON schema generation."""
        schema = FlextConfigCore.schema()
        assert isinstance(schema, dict)
        assert "properties" in schema

        properties = cast("dict[str, object]", schema["properties"])
        if "app_name" not in properties:
            msg = "app_name not found in schema properties"
            raise AssertionError(msg)

    def test_dump_and_dump_json(self) -> None:
        """Test dumping config to dict and JSON."""
        config = FlextConfigCore.from_defaults()

        # Test dump to dict
        config_dict = FlextConfigCore.dump(config)
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "flext-app"

        # Test dump to JSON
        config_json = FlextConfigCore.dump_json(config)
        assert isinstance(config_json, str)
        parsed = json.loads(config_json)
        assert parsed["app_name"] == "flext-app"

    def test_with_overrides(self) -> None:
        """Test config with overrides."""
        config = FlextConfigCore.from_defaults()
        overrides: FlextTypes.Config.ConfigDict = {
            "app_name": "overridden-app",
            "debug": True,
        }

        result = FlextConfigCore.with_overrides(config, overrides)
        assert result.is_success
        new_config = result.unwrap()
        assert new_config.app_name == "overridden-app"
        assert new_config.debug is True

    def test_defaults_dict(self) -> None:
        """Test getting defaults as dict."""
        defaults = FlextConfigCore.defaults_dict()
        assert isinstance(defaults, dict)
        assert defaults["app_name"] == "flext-app"
        assert defaults["config_source"] == "constants"

    def test_env_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting env config as dict."""
        monkeypatch.setenv("FLEXT_APP_NAME", "env-test")

        env_dict = FlextConfigCore.env_dict()
        assert isinstance(env_dict, dict)
        assert env_dict["app_name"] == "env-test"
        assert env_dict["config_source"] == "env"

    def test_legacy_methods(self) -> None:
        """Test legacy compatibility methods."""
        # Test safe_load_json_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"app_name": "legacy-app"}, f)
            temp_path = f.name

        try:
            result = FlextConfigCore.safe_load_json_file(temp_path)
            assert result.is_success
            assert result.value["app_name"] == "legacy-app"

            # Test load_from_file
            result2 = FlextConfigCore.load_from_file(temp_path)
            assert result2.is_success
            assert result2.value.app_name == "legacy-app"

            # Test load_and_validate_from_file with required keys
            result3 = FlextConfigCore.load_and_validate_from_file(
                temp_path, required_keys=["app_name"]
            )
            assert result3.is_success

            result4 = FlextConfigCore.load_and_validate_from_file(
                temp_path, required_keys=["missing_key"]
            )
            assert result4.is_failure
            assert "Missing required keys" in str(result4.error)
        finally:
            Path(temp_path).unlink()

    def test_safe_get_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test safe environment variable retrieval."""
        monkeypatch.setenv("TEST_VAR", "test_value")

        result = FlextConfigCore.safe_get_env_var("TEST_VAR")
        assert result.is_success
        assert result.value == "test_value"

        result2 = FlextConfigCore.safe_get_env_var("MISSING_VAR", "default")
        assert result2.is_success
        assert result2.value == "default"

        result3 = FlextConfigCore.safe_get_env_var("MISSING_VAR")
        assert result3.is_failure

    def test_validate_business_rules(self) -> None:
        """Test business rules validation."""
        valid_data: FlextTypes.Config.ConfigDict = {
            "app_name": "business-app",
            "app_version": "1.0.0",
        }

        result = FlextConfigCore.validate_business_rules(valid_data)
        assert result.is_success
        assert result.value["app_name"] == "business-app"

    def test_create_with_validation(self) -> None:
        """Test creating config with validation."""
        config_data: FlextTypes.Config.ConfigDict = {
            "app_name": "validated-app",
            "app_version": "1.0.0",
        }

        result = FlextConfigCore.create_with_validation(config_data)
        assert result.is_success
        assert result.value.app_name == "validated-app"


class TestFlextConfigProviders:
    """Test FlextConfigProviders."""

    def test_from_constants(self) -> None:
        """Test loading from constants."""
        config = FlextConfigProviders.from_constants()
        assert config["app_name"] == "flext-app"
        assert config["config_source"] == "constants"

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading from environment."""
        monkeypatch.setenv("FLEXT_APP_NAME", "provider-env")
        monkeypatch.setenv("FLEXT_LOG_LEVEL", "DEBUG")

        config = FlextConfigProviders.from_env()
        assert config["app_name"] == "provider-env"
        assert config["log_level"] == "DEBUG"
        assert config["config_source"] == "env"

    def test_from_file_json(self) -> None:
        """Test loading from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"app_name": "json-app", "debug": True}, f)
            temp_path = f.name

        try:
            result = FlextConfigProviders.from_file(temp_path)
            assert result.is_success
            assert result.value["app_name"] == "json-app"
            assert result.value["debug"] is True
            assert result.value["config_source"] == "file"
        finally:
            Path(temp_path).unlink()

    def test_from_file_yaml(self) -> None:
        """Test loading from YAML file (should fail without yaml)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("app_name: yaml-app\n")
            temp_path = f.name

        try:
            result = FlextConfigProviders.from_file(temp_path)
            # YAML support requires pyyaml, which might not be installed
            if result.is_success:
                assert result.value["app_name"] == "yaml-app"
            else:
                assert (
                    "yaml" in str(result.error).lower()
                    or "unsupported" in str(result.error).lower()
                )
        finally:
            Path(temp_path).unlink()


class TestFlextConfigLoader:
    """Test FlextConfigLoader."""

    def test_load_and_validate(self) -> None:
        """Test loading and validating configs."""
        configs: list[FlextTypes.Config.ConfigDict] = [
            cast("FlextTypes.Config.ConfigDict", {"app_name": "first", "priority": 1}),
            cast(
                "FlextTypes.Config.ConfigDict", {"app_version": "2.0.0", "priority": 2}
            ),
            cast("FlextTypes.Config.ConfigDict", {"debug": True, "priority": 3}),
        ]

        result = FlextConfigLoader.load_and_validate(configs)
        assert result.is_success
        config = result.unwrap()
        assert config.debug is True  # Highest priority
        assert config.app_version == "2.0.0"  # Priority 2
        assert config.app_name == "first"  # Priority 1

    def test_merge_configs(self) -> None:
        """Test merging configurations."""
        configs: list[FlextTypes.Config.ConfigDict] = [
            cast("FlextTypes.Config.ConfigDict", {"app_name": "base", "debug": False}),
            cast(
                "FlextTypes.Config.ConfigDict", {"app_version": "1.0.0", "debug": True}
            ),
            cast("FlextTypes.Config.ConfigDict", {"log_level": "INFO"}),
        ]

        merged = FlextConfigLoader.merge_configs(configs)
        assert merged["app_name"] == "base"  # From first
        assert merged["debug"] is True  # Overridden by second
        assert merged["app_version"] == "1.0.0"  # From second
        assert merged["log_level"] == "INFO"  # From third

    def test_validate_config(self) -> None:
        """Test config validation."""
        valid_config: FlextTypes.Config.ConfigDict = cast(
            "FlextTypes.Config.ConfigDict", {"app_name": "test", "app_version": "1.0.0"}
        )

        result = FlextConfigLoader.validate_config(valid_config)
        assert result.is_success

        invalid_config: FlextTypes.Config.ConfigDict = cast(
            "FlextTypes.Config.ConfigDict", {}
        )
        result = FlextConfigLoader.validate_config(invalid_config)
        assert result.is_failure


class TestAppConfigSchema:
    """Test FlextConfigSchemaAppConfig schema."""

    def test_app_config_defaults(self) -> None:
        """Test FlextConfigSchemaAppConfig with defaults."""
        config = SchemaAppConfig()
        assert config.app_name == "flext-app"
        assert config.app_version == "0.1.0"
        assert config.debug is False
        assert config.log_level == "INFO"

    def test_app_config_custom(self) -> None:
        """Test FlextConfigSchemaAppConfig with custom values."""
        config = SchemaAppConfig(
            app_name="custom-app",
            app_version="2.0.0",
            debug=True,
            log_level="DEBUG",
            database_url="postgresql://localhost/test",
            redis_url="redis://localhost:6379",
            api_keys={"service1": "key1"},
            feature_flags={"feature1": True},
            custom_settings={"setting1": "value1"},
        )

        assert config.app_name == "custom-app"
        assert config.app_version == "2.0.0"
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.database_url == "postgresql://localhost/test"
        assert config.redis_url == "redis://localhost:6379"
        assert config.api_keys == {"service1": "key1"}
        assert config.feature_flags == {"feature1": True}
        assert config.custom_settings == {"setting1": "value1"}

    def test_app_config_validation(self) -> None:
        """Test FlextConfigSchemaAppConfig validation."""
        # Valid config
        config = SchemaAppConfig(app_name="valid", app_version="1.0.0")
        assert config.app_name == "valid"

        # Test model_dump
        dumped = config.model_dump()
        assert dumped["app_name"] == "valid"
        assert dumped["app_version"] == "1.0.0"

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert "valid" in json_str
        assert "1.0.0" in json_str
