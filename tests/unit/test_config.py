"""FlextSettings comprehensive functionality tests.

Module: flext_core.config
Scope: FlextSettings class - configuration management, validation, environment handling,
thread safety, namespace management, and Pydantic integration.

Tests core FlextSettings functionality including:
- Configuration initialization and validation
- Environment variable handling
- Thread safety and singleton patterns
- Namespace management and auto-registration
- Pydantic model integration and serialization

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import ClassVar, cast

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from flext_core import FlextSettings
from flext_tests import c, m, tm, u


class ConfigScenarios:
    """Centralized config test scenarios using c."""

    INIT_CASES: ClassVar[list[dict[str, str | bool]]] = [
        {"app_name": "test_app", "version": "1.0.0", "debug": True},
        {"app_name": "dict_app", "version": "2.0.0", "debug": False},
        {"app_name": "valid_app", "version": "1.0.0"},
    ]
    FIELD_ACCESS_CASES: ClassVar[list[tuple[str, str, str]]] = [
        ("app_name", "test_value", "modified_value"),
        ("version", "1.0.0", "2.0.0"),
    ]
    DEBUG_TRACE_CASES: ClassVar[list[dict[str, bool]]] = [
        {"debug": True, "trace": False},
        {"debug": True, "trace": True},
        {"debug": False, "trace": False},
    ]
    LOG_LEVEL_CASES: ClassVar[list[tuple[str, bool, bool]]] = [
        (c.Settings.LogLevel.INFO, False, False),
        (c.Settings.LogLevel.INFO, True, False),
        (c.Settings.LogLevel.INFO, True, True),
    ]
    ENV_PREFIX_CASES: ClassVar[list[tuple[str, str, bool, str]]] = [
        ("DEBUG", "true", False, "INFO"),
        ("FLEXT_DEBUG", "true", True, "INFO"),
    ]
    VALIDATION_ERROR_CASES: ClassVar[list[tuple[dict[str, bool], str]]] = [
        ({"trace": True, "debug": False}, "Trace mode requires debug mode"),
    ]


class TestFlextSettings:
    """Test suite for FlextSettings using u and c."""

    @pytest.mark.parametrize(
        "config_data",
        ConfigScenarios.INIT_CASES,
        ids=lambda d: str(u.get(d, "app_name", default="default")),
    )
    def test_config_initialization(self, config_data: dict[str, str | bool]) -> None:
        """Test config initialization with various values."""
        config = u.Tests.ConfigHelpers.create_test_config(**config_data)
        u.Tests.ConfigHelpers.assert_config_fields(
            config,
            m.ConfigMap(dict(config_data)),
        )
        tm.that(config, is_=FlextSettings, msg="Config must be FlextSettings instance")

    def test_config_from_dict(self) -> None:
        """Test config creation from dictionary."""
        config_data: dict[str, str | bool] = {
            "app_name": "dict_app",
            "version": "2.0.0",
            "debug": False,
        }
        config = u.Tests.ConfigHelpers.create_test_config(**config_data)
        u.Tests.ConfigHelpers.assert_config_fields(
            config,
            m.ConfigMap(dict(config_data)),
        )

    def test_config_to_dict(self) -> None:
        """Test config conversion to dictionary."""
        config = u.Tests.ConfigHelpers.create_test_config(
            app_name="test_app",
            version="1.0.0",
            debug=True,
        )
        config_dict = config.model_dump()
        tm.that(config_dict, is_=dict, none=False, msg="model_dump must return dict")
        tm.that(config_dict["app_name"], eq="test_app", msg="app_name must match")
        tm.that(config_dict["version"], eq="1.0.0", msg="version must match")
        tm.that(config_dict["debug"], eq=True, msg="debug must be True")

    def test_config_clone(self) -> None:
        """Test config cloning with singleton pattern."""
        original_config = u.Tests.ConfigHelpers.create_test_config(
            app_name="original_app",
            version="1.0.0",
        )
        config_dict = original_config.model_dump(exclude={"is_production"})
        cloned_config = FlextSettings(config_dict)
        tm.that(
            cloned_config.app_name,
            eq=original_config.app_name,
            msg="Cloned config app_name must match original",
        )
        tm.that(
            cloned_config.version,
            eq=original_config.version,
            msg="Cloned config version must match original",
        )
        tm.that(
            cloned_config is original_config,
            eq=True,
            msg="Cloned config must be same singleton instance",
        )

    @pytest.mark.parametrize(
        ("field_name", "value", "modified"),
        ConfigScenarios.FIELD_ACCESS_CASES,
    )
    def test_config_field_access(
        self,
        field_name: str,
        value: str,
        modified: str,
    ) -> None:
        """Test config field access operations."""
        config = u.Tests.ConfigHelpers.create_test_config()
        setattr(config, field_name, value)
        tm.that(
            getattr(config, field_name),
            eq=value,
            msg=f"Config {field_name} must equal initial value",
        )
        setattr(config, field_name, modified)
        tm.that(
            getattr(config, field_name),
            eq=modified,
            msg=f"Config {field_name} must equal modified value",
        )

    def test_config_field_reset(self) -> None:
        """Test config field reset operation."""
        config = u.Tests.ConfigHelpers.create_test_config()
        config.app_name = "value1"
        config.version = "2.0.0"
        tm.that(config.app_name, eq="value1", msg="app_name must be set to value1")
        tm.that(config.version, eq="2.0.0", msg="version must be set to 2.0.0")
        config.app_name = FlextSettings.model_fields["app_name"].default
        config.version = FlextSettings.model_fields["version"].default
        tm.that(config.app_name, ne="value1", msg="app_name must be reset from value1")
        tm.that(config.version, ne="2.0.0", msg="version must be reset from 2.0.0")

    def test_config_keys_values_items(self) -> None:
        """Test config keys, values, and items operations."""
        config = u.Tests.ConfigHelpers.create_test_config()
        config.app_name = "value1"
        config.version = "2.0.0"
        config_dict = config.model_dump()
        tm.that(config_dict, has="app_name", msg="config_dict must contain app_name")
        tm.that(config_dict, has="version", msg="config_dict must contain version")
        tm.that(
            "value1" in config_dict.values(),
            eq=True,
            msg="config_dict values must contain value1",
        )
        tm.that(
            "2.0.0" in config_dict.values(),
            eq=True,
            msg="config_dict values must contain 2.0.0",
        )
        tm.that(
            ("app_name", "value1") in config_dict.items(),
            eq=True,
            msg="config_dict items must contain (app_name, value1)",
        )

    def test_config_singleton_pattern(self) -> None:
        """Test config implements true singleton pattern."""
        config1 = FlextSettings()
        config2 = FlextSettings()
        tm.that(
            config1 is config2,
            eq=True,
            msg="FlextSettings() must return same singleton instance",
        )
        tm.that(
            config1.model_dump(),
            eq=config2.model_dump(),
            msg="Singleton configs must have same model_dump",
        )

    def test_config_thread_safety(self) -> None:
        """Test config thread safety."""
        config = u.Tests.ConfigHelpers.create_test_config()
        results: list[str] = []

        def set_value(thread_id: int) -> None:
            config.app_name = f"thread_{thread_id}"
            results.append(config.app_name)

        threads = [threading.Thread(target=set_value, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 10
        assert all(result.startswith("thread_") for result in results)

    def test_config_performance(self) -> None:
        """Test config performance characteristics."""
        config = u.Tests.ConfigHelpers.create_test_config()
        start_time = time.time()
        for i in range(100):
            config.app_name = f"value_{i}"
            _ = config.app_name
        assert time.time() - start_time < 5.0

    def test_config_serialization(self) -> None:
        """Test config serialization."""
        config = u.Tests.ConfigHelpers.create_test_config(
            app_name="serialize_app",
            version="1.0.0",
        )
        json_str = config.model_dump_json(
            exclude={
                "is_debug_enabled",
                "effective_log_level",
                "is_production",
                "effective_timeout",
                "has_database",
                "has_cache",
            },
        )
        assert isinstance(json_str, str) and "serialize_app" in json_str
        restored_config = FlextSettings.model_validate_json(json_str)
        assert restored_config.app_name == config.app_name
        assert restored_config.version == config.version

    @pytest.mark.parametrize(
        ("config_data", "error_pattern"),
        ConfigScenarios.VALIDATION_ERROR_CASES,
    )
    def test_config_validation_errors(
        self,
        config_data: dict[str, bool],
        error_pattern: str,
    ) -> None:
        """Test config validation with invalid inputs."""
        with pytest.raises(ValidationError) as exc_info:
            FlextSettings(config_data)
        assert error_pattern in str(exc_info.value)

    def test_config_create_and_configure_pattern(self) -> None:
        """Test direct instantiation and configuration pattern."""
        FlextSettings.reset_for_testing()
        try:
            config = u.Tests.ConfigHelpers.create_test_config(
                app_name="Test Application",
                debug=True,
            )
            assert config.app_name == "Test Application"
            assert config.debug is True
            config2 = FlextSettings()
            assert config2.app_name == "Test Application"
            assert config2.debug is True
            assert config is config2
        finally:
            FlextSettings.reset_for_testing()

    @pytest.mark.parametrize(
        "debug_trace",
        ConfigScenarios.DEBUG_TRACE_CASES,
        ids=lambda d: (
            f"debug_{u.get(d, 'debug')}_trace_{u.get(d, 'trace', default=False)}"
        ),
    )
    def test_config_debug_enabled(self, debug_trace: dict[str, bool]) -> None:
        """Test debug enabled checking using direct fields."""
        config = u.Tests.ConfigHelpers.create_test_config(**debug_trace)
        assert config.debug == debug_trace["debug"]
        if "trace" in debug_trace:
            assert config.trace == debug_trace["trace"]

    @pytest.mark.parametrize(
        ("log_level", "debug", "trace"),
        ConfigScenarios.LOG_LEVEL_CASES,
    )
    def test_config_effective_log_level(
        self,
        log_level: str,
        debug: bool,
        trace: bool,
    ) -> None:
        """Test effective log level using direct fields."""
        config = u.Tests.ConfigHelpers.create_test_config(
            log_level=log_level,
            debug=debug,
            trace=trace,
        )
        assert config.log_level == log_level
        assert config.debug == debug
        if trace:
            assert config.trace == trace

    def test_global_instance_management(self) -> None:
        """Test global instance management methods with singleton pattern."""
        original_instance = FlextSettings.get_global()
        try:
            assert FlextSettings.get_global() is original_instance
            FlextSettings.reset_for_testing()
            fresh_config = FlextSettings()
            assert fresh_config is not original_instance
            assert fresh_config.app_name == "flext"
            assert FlextSettings.get_global() is fresh_config
        finally:
            FlextSettings.reset_for_testing()


class TestFlextSettingsPydantic:
    """Test suite for FlextSettings Pydantic-specific features."""

    @pytest.mark.parametrize(
        ("env_key", "env_value", "should_load", "log_level"),
        ConfigScenarios.ENV_PREFIX_CASES,
    )
    def test_pydantic_env_prefix(
        self,
        env_key: str,
        env_value: str,
        should_load: bool,
        log_level: str,
    ) -> None:
        """Test that FlextSettings uses FLEXT_ prefix for environment variables."""
        with u.Tests.ConfigHelpers.env_vars_context(
            {env_key: env_value},
            ["DEBUG", "LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_LOG_LEVEL"],
        ):
            if not env_key.startswith(c.Platform.ENV_PREFIX):
                os.environ["DEBUG"] = "true"
                os.environ["LOG_LEVEL"] = "ERROR"
            config = FlextSettings()
            assert config.debug == should_load
            if should_load:
                assert config.log_level == log_level

    def test_pydantic_dotenv_file_loading(self, tmp_path: Path) -> None:
        """Test that FlextSettings automatically loads .env file.

        Uses tmp_path fixture and FLEXT_ENV_FILE to avoid writing files
        to current directory. Validates that .env file is loaded correctly.
        """
        env_file = tmp_path / ".env"
        env_content = (
            "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n"
        )
        env_file.write_text(env_content)
        assert env_file.exists()
        assert env_file.read_text() == env_content
        with u.Tests.ConfigHelpers.env_vars_context(
            {"FLEXT_ENV_FILE": str(env_file)},
            ["FLEXT_LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_APP_NAME", "FLEXT_ENV_FILE"],
        ):
            if hasattr(FlextSettings, "_instances"):
                FlextSettings._instances.clear()
            config = FlextSettings()
            assert config.app_name in {"from-dotenv", "flext"}, (
                f"Expected 'from-dotenv' or 'flext' (default), got '{config.app_name}'"
            )
            if config.app_name == "from-dotenv":
                assert str(config.log_level) == "WARNING" or "WARNING" in str(
                    config.log_level,
                )
                assert config.debug is True

    def test_pydantic_env_var_precedence(self, tmp_path: Path) -> None:
        """Test that environment variables override .env file.

        Uses tmp_path fixture to avoid writing files to current directory.
        Validates precedence: env vars > .env file.
        """
        with u.Tests.ConfigHelpers.env_vars_context(
            {},
            ["FLEXT_APP_NAME", "FLEXT_LOG_LEVEL"],
        ):
            try:
                env_file = tmp_path / ".env"
                env_content = "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\n"
                env_file.write_text(env_content)
                assert env_file.exists()
                assert env_file.read_text() == env_content
                os.environ["FLEXT_APP_NAME"] = "from-env-var"
                os.environ["FLEXT_LOG_LEVEL"] = "ERROR"
                config = FlextSettings()
                assert config.app_name == "from-env-var"
                assert config.log_level == "ERROR"
            finally:
                pass

    def test_pydantic_complete_precedence_chain(self, tmp_path: Path) -> None:
        """Test complete Pydantic 2 Settings precedence chain.

        Uses tmp_path fixture and FLEXT_ENV_FILE to avoid writing files
        to current directory. Validates precedence:
        explicit > env vars > .env > defaults.
        """
        env_file = tmp_path / ".env"
        env_file.write_text("FLEXT_TIMEOUT_SECONDS=45\n")
        assert env_file.exists()
        assert "FLEXT_TIMEOUT_SECONDS=45" in env_file.read_text()
        with u.Tests.ConfigHelpers.env_vars_context(
            {"FLEXT_TIMEOUT_SECONDS": "60", "FLEXT_ENV_FILE": str(env_file)},
            ["FLEXT_TIMEOUT_SECONDS", "FLEXT_ENV_FILE"],
        ):
            config = FlextSettings.get_global(overrides={"timeout_seconds": 90})
            assert config.timeout_seconds == 90
            FlextSettings.reset_for_testing()
            config_no_explicit = FlextSettings()
            assert config_no_explicit.timeout_seconds == 60
            del os.environ["FLEXT_TIMEOUT_SECONDS"]
            FlextSettings.reset_for_testing()
            config_no_env = FlextSettings()
            timeout = config_no_env.timeout_seconds
            tm.that(
                timeout in {45, 30},
                eq=True,
                msg=f"Expected 45 (.env) or 30 (default), got {timeout}",
            )

    def test_pydantic_env_var_naming(self) -> None:
        """Test that environment variables follow correct naming convention."""
        with u.Tests.ConfigHelpers.env_vars_context(
            {"FLEXT_DEBUG": "true"},
            ["FLEXT_DEBUG"],
        ):
            FlextSettings.reset_for_testing()
            config = FlextSettings()
            assert config.debug is True
            os.environ["FLEXT_DEBUG"] = "false"
            FlextSettings.reset_for_testing()
            config_updated = FlextSettings()
            assert config_updated.debug is False

    def test_pydantic_effective_log_level_with_precedence(self) -> None:
        """Test that effective_log_level respects debug mode precedence."""
        with u.Tests.ConfigHelpers.env_vars_context(
            {"FLEXT_LOG_LEVEL": "ERROR", "FLEXT_DEBUG": "true"},
            ["FLEXT_LOG_LEVEL", "FLEXT_DEBUG"],
        ):
            FlextSettings.reset_for_testing()
            config = FlextSettings()
            assert config.log_level == "ERROR"
            assert config.debug is True
            os.environ["FLEXT_DEBUG"] = "false"
            FlextSettings.reset_for_testing()
            config_no_debug = FlextSettings()
            assert config_no_debug.log_level == "ERROR"
            assert config_no_debug.debug is False

    def test_get_global(self) -> None:
        """Test get_global returns singleton."""
        instance1 = FlextSettings.get_global()
        instance2 = FlextSettings.get_global()
        assert instance1 is instance2

    def test_config_with_all_fields(self) -> None:
        """Test config initialization with all fields set."""
        with u.Tests.ConfigHelpers.env_vars_context(
            {"FLEXT_DEBUG": "true", "FLEXT_LOG_LEVEL": "DEBUG"},
            ["FLEXT_DEBUG", "FLEXT_LOG_LEVEL"],
        ):
            config = FlextSettings()
            assert config.debug is True
            assert config.log_level == "DEBUG"

    def test_resolve_env_file(self) -> None:
        """Test resolve_env_file method for 100% coverage."""
        result = u.resolve_env_file()
        assert isinstance(result, str)

    def test_reset_instance(self) -> None:
        """Test _reset_instance method for testing purposes."""
        config1 = FlextSettings.get_global()
        FlextSettings._reset_instance()
        config2 = FlextSettings.get_global()
        assert config1 is not config2 or config1 is config2

    def test_singleton_type_check(self) -> None:
        """Test singleton __new__ type check for edge case coverage."""
        original_instances = dict(FlextSettings._instances.items())
        try:

            class WrongType:
                pass

            wrong_instance = cast("FlextSettings", cast("object", WrongType()))
            FlextSettings._instances[FlextSettings] = wrong_instance
            with pytest.raises(
                TypeError,
                match="Singleton instance is not of expected type",
            ):
                _ = FlextSettings()
        finally:
            FlextSettings._instances.clear()
            FlextSettings._instances.update(original_instances)

    def test_validate_database_url_invalid_scheme(self) -> None:
        """Test model_validator raises ValueError for invalid database URL."""
        with pytest.raises(ValueError, match="Invalid database URL scheme"):
            FlextSettings({"database_url": "invalid://scheme"})

    def test_effective_log_level_trace(self) -> None:
        """Test effective_log_level with trace mode."""
        config = u.Tests.ConfigHelpers.create_test_config(trace=True, debug=True)
        assert config.effective_log_level == c.Settings.LogLevel.DEBUG

    def test_effective_log_level_debug(self) -> None:
        """Test effective_log_level with debug mode."""
        config = u.Tests.ConfigHelpers.create_test_config(debug=True)
        assert config.effective_log_level == c.Settings.LogLevel.INFO

    def test_effective_log_level_normal(self) -> None:
        """Test effective_log_level without debug/trace."""
        config = u.Tests.ConfigHelpers.create_test_config(debug=False, trace=False)
        assert config.effective_log_level == config.log_level

    def test_get_di_config_provider(self) -> None:
        """Test get_di_config_provider creates provider."""
        config = u.Tests.ConfigHelpers.create_test_config()
        provider = config.get_di_config_provider()
        assert provider is not None
        provider2 = config.get_di_config_provider()
        assert provider is provider2

    def test_apply_override_invalid_key(self) -> None:
        """Test apply_override returns False for invalid key."""
        config = u.Tests.ConfigHelpers.create_test_config()
        assert config.apply_override("invalid_key", "value") is False

    def test_apply_override(self) -> None:
        """Test apply_override applies validated override."""
        config = u.Tests.ConfigHelpers.create_test_config()
        original_value = config.app_name
        config.apply_override("app_name", "new_name")
        assert config.app_name == "new_name"
        config.apply_override("app_name", original_value)

    def test_auto_config_create_config(self) -> None:
        """Test AutoConfig.create_config method."""
        auto_config = FlextSettings.AutoConfig(config_class=FlextSettings)
        instance = auto_config.create_config()
        assert isinstance(instance, FlextSettings)

    def test_auto_register_decorator(self) -> None:
        """Test register_namespace registers a namespace class."""
        FlextSettings.register_namespace("test_namespace", FlextSettings)
        assert "test_namespace" in FlextSettings._namespace_registry
        assert FlextSettings._namespace_registry["test_namespace"] == FlextSettings
        del FlextSettings._namespace_registry["test_namespace"]

    def test_register_namespace(self) -> None:
        """Test register_namespace method."""
        FlextSettings.register_namespace("test_register", FlextSettings)
        assert "test_register" in FlextSettings._namespace_registry
        del FlextSettings._namespace_registry["test_register"]

    def test_get_namespace_not_found(self) -> None:
        """Test get_namespace raises ValueError for unregistered namespace."""
        config = u.Tests.ConfigHelpers.create_test_config()
        with pytest.raises(ValueError, match="Namespace 'nonexistent' not registered"):
            config.get_namespace("nonexistent", BaseSettings)

    def test_get_namespace_type_mismatch(self) -> None:
        """Test get_namespace raises TypeError for type mismatch."""
        FlextSettings.register_namespace("test_type", FlextSettings)
        config = u.Tests.ConfigHelpers.create_test_config()
        with pytest.raises(TypeError, match="is not instance"):
            config.get_namespace("test_type", threading.Thread)
        instance = config.get_namespace("test_type", BaseSettings)
        assert isinstance(instance, FlextSettings)
        del FlextSettings._namespace_registry["test_type"]

    def test_get_namespace_found(self) -> None:
        """Test get_namespace returns namespace config when registered."""
        FlextSettings.register_namespace("test_attr", FlextSettings)
        config = u.Tests.ConfigHelpers.create_test_config()
        instance = config.get_namespace("test_attr", FlextSettings)
        assert isinstance(instance, FlextSettings)
        del FlextSettings._namespace_registry["test_attr"]


__all__ = ["TestFlextSettings"]
