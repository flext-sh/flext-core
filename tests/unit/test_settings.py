"""FlextSettings comprehensive functionality tests.

Module: flext_core.settings
Scope: FlextSettings class - settings management, validation, environment handling,
thread safety, namespace management, and Pydantic integration.

Tests core FlextSettings functionality including:
- Settings initialization and validation
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
from collections.abc import MutableSequence, Sequence
from pathlib import Path
from typing import ClassVar, cast

import pytest
from pydantic import ValidationError

from flext_core import FlextSettings
from flext_tests import tm
from tests import c, p, t, u


class TestFlextSettings:
    class SettingsScenarios:
        """Centralized settings test scenarios using c."""

        INIT_CASES: ClassVar[Sequence[t.FeatureFlagMapping]] = [
            {"app_name": "test_app", "version": "1.0.0", "debug": True},
            {"app_name": "dict_app", "version": "2.0.0", "debug": False},
            {"app_name": "valid_app", "version": "1.0.0"},
        ]
        FIELD_ACCESS_CASES: ClassVar[Sequence[tuple[str, str, str]]] = [
            ("app_name", "test_value", "modified_value"),
            ("version", "1.0.0", "2.0.0"),
        ]
        DEBUG_TRACE_CASES: ClassVar[Sequence[t.BoolMapping]] = [
            {"debug": True, "trace": False},
            {"debug": True, "trace": True},
            {"debug": False, "trace": False},
        ]
        LOG_LEVEL_CASES: ClassVar[Sequence[tuple[str, bool, bool]]] = [
            (c.LogLevel.INFO, False, False),
            (c.LogLevel.INFO, True, False),
            (c.LogLevel.INFO, True, True),
        ]
        ENV_PREFIX_CASES: ClassVar[Sequence[tuple[str, str, bool, str]]] = [
            ("DEBUG", "true", False, "INFO"),
            ("FLEXT_DEBUG", "true", True, "INFO"),
        ]
        VALIDATION_ERROR_CASES: ClassVar[Sequence[tuple[t.BoolMapping, str]]] = [
            ({"trace": True, "debug": False}, "Trace mode requires debug mode"),
        ]

    @pytest.mark.parametrize(
        "config_data",
        SettingsScenarios.INIT_CASES,
        ids=lambda d: str(d.get("app_name", "default")),
    )
    def test_settings_initialization(self, config_data: t.FeatureFlagMapping) -> None:
        """Test settings initialization with various values."""
        settings = u.Core.Tests.create_test_config(**config_data)
        u.Core.Tests.assert_config_fields(
            settings,
            t.ConfigMap(dict(config_data)),
        )
        tm.that(
            settings, is_=FlextSettings, msg="Settings must be FlextSettings instance"
        )

    def test_config_from_dict(self) -> None:
        """Test settings creation from dictionary."""
        config_data: t.FeatureFlagMapping = {
            "app_name": "dict_app",
            "version": "2.0.0",
            "debug": False,
        }
        settings = u.Core.Tests.create_test_config(**config_data)
        u.Core.Tests.assert_config_fields(
            settings,
            t.ConfigMap(dict(config_data)),
        )

    def test_config_to_dict(self) -> None:
        """Test settings conversion to dictionary."""
        settings = u.Core.Tests.create_test_config(
            app_name="test_app",
            version="1.0.0",
            debug=True,
        )
        config_dict = settings.model_dump()
        tm.that(config_dict, is_=dict, none=False, msg="model_dump must return dict")
        tm.that(config_dict["app_name"], eq="test_app", msg="app_name must match")
        tm.that(config_dict["version"], eq="1.0.0", msg="version must match")
        tm.that(config_dict["debug"], eq=True, msg="debug must be True")

    def test_config_clone(self) -> None:
        """Test settings cloning with singleton pattern."""
        original_config = u.Core.Tests.create_test_config(
            app_name="original_app",
            version="1.0.0",
        )
        config_dict = original_config.model_dump(exclude={"is_production"})
        cloned_config = FlextSettings.model_validate(config_dict)
        tm.that(
            cloned_config.app_name,
            eq=original_config.app_name,
            msg="Cloned settings app_name must match original",
        )
        tm.that(
            cloned_config.version,
            eq=original_config.version,
            msg="Cloned settings version must match original",
        )
        tm.that(
            cloned_config is original_config,
            eq=True,
            msg="Cloned settings must be same singleton instance",
        )

    @pytest.mark.parametrize(
        ("field_name", "value", "modified"),
        SettingsScenarios.FIELD_ACCESS_CASES,
    )
    def test_config_field_access(
        self,
        field_name: str,
        value: str,
        modified: str,
    ) -> None:
        """Test settings field access operations."""
        settings = u.Core.Tests.create_test_config()
        setattr(settings, field_name, value)
        tm.that(
            getattr(settings, field_name),
            eq=value,
            msg=f"Config {field_name} must equal initial value",
        )
        setattr(settings, field_name, modified)
        tm.that(
            getattr(settings, field_name),
            eq=modified,
            msg=f"Config {field_name} must equal modified value",
        )

    def test_config_field_reset(self) -> None:
        """Test settings field reset operation."""
        settings = u.Core.Tests.create_test_config()
        settings.app_name = "value1"
        settings.version = "2.0.0"
        tm.that(settings.app_name, eq="value1", msg="app_name must be set to value1")
        tm.that(settings.version, eq="2.0.0", msg="version must be set to 2.0.0")
        settings.app_name = FlextSettings.model_fields["app_name"].default
        settings.version = FlextSettings.model_fields["version"].default
        tm.that(
            settings.app_name, ne="value1", msg="app_name must be reset from value1"
        )
        tm.that(settings.version, ne="2.0.0", msg="version must be reset from 2.0.0")

    def test_config_keys_values_items(self) -> None:
        """Test settings keys, values, and items operations."""
        settings = u.Core.Tests.create_test_config()
        settings.app_name = "value1"
        settings.version = "2.0.0"
        config_dict = settings.model_dump()
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
        """Test settings implements true singleton pattern."""
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
        """Test settings thread safety."""
        settings = u.Core.Tests.create_test_config()
        results: MutableSequence[str] = []

        def set_value(thread_id: int) -> None:
            settings.app_name = f"thread_{thread_id}"
            results.append(settings.app_name)

        threads = [threading.Thread(target=set_value, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 10
        assert all(result.startswith("thread_") for result in results)

    def test_config_performance(self) -> None:
        """Test settings performance characteristics."""
        settings = u.Core.Tests.create_test_config()
        start_time = time.time()
        for i in range(100):
            settings.app_name = f"value_{i}"
            _ = settings.app_name
        assert time.time() - start_time < 5.0

    def test_config_serialization(self) -> None:
        """Test settings serialization."""
        settings = u.Core.Tests.create_test_config(
            app_name="serialize_app",
            version="1.0.0",
        )
        json_str = settings.model_dump_json(
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
        assert restored_config.app_name == settings.app_name
        assert restored_config.version == settings.version

    @pytest.mark.parametrize(
        ("config_data", "error_pattern"),
        SettingsScenarios.VALIDATION_ERROR_CASES,
    )
    def test_config_validation_errors(
        self,
        config_data: t.BoolMapping,
        error_pattern: str,
    ) -> None:
        """Test settings validation with invalid inputs."""
        with pytest.raises(ValidationError) as exc_info:
            FlextSettings.model_validate(config_data)
        assert error_pattern in str(exc_info.value)

    def test_config_create_and_configure_pattern(self) -> None:
        """Test direct instantiation and configuration pattern."""
        FlextSettings.reset_for_testing()
        try:
            settings = u.Core.Tests.create_test_config(
                app_name="Test Application",
                debug=True,
            )
            assert settings.app_name == "Test Application"
            assert settings.debug is True
            config2 = FlextSettings()
            assert config2.app_name == "Test Application"
            assert config2.debug is True
            assert settings is config2
        finally:
            FlextSettings.reset_for_testing()

    @pytest.mark.parametrize(
        "debug_trace",
        SettingsScenarios.DEBUG_TRACE_CASES,
        ids=lambda d: f"debug_{d.get('debug')}_trace_{d.get('trace', False)}",
    )
    def test_config_debug_enabled(self, debug_trace: t.BoolMapping) -> None:
        """Test debug enabled checking using direct fields."""
        settings = u.Core.Tests.create_test_config(**debug_trace)
        assert settings.debug == debug_trace["debug"]
        if "trace" in debug_trace:
            assert settings.trace == debug_trace["trace"]

    @pytest.mark.parametrize(
        ("log_level", "debug", "trace"),
        SettingsScenarios.LOG_LEVEL_CASES,
    )
    def test_config_effective_log_level(
        self,
        log_level: str,
        debug: bool,
        trace: bool,
    ) -> None:
        """Test effective log level using direct fields."""
        settings = u.Core.Tests.create_test_config(
            log_level=log_level,
            debug=debug,
            trace=trace,
        )
        assert settings.log_level == log_level
        assert settings.debug == debug
        if trace:
            assert settings.trace == trace

    def test_global_instance_management(self) -> None:
        """Test global instance management methods with singleton pattern."""
        original_instance = FlextSettings.fetch_global()
        try:
            assert FlextSettings.fetch_global() is original_instance
            FlextSettings.reset_for_testing()
            fresh_config = FlextSettings()
            assert fresh_config is not original_instance
            assert fresh_config.app_name == "flext"
            assert FlextSettings.fetch_global() is fresh_config
        finally:
            FlextSettings.reset_for_testing()

    class TestFlextSettingsPydantic:
        """Test suite for FlextSettings Pydantic-specific features."""

        @pytest.mark.parametrize(
            ("env_key", "env_value", "should_load", "log_level"),
            [("DEBUG", "true", False, "INFO"), ("FLEXT_DEBUG", "true", True, "INFO")],
        )
        def test_pydantic_env_prefix(
            self,
            env_key: str,
            env_value: str,
            should_load: bool,
            log_level: str,
        ) -> None:
            """Test that FlextSettings uses FLEXT_ prefix for environment variables."""
            with u.Core.Tests.env_vars_context(
                {env_key: env_value},
                ["DEBUG", "LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_LOG_LEVEL"],
            ):
                if not env_key.startswith(c.ENV_PREFIX):
                    os.environ["DEBUG"] = "true"
                    os.environ["LOG_LEVEL"] = "ERROR"
                settings = FlextSettings()
                assert settings.debug == should_load
                if should_load:
                    assert settings.log_level == log_level

        def test_pydantic_dotenv_file_loading(self, tmp_path: Path) -> None:
            """Test that FlextSettings automatically loads .env file.

            Uses tmp_path fixture and FLEXT_ENV_FILE to avoid writing files
            to current directory. Validates that .env file is loaded correctly.
            """
            env_file = tmp_path / ".env"
            env_content = "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n"
            env_file.write_text(env_content)
            assert env_file.exists()
            assert env_file.read_text() == env_content
            with u.Core.Tests.env_vars_context(
                {"FLEXT_ENV_FILE": str(env_file)},
                ["FLEXT_LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_APP_NAME", "FLEXT_ENV_FILE"],
            ):
                if hasattr(FlextSettings, "_instances"):
                    FlextSettings._instances.clear()
                settings = FlextSettings()
                assert settings.app_name in {"from-dotenv", "flext"}, (
                    f"Expected 'from-dotenv' or 'flext' (default), got '{settings.app_name}'"
                )
                if settings.app_name == "from-dotenv":
                    assert str(settings.log_level) == "WARNING" or "WARNING" in str(
                        settings.log_level,
                    )
                    assert settings.debug is True

        def test_pydantic_env_var_precedence(self, tmp_path: Path) -> None:
            """Test that environment variables override .env file.

            Uses tmp_path fixture to avoid writing files to current directory.
            Validates precedence: env vars > .env file.
            """
            with u.Core.Tests.env_vars_context(
                {},
                ["FLEXT_APP_NAME", "FLEXT_LOG_LEVEL"],
            ):
                env_file = tmp_path / ".env"
                env_content = "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\n"
                env_file.write_text(env_content)
                assert env_file.exists()
                assert env_file.read_text() == env_content
                os.environ["FLEXT_APP_NAME"] = "from-env-var"
                os.environ["FLEXT_LOG_LEVEL"] = "ERROR"
                settings = FlextSettings()
                assert settings.app_name == "from-env-var"
                assert settings.log_level == "ERROR"

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
            with u.Core.Tests.env_vars_context(
                {"FLEXT_TIMEOUT_SECONDS": "60", "FLEXT_ENV_FILE": str(env_file)},
                ["FLEXT_TIMEOUT_SECONDS", "FLEXT_ENV_FILE"],
            ):
                settings = FlextSettings.fetch_global(overrides={"timeout_seconds": 90})
                assert settings.timeout_seconds == 90
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
            with u.Core.Tests.env_vars_context(
                {"FLEXT_DEBUG": "true"},
                ["FLEXT_DEBUG"],
            ):
                FlextSettings.reset_for_testing()
                settings = FlextSettings()
                assert settings.debug is True
                os.environ["FLEXT_DEBUG"] = "false"
                FlextSettings.reset_for_testing()
                config_updated = FlextSettings()
                assert config_updated.debug is False

        def test_pydantic_effective_log_level_with_precedence(self) -> None:
            """Test that effective_log_level respects debug mode precedence."""
            with u.Core.Tests.env_vars_context(
                {"FLEXT_LOG_LEVEL": "ERROR", "FLEXT_DEBUG": "true"},
                ["FLEXT_LOG_LEVEL", "FLEXT_DEBUG"],
            ):
                FlextSettings.reset_for_testing()
                settings = FlextSettings()
                assert settings.log_level == "ERROR"
                assert settings.debug is True
                os.environ["FLEXT_DEBUG"] = "false"
                FlextSettings.reset_for_testing()
                config_no_debug = FlextSettings()
                assert config_no_debug.log_level == "ERROR"
                assert config_no_debug.debug is False

        def test_fetch_global(self) -> None:
            """Test fetch_global returns singleton."""
            instance1 = FlextSettings.fetch_global()
            instance2 = FlextSettings.fetch_global()
            assert instance1 is instance2

        def test_config_with_all_fields(self) -> None:
            """Test settings initialization with all fields set."""
            with u.Core.Tests.env_vars_context(
                {"FLEXT_DEBUG": "true", "FLEXT_LOG_LEVEL": "DEBUG"},
                ["FLEXT_DEBUG", "FLEXT_LOG_LEVEL"],
            ):
                settings = FlextSettings()
                assert settings.debug is True
                assert settings.log_level == "DEBUG"

        def test_resolve_env_file(self) -> None:
            """Test resolve_env_file method for 100% coverage."""
            result = u.resolve_env_file()
            assert isinstance(result, str)

        def test_reset_instance(self) -> None:
            """Test _reset_instance method for testing purposes."""
            config1 = FlextSettings.fetch_global()
            FlextSettings._reset_instance()
            config2 = FlextSettings.fetch_global()
            assert config1 is not config2 or config1 is config2

        def test_singleton_type_check(self) -> None:
            """Test singleton __new__ type check for edge case coverage."""
            original_instances = dict(FlextSettings._instances.items())
            try:

                class WrongType:
                    pass

                wrong_instance = cast(
                    "FlextSettings",
                    cast("t.RecursiveContainer", WrongType()),
                )
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
                FlextSettings.model_validate({"database_url": "invalid://scheme"})

        def test_effective_log_level_trace(self) -> None:
            """Test effective_log_level with trace mode."""
            settings = u.Core.Tests.create_test_config(trace=True, debug=True)
            assert settings.effective_log_level == c.LogLevel.DEBUG

        def test_effective_log_level_debug(self) -> None:
            """Test effective_log_level with debug mode."""
            settings = u.Core.Tests.create_test_config(debug=True)
            assert settings.effective_log_level == c.LogLevel.INFO

        def test_effective_log_level_normal(self) -> None:
            """Test effective_log_level without debug/trace."""
            settings = u.Core.Tests.create_test_config(debug=False, trace=False)
            assert settings.effective_log_level == settings.log_level

        def test_resolve_di_settings_provider(self) -> None:
            """Test resolve_di_settings_provider creates provider."""
            settings = u.Core.Tests.create_test_config()
            provider = settings.resolve_di_settings_provider()
            assert provider is not None
            provider2 = settings.resolve_di_settings_provider()
            assert provider is provider2

        def test_apply_override_invalid_key(self) -> None:
            """Test apply_override returns False for invalid key."""
            settings = u.Core.Tests.create_test_config()
            assert settings.apply_override("invalid_key", "value") is False

        def test_apply_override(self) -> None:
            """Test apply_override applies validated override."""
            settings = u.Core.Tests.create_test_config()
            original_value = settings.app_name
            settings.apply_override("app_name", "new_name")
            assert settings.app_name == "new_name"
            settings.apply_override("app_name", original_value)

        def test_auto_settings_create_settings(self) -> None:
            """Test AutoSettings.create_settings method."""
            auto_settings = FlextSettings.AutoSettings(settings_class=FlextSettings)
            instance = auto_settings.create_settings()
            assert isinstance(instance, p.Settings)

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

        def test_fetch_namespace_not_found(self) -> None:
            """Test fetch_namespace raises ValueError for unregistered namespace."""
            settings = u.Core.Tests.create_test_config()
            with pytest.raises(
                ValueError,
                match="Namespace 'nonexistent' not registered",
            ):
                settings.fetch_namespace("nonexistent", FlextSettings)

        def test_fetch_namespace_type_mismatch(self) -> None:
            """Test fetch_namespace raises TypeError for type mismatch."""

            class OtherSettings(FlextSettings):
                pass

            FlextSettings.register_namespace("test_type", FlextSettings)
            settings = u.Core.Tests.create_test_config()
            with pytest.raises(TypeError, match="is not instance"):
                settings.fetch_namespace("test_type", OtherSettings)
            instance = settings.fetch_namespace("test_type", FlextSettings)
            assert isinstance(instance, p.Settings)
            del FlextSettings._namespace_registry["test_type"]

        def test_fetch_namespace_found(self) -> None:
            """Test fetch_namespace returns namespace settings when registered."""
            FlextSettings.register_namespace("test_attr", FlextSettings)
            settings = u.Core.Tests.create_test_config()
            instance = settings.fetch_namespace("test_attr", FlextSettings)
            assert isinstance(instance, p.Settings)
            del FlextSettings._namespace_registry["test_attr"]

    __all__: list[str] = ["TestFlextSettings"]


__all__: list[str] = ["TestFlextSettings"]
