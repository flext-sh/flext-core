"""FlextConfig comprehensive functionality tests.

Module: flext_core.config
Scope: FlextConfig class - configuration management, validation, environment handling,
thread safety, namespace management, and Pydantic integration.

Tests core FlextConfig functionality including:
- Configuration initialization and validation
- Environment variable handling
- Thread safety and singleton patterns
- Namespace management and auto-registration
- Pydantic model integration and serialization

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from flext_core import FlextConfig, FlextConstants


class ConfigTestHelpers:
    """Generalized helpers for config testing to reduce duplication."""

    @staticmethod
    @contextmanager
    def env_vars_context(
        vars_to_set: dict[str, str], vars_to_clear: list[str] | None = None
    ) -> Generator[None]:
        """Context manager for managing environment variables."""
        saved: dict[str, str | None] = {}
        vars_to_clear = vars_to_clear or []

        try:
            # Save and clear
            for key in vars_to_clear:
                saved[key] = os.environ.pop(key, None)

            # Save and set
            for key, value in vars_to_set.items():
                saved[key] = os.environ.get(key)
                os.environ[key] = value

            yield
        finally:
            # Restore
            for key, saved_value in saved.items():
                if saved_value is not None:
                    os.environ[key] = saved_value
                elif key in os.environ:
                    del os.environ[key]

    @staticmethod
    def create_test_config(**kwargs: object) -> FlextConfig:
        """Create test config with reset."""
        FlextConfig.reset_global_instance()
        try:
            return FlextConfig(**kwargs)
        except Exception:
            FlextConfig.reset_global_instance()
            raise

    @staticmethod
    def assert_config_fields(config: FlextConfig, expected: dict[str, object]) -> None:
        """Assert config fields match expected values."""
        for key, value in expected.items():
            assert getattr(config, key) == value


class ConfigScenarios:
    """Centralized config test scenarios using FlextConstants."""

    INIT_CASES: ClassVar[list[dict[str, object]]] = [
        {"app_name": "test_app", "version": "1.0.0", "debug": True},
        {"app_name": "dict_app", "version": "2.0.0", "debug": False},
        {"app_name": "valid_app", "version": "1.0.0"},
    ]

    FIELD_ACCESS_CASES: ClassVar[list[tuple[str, object, object]]] = [
        ("app_name", "test_value", "modified_value"),
        ("version", "1.0.0", "2.0.0"),
    ]

    DEBUG_TRACE_CASES: ClassVar[list[dict[str, object]]] = [
        {"debug": True, "trace": False},
        {"debug": True, "trace": True},
        {"debug": False, "trace": False},
    ]

    LOG_LEVEL_CASES: ClassVar[list[tuple[str, bool, bool]]] = [
        (FlextConstants.Settings.LogLevel.INFO, False, False),
        (FlextConstants.Settings.LogLevel.INFO, True, False),
        (FlextConstants.Settings.LogLevel.INFO, True, True),
    ]

    ENV_PREFIX_CASES: ClassVar[list[tuple[str, str, bool, str]]] = [
        ("DEBUG", "true", False, "INFO"),
        # When FLEXT_DEBUG=true, log_level defaults to INFO (not ERROR)
        ("FLEXT_DEBUG", "true", True, "INFO"),
    ]

    VALIDATION_ERROR_CASES: ClassVar[list[tuple[dict[str, object], str]]] = [
        ({"log_level": "INVALID"}, "log_level"),
        ({"trace": True, "debug": False}, "Trace mode requires debug mode"),
    ]


class TestFlextConfig:
    """Test suite for FlextConfig using FlextTestsUtilities and FlextConstants."""

    @pytest.mark.parametrize(
        "config_data",
        ConfigScenarios.INIT_CASES,
        ids=lambda d: str(d.get("app_name", "default")),
    )
    def test_config_initialization(self, config_data: dict[str, object]) -> None:
        """Test config initialization with various values."""
        config = ConfigTestHelpers.create_test_config(**config_data)
        ConfigTestHelpers.assert_config_fields(config, config_data)
        assert isinstance(config, FlextConfig)

    def test_config_from_dict(self) -> None:
        """Test config creation from dictionary."""
        config_data: dict[str, object] = {
            "app_name": "dict_app",
            "version": "2.0.0",
            "debug": False,
        }
        config = ConfigTestHelpers.create_test_config(**config_data)
        ConfigTestHelpers.assert_config_fields(config, config_data)

    def test_config_to_dict(self) -> None:
        """Test config conversion to dictionary."""
        config = ConfigTestHelpers.create_test_config(
            app_name="test_app", version="1.0.0", debug=True
        )
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "test_app"
        assert config_dict["version"] == "1.0.0"
        assert config_dict["debug"] is True

    def test_config_clone(self) -> None:
        """Test config cloning with singleton pattern."""
        original_config = ConfigTestHelpers.create_test_config(
            app_name="original_app", version="1.0.0"
        )
        config_dict = original_config.model_dump()
        cloned_config = FlextConfig.model_validate(config_dict)
        assert cloned_config.app_name == original_config.app_name
        assert cloned_config.version == original_config.version
        assert cloned_config is original_config

    @pytest.mark.parametrize(
        ("field_name", "value", "modified"), ConfigScenarios.FIELD_ACCESS_CASES
    )
    def test_config_field_access(
        self, field_name: str, value: object, modified: object
    ) -> None:
        """Test config field access operations."""
        config = ConfigTestHelpers.create_test_config()
        setattr(config, field_name, value)
        assert getattr(config, field_name) == value
        setattr(config, field_name, modified)
        assert getattr(config, field_name) == modified

    def test_config_field_reset(self) -> None:
        """Test config field reset operation."""
        config = ConfigTestHelpers.create_test_config()
        config.app_name = "value1"
        config.version = "2.0.0"
        assert config.app_name == "value1"
        assert config.version == "2.0.0"
        config.app_name = FlextConfig.model_fields["app_name"].default
        config.version = FlextConfig.model_fields["version"].default
        assert config.app_name != "value1"
        assert config.version != "2.0.0"

    def test_config_keys_values_items(self) -> None:
        """Test config keys, values, and items operations."""
        config = ConfigTestHelpers.create_test_config()
        config.app_name = "value1"
        config.version = "2.0.0"
        config_dict = config.model_dump()
        assert "app_name" in config_dict
        assert "version" in config_dict
        assert "value1" in config_dict.values()
        assert "2.0.0" in config_dict.values()
        assert ("app_name", "value1") in config_dict.items()

    def test_config_singleton_pattern(self) -> None:
        """Test config implements true singleton pattern."""
        config1 = FlextConfig()
        config2 = FlextConfig()
        assert config1 is config2
        assert config1.model_dump() == config2.model_dump()

    def test_config_thread_safety(self) -> None:
        """Test config thread safety."""
        config = ConfigTestHelpers.create_test_config()
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
        config = ConfigTestHelpers.create_test_config()
        start_time = time.time()
        for i in range(100):
            config.app_name = f"value_{i}"
            _ = config.app_name
        assert time.time() - start_time < 5.0

    def test_config_serialization(self) -> None:
        """Test config serialization."""
        config = ConfigTestHelpers.create_test_config(
            app_name="serialize_app", version="1.0.0"
        )
        json_str = config.model_dump_json(
            exclude={
                "is_debug_enabled",
                "effective_log_level",
                "is_production",
                "effective_timeout",
                "has_database",
                "has_cache",
            }
        )
        assert isinstance(json_str, str) and "serialize_app" in json_str
        restored_config = FlextConfig.model_validate_json(json_str)
        assert restored_config.app_name == config.app_name
        assert restored_config.version == config.version

    @pytest.mark.parametrize(
        ("config_data", "error_pattern"), ConfigScenarios.VALIDATION_ERROR_CASES
    )
    def test_config_validation_errors(
        self, config_data: dict[str, object], error_pattern: str
    ) -> None:
        """Test config validation with invalid inputs."""
        with pytest.raises(ValidationError) as exc_info:
            FlextConfig.model_validate(config_data)
        assert error_pattern in str(exc_info.value)

    def test_config_create_and_configure_pattern(self) -> None:
        """Test direct instantiation and configuration pattern."""
        FlextConfig.reset_global_instance()
        try:
            config = ConfigTestHelpers.create_test_config(
                app_name="Test Application", debug=True
            )
            assert config.app_name == "Test Application"
            assert config.debug is True
            config2 = ConfigTestHelpers.create_test_config(
                app_name="Test Application", debug=True
            )
            assert config2.app_name == "Test Application"
            assert config2.debug is True
            assert config is config2
        finally:
            FlextConfig.reset_global_instance()

    @pytest.mark.parametrize(
        "debug_trace",
        ConfigScenarios.DEBUG_TRACE_CASES,
        ids=lambda d: f"debug_{d.get('debug')}_trace_{d.get('trace', False)}",
    )
    def test_config_debug_enabled(self, debug_trace: dict[str, object]) -> None:
        """Test debug enabled checking using direct fields."""
        config = ConfigTestHelpers.create_test_config(**debug_trace)
        assert config.debug == debug_trace["debug"]
        if "trace" in debug_trace:
            assert config.trace == debug_trace["trace"]

    @pytest.mark.parametrize(
        ("log_level", "debug", "trace"), ConfigScenarios.LOG_LEVEL_CASES
    )
    def test_config_effective_log_level(
        self, log_level: str, debug: bool, trace: bool
    ) -> None:
        """Test effective log level using direct fields."""
        config = ConfigTestHelpers.create_test_config(
            log_level=log_level, debug=debug, trace=trace
        )
        assert config.log_level == log_level
        assert config.debug == debug
        if trace:
            assert config.trace == trace

    def test_global_instance_management(self) -> None:
        """Test global instance management methods with singleton pattern."""
        original_instance = FlextConfig.get_global_instance()
        try:
            assert FlextConfig.get_global_instance() is original_instance
            FlextConfig.reset_global_instance()
            fresh_config = FlextConfig()
            assert fresh_config is not original_instance
            assert fresh_config.app_name == "flext"
            assert FlextConfig.get_global_instance() is fresh_config
        finally:
            FlextConfig.reset_global_instance()

    @pytest.mark.parametrize(
        ("env_key", "env_value", "should_load", "log_level"),
        ConfigScenarios.ENV_PREFIX_CASES,
    )
    def test_pydantic_env_prefix(
        self, env_key: str, env_value: str, should_load: bool, log_level: str
    ) -> None:
        """Test that FlextConfig uses FLEXT_ prefix for environment variables."""
        with ConfigTestHelpers.env_vars_context(
            {env_key: env_value},
            ["DEBUG", "LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_LOG_LEVEL"],
        ):
            if not env_key.startswith(FlextConstants.Platform.ENV_PREFIX):
                os.environ["DEBUG"] = "true"
                os.environ["LOG_LEVEL"] = "ERROR"
            config = FlextConfig()
            assert config.debug == should_load
            if should_load:
                assert config.log_level == log_level

    def test_pydantic_dotenv_file_loading(self, tmp_path: Path) -> None:
        """Test that FlextConfig automatically loads .env file."""
        original_dir = Path.cwd()
        with ConfigTestHelpers.env_vars_context(
            {}, ["FLEXT_LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_APP_NAME"]
        ):
            try:
                os.chdir(tmp_path)
                env_file = tmp_path / ".env"
                env_file.write_text(
                    "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n"
                )
                if hasattr(FlextConfig, "_instances"):
                    FlextConfig._instances.clear()
                config = FlextConfig()
                assert config.app_name == "from-dotenv"
                assert str(config.log_level) == "WARNING" or "WARNING" in str(
                    config.log_level
                )
                assert config.debug is True
            finally:
                os.chdir(original_dir)

    def test_pydantic_env_var_precedence(self, tmp_path: Path) -> None:
        """Test that environment variables override .env file."""
        original_dir = Path.cwd()
        with ConfigTestHelpers.env_vars_context(
            {}, ["FLEXT_APP_NAME", "FLEXT_LOG_LEVEL"]
        ):
            try:
                os.chdir(tmp_path)
                env_file = tmp_path / ".env"
                env_file.write_text(
                    "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\n"
                )
                os.environ["FLEXT_APP_NAME"] = "from-env-var"
                os.environ["FLEXT_LOG_LEVEL"] = "ERROR"
                config = FlextConfig()
                assert config.app_name == "from-env-var"
                assert config.log_level == "ERROR"
            finally:
                os.chdir(original_dir)

    def test_pydantic_complete_precedence_chain(self, tmp_path: Path) -> None:
        """Test complete Pydantic 2 Settings precedence chain."""
        original_dir = Path.cwd()
        with ConfigTestHelpers.env_vars_context(
            {"FLEXT_TIMEOUT_SECONDS": "60"}, ["FLEXT_TIMEOUT_SECONDS"]
        ):
            try:
                os.chdir(tmp_path)
                env_file = tmp_path / ".env"
                env_file.write_text("FLEXT_TIMEOUT_SECONDS=45\n")
                config = ConfigTestHelpers.create_test_config(timeout_seconds=90)
                assert config.timeout_seconds == 90
                config_no_explicit = FlextConfig()
                assert config_no_explicit.timeout_seconds == 60
                del os.environ["FLEXT_TIMEOUT_SECONDS"]
                config_no_env = FlextConfig()
                assert config_no_env.timeout_seconds == 45
            finally:
                os.chdir(original_dir)

    def test_pydantic_env_var_naming(self) -> None:
        """Test that environment variables follow correct naming convention."""
        with ConfigTestHelpers.env_vars_context(
            {"FLEXT_DEBUG": "true"}, ["FLEXT_DEBUG"]
        ):
            config = FlextConfig()
            assert config.debug is True
            os.environ["FLEXT_DEBUG"] = "false"
            config_updated = FlextConfig()
            assert config_updated.debug is False

    def test_pydantic_effective_log_level_with_precedence(self) -> None:
        """Test that effective_log_level respects debug mode precedence."""
        with ConfigTestHelpers.env_vars_context(
            {"FLEXT_LOG_LEVEL": "ERROR", "FLEXT_DEBUG": "true"},
            ["FLEXT_LOG_LEVEL", "FLEXT_DEBUG"],
        ):
            config = FlextConfig()
            assert config.log_level == "ERROR"
            assert config.debug is True
            os.environ["FLEXT_DEBUG"] = "false"
            config_no_debug = FlextConfig()
            assert config_no_debug.log_level == "ERROR"
            assert config_no_debug.debug is False

    def test_get_global_instance(self) -> None:
        """Test get_global_instance returns singleton."""
        instance1 = FlextConfig.get_global_instance()
        instance2 = FlextConfig.get_global_instance()
        assert instance1 is instance2

    def test_config_with_all_fields(self) -> None:
        """Test config initialization with all fields set."""
        with ConfigTestHelpers.env_vars_context(
            {"FLEXT_DEBUG": "true", "FLEXT_LOG_LEVEL": "DEBUG"},
            ["FLEXT_DEBUG", "FLEXT_LOG_LEVEL"],
        ):
            config = FlextConfig()
            assert config.debug is True
            assert config.log_level == "DEBUG"


__all__ = ["TestFlextConfig"]
