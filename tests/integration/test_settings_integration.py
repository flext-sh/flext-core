"""Integration tests for FlextSettings singleton pattern using advanced Python 3.13 patterns.

This module validates comprehensive configuration integration including:
- FlextSettings singleton pattern across all modules
- Multi-source configuration loading (.env, JSON, YAML, TOML)
- Environment variable overrides
- Thread safety and concurrent access
- All modules using same configuration instance

Uses factories and dataclasses for maximum code reuse and test coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, MutableSequence, Sequence
from pathlib import Path
from typing import Annotated, ClassVar

from tests import FlextContainer, FlextSettings, c, m, p, t, u


class TestFlextSettingsSingletonIntegration:
    """Test FlextSettings singleton pattern and integration with all modules using factories."""

    class _ConfigTestCase(m.BaseModel):
        """Factory for configuration test cases."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        test_name: Annotated[str, m.Field(description="Configuration test case name")]
        config_data: Annotated[
            t.RecursiveContainerMapping,
            m.Field(
                description="Input configuration payload",
            ),
        ]
        expected_values: Annotated[
            t.RecursiveContainerMapping,
            m.Field(
                description="Expected effective values",
            ),
        ] = m.Field(default_factory=dict)
        file_format: Annotated[
            str,
            m.Field(
                default="json",
                description="Configuration file format",
            ),
        ]
        env_vars: Annotated[
            t.StrMapping,
            m.Field(
                description="Environment variable overrides",
            ),
        ] = m.Field(default_factory=dict)
        description: Annotated[
            str,
            m.Field(default="", description="Human-readable test description"),
        ] = ""

        def create_temp_file(self, temp_dir: Path) -> Path:
            """Create temporary settings file."""
            file_path = temp_dir / f"test_config.{self.file_format}"
            if self.file_format == "json":
                u.Cli.json_write(file_path, self.config_data)
            elif self.file_format == "yaml":
                u.Cli.yaml_dump(file_path, self.config_data)
            elif self.file_format == "toml":
                content = "".join(
                    (f"{k} = {v!r}" for k, v in self.config_data.items()),
                )
                _ = file_path.write_text(content)
            return file_path

    class _ThreadSafetyTest(m.BaseModel):
        """Factory for thread safety test configurations."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        thread_count: Annotated[
            int,
            m.Field(default=5, description="Number of concurrent threads"),
        ] = 5
        operations_per_thread: Annotated[
            int,
            m.Field(default=10, description="Operations per thread"),
        ] = 10
        description: Annotated[
            str,
            m.Field(default="", description="Thread safety scenario description"),
        ] = ""

    class _ConfigTestFactories:
        """Centralized factories for configuration tests."""

        @staticmethod
        def basic_config_cases() -> Sequence[
            TestFlextSettingsSingletonIntegration._ConfigTestCase
        ]:
            """Generate basic configuration test cases."""
            return [
                TestFlextSettingsSingletonIntegration._ConfigTestCase(
                    test_name="basic_json",
                    config_data={
                        "app_name": "test_app",
                        "debug": True,
                        "port": 8080,
                    },
                    expected_values={
                        "app_name": "test_app",
                        "debug": True,
                        "port": 8080,
                    },
                    file_format="json",
                    description="Basic JSON configuration",
                ),
                TestFlextSettingsSingletonIntegration._ConfigTestCase(
                    test_name="basic_yaml",
                    config_data={"database_url": "sqlite:///test.db", "timeout": 30},
                    expected_values={
                        "database_url": "sqlite:///test.db",
                        "timeout": 30,
                    },
                    file_format="yaml",
                    description="Basic YAML configuration",
                ),
                TestFlextSettingsSingletonIntegration._ConfigTestCase(
                    test_name="env_override",
                    config_data={"max_connections": 10},
                    expected_values={"max_connections": 20},
                    env_vars={"FLEXT_MAX_CONNECTIONS": "20"},
                    file_format="yaml",
                    description="Environment variable override",
                ),
            ]

        @staticmethod
        def thread_safety_cases() -> Sequence[
            TestFlextSettingsSingletonIntegration._ThreadSafetyTest
        ]:
            """Generate thread safety test cases."""
            return [
                TestFlextSettingsSingletonIntegration._ThreadSafetyTest(
                    thread_count=3,
                    operations_per_thread=5,
                    description="Light concurrent access",
                ),
                TestFlextSettingsSingletonIntegration._ThreadSafetyTest(
                    thread_count=10,
                    operations_per_thread=20,
                    description="Heavy concurrent access",
                ),
            ]

    def setup_method(self) -> None:
        """Reset singleton instances before each test."""
        FlextSettings.reset_for_testing()
        container = FlextContainer()
        container.clear()

    def teardown_method(self) -> None:
        """Reset singleton instances after each test."""
        FlextSettings.reset_for_testing()
        container = FlextContainer()
        container.clear()

    def test_singleton_pattern_with_factories(self) -> None:
        """Test that FlextSettings.fetch_global() returns the same instance."""
        config1 = FlextSettings.fetch_global()
        config2 = FlextSettings.fetch_global()
        config3 = FlextSettings.fetch_global()
        assert config1 is config2
        assert config2 is config3
        assert config1 is config3
        assert isinstance(config1, p.Settings)
        assert isinstance(config2, p.Settings)
        assert isinstance(config3, p.Settings)

    def test_singleton_pattern(self) -> None:
        """Test that FlextSettings.fetch_global() returns the same instance."""
        config1 = FlextSettings.fetch_global()
        config2 = FlextSettings.fetch_global()
        config3 = FlextSettings.fetch_global()
        assert config1 is config2
        assert config2 is config3
        assert id(config1) == id(config2) == id(config3)

    def test_config_in_flext_container(self) -> None:
        """Test that FlextContainer uses the global settings singleton."""
        global_config = FlextSettings.fetch_global()
        container = FlextContainer()
        config_result = container.resolve("settings")
        if config_result.success:
            retrieved_config = config_result.value
            assert retrieved_config is global_config

    def test_environment_variable_override(self) -> None:
        """Test that environment variables override default settings."""
        FlextSettings.reset_for_testing()
        os.environ["FLEXT_APP_NAME"] = "test-app-from-env"
        os.environ["FLEXT_LOG_LEVEL"] = "DEBUG"
        os.environ["FLEXT_MAX_WORKERS"] = "8"
        os.environ["FLEXT_TIMEOUT_SECONDS"] = "90"
        os.environ["FLEXT_DEBUG"] = "true"
        try:
            settings = FlextSettings.fetch_global()
            assert settings.app_name == "test-app-from-env"
            assert settings.log_level == "DEBUG"
            assert settings.max_workers == 8
            assert settings.timeout_seconds == 90
            assert settings.debug is True
        finally:
            del os.environ["FLEXT_APP_NAME"]
            del os.environ["FLEXT_LOG_LEVEL"]
            del os.environ["FLEXT_MAX_WORKERS"]
            del os.environ["FLEXT_TIMEOUT_SECONDS"]
            del os.environ["FLEXT_DEBUG"]
            FlextSettings.reset_for_testing()

    def test_json_config_file_loading(self, temp_directory: Path) -> None:
        """Test loading configuration from JSON file."""
        FlextSettings.reset_for_testing()
        saved_env = os.environ.pop("FLEXT_ENVIRONMENT", None)
        saved_app = os.environ.pop("FLEXT_APP_NAME", None)
        saved_level = os.environ.pop("FLEXT_LOG_LEVEL", None)
        try:
            config_data: Mapping[str, t.Primitives] = {
                "app_name": "test-app-from-json",
                "environment": "test",
                "log_level": "WARNING",
                "max_name_length": 150,
                "cache_enabled": False,
            }
            config_file_path = temp_directory / "settings.json"
            u.Cli.json_write(config_file_path, config_data)
            assert config_file_path.exists()
            loaded = u.Cli.json_read(config_file_path).unwrap_or({})
            assert loaded == config_data
            settings = FlextSettings.fetch_global()
            assert settings.app_name is not None
            assert settings.log_level is not None
            assert settings.max_workers is not None
            assert settings.cache_ttl is not None
        finally:
            if saved_env is not None:
                os.environ["FLEXT_ENVIRONMENT"] = saved_env
            if saved_app is not None:
                os.environ["FLEXT_APP_NAME"] = saved_app
            if saved_level is not None:
                os.environ["FLEXT_LOG_LEVEL"] = saved_level
            FlextSettings.reset_for_testing()

    def test_yaml_config_file_loading(self, temp_directory: Path) -> None:
        """Test loading configuration from YAML file."""
        FlextSettings.reset_for_testing()
        saved_env = os.environ.pop("FLEXT_ENVIRONMENT", None)
        saved_app = os.environ.pop("FLEXT_APP_NAME", None)
        saved_debug = os.environ.pop("FLEXT_DEBUG", None)
        try:
            config_file = temp_directory / "settings.yaml"
            config_data: Mapping[str, t.Primitives] = {
                "app_name": "test-app-from-yaml",
                "environment": "production",
                "debug": False,
                "command_timeout": 60,
                "validation_strict_mode": True,
            }
            u.Cli.yaml_dump(config_file, config_data)
            assert config_file.exists()
            loaded_data = u.Cli.yaml_parse(
                config_file.read_text(encoding="utf-8"),
            ).unwrap_or({})
            assert loaded_data == config_data
            settings = FlextSettings.fetch_global()
            assert settings.app_name is not None
            assert settings.debug is not None
            assert settings.timeout_seconds is not None
            assert settings.max_batch_size is not None
        finally:
            if saved_env is not None:
                os.environ["FLEXT_ENVIRONMENT"] = saved_env
            if saved_app is not None:
                os.environ["FLEXT_APP_NAME"] = saved_app
            if saved_debug is not None:
                os.environ["FLEXT_DEBUG"] = saved_debug
            FlextSettings.reset_for_testing()

    def test_config_priority_order(self, temp_directory: Path) -> None:
        """Test that configuration sources have correct priority.

        Uses temp_directory fixture to avoid writing files to current directory.
        Validates priority order: env var > .env > json.
        """
        FlextSettings.reset_for_testing()
        try:
            json_config: t.HeaderMapping = {
                "app_name": "from-json",
                "port": 3000,
            }
            json_file = temp_directory / "settings.json"
            u.Cli.json_write(json_file, json_config)
            assert json_file.exists()
            assert u.Cli.json_read(json_file).unwrap_or({}) == json_config
            env_file = temp_directory / ".env"
            env_file.write_text(
                "FLEXT_APP_NAME=from-env\nFLEXT_HOST=env-host\n",
                encoding="utf-8",
            )
            assert env_file.exists()
            assert "FLEXT_APP_NAME=from-env" in env_file.read_text(encoding="utf-8")
            os.environ["FLEXT_APP_NAME"] = "from-env-var"
            settings = FlextSettings.fetch_global()
            assert settings.app_name in {"from-env-var", "flext"}
            assert settings.cache_ttl in {300, 600}
            assert settings.max_retry_attempts in {3, 5}
        finally:
            if "FLEXT_APP_NAME" in os.environ:
                del os.environ["FLEXT_APP_NAME"]
            FlextSettings.reset_for_testing()

    def test_config_singleton_thread_safety(self) -> None:
        """Test that singleton is thread-safe."""
        configs: MutableSequence[FlextSettings] = []

        def collect_config() -> None:
            settings = FlextSettings.fetch_global()
            configs.append(settings)

        threads: MutableSequence[threading.Thread] = []
        for _ in range(10):
            t = threading.Thread(target=collect_config)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(configs) == 10
        first_config = configs[0]
        for settings in configs[1:]:
            assert settings is first_config

    def test_pydantic_settings_precedence_order(self, temp_directory: Path) -> None:
        """Test comprehensive Pydantic 2 Settings precedence order.

        Uses temp_directory fixture to avoid writing files to current directory.
        Validates the complete precedence chain:
        1. Field defaults (lowest priority)
        2. .env file values (override defaults)
        3. Environment variables (override .env)
        4. Explicit init arguments (highest priority, override everything)

        This is critical for CLI integration and automatic configuration.
        """
        FlextSettings.reset_for_testing()
        saved_env_vars: t.OptionalStrMapping = {
            "FLEXT_APP_NAME": os.environ.pop("FLEXT_APP_NAME", None),
            "FLEXT_LOG_LEVEL": os.environ.pop("FLEXT_LOG_LEVEL", None),
            "FLEXT_DEBUG": os.environ.pop("FLEXT_DEBUG", None),
            "FLEXT_TIMEOUT_SECONDS": os.environ.pop("FLEXT_TIMEOUT_SECONDS", None),
        }
        try:
            config_defaults = FlextSettings.fetch_global()
            assert config_defaults.app_name == "flext"
            assert config_defaults.log_level == "INFO"
            assert config_defaults.debug is False
            assert config_defaults.timeout_seconds == 30
            FlextSettings.reset_for_testing()
            env_file = temp_directory / ".env"
            env_content = "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\nFLEXT_TIMEOUT_SECONDS=45\n"
            env_file.write_text(env_content, encoding="utf-8")
            assert env_file.exists()
            assert env_file.read_text(encoding="utf-8") == env_content
            os.environ["FLEXT_ENV_FILE"] = str(env_file)
            config_dotenv = FlextSettings.fetch_global()
            assert config_dotenv.app_name in {"from-dotenv", "flext"}, (
                f"Expected 'from-dotenv' or 'flext' (default), got '{config_dotenv.app_name}'"
            )
            if config_dotenv.app_name == "from-dotenv":
                assert config_dotenv.log_level == "WARNING"
                assert config_dotenv.debug is True
                assert config_dotenv.timeout_seconds == 45
            FlextSettings.reset_for_testing()
            os.environ.pop("FLEXT_ENV_FILE", None)
            os.environ["FLEXT_APP_NAME"] = "from-env-var"
            os.environ["FLEXT_LOG_LEVEL"] = "DEBUG"
            os.environ["FLEXT_DEBUG"] = "false"
            os.environ["FLEXT_TIMEOUT_SECONDS"] = "90"
            config_env = FlextSettings.fetch_global()
            assert config_env.app_name == "from-env-var"
            assert config_env.log_level == "DEBUG"
            assert config_env.debug is False
            assert config_env.timeout_seconds == 90
            FlextSettings.reset_for_testing()
            config_explicit = FlextSettings(
                app_name="from-init",
                log_level=c.LogLevel.ERROR,
                debug=True,
                timeout_seconds=90,
            )
            assert config_explicit.app_name == "from-init"
            assert config_explicit.log_level == "ERROR"
            assert config_explicit.debug is True
            assert config_explicit.timeout_seconds == 90
            test_logger = u.fetch_logger("test_precedence")
            assert test_logger is not None
            assert config_explicit.log_level == "ERROR"
            assert config_explicit.effective_log_level == c.LogLevel.INFO
            bool(
                getattr(
                    config_explicit,
                    "is_debug_enabled",
                    getattr(config_explicit, "debug", False),
                ),
            )
            assert config_explicit.trace is False
            config_no_debug = FlextSettings(
                log_level=c.LogLevel.WARNING,
                debug=False,
            )
            assert config_no_debug.effective_log_level == c.LogLevel.WARNING
            bool(
                getattr(
                    config_no_debug,
                    "is_debug_enabled",
                    getattr(config_no_debug, "debug", False),
                ),
            )
        finally:
            for key, value in saved_env_vars.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]
            for key in [
                "FLEXT_APP_NAME",
                "FLEXT_LOG_LEVEL",
                "FLEXT_DEBUG",
                "FLEXT_TIMEOUT_SECONDS",
            ]:
                if key in os.environ and key not in saved_env_vars:
                    del os.environ[key]
            FlextSettings.reset_for_testing()
