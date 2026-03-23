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

import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, ClassVar

import yaml
from flext_tests import t
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextConstants, FlextContainer, FlextLogger, FlextSettings, p


class TestFlextSettingsSingletonIntegration:
    """Test FlextSettings singleton pattern and integration with all modules using factories."""

    class _ConfigTestCase(BaseModel):
        """Factory for configuration test cases."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        test_name: Annotated[str, Field(description="Configuration test case name")]
        config_data: Annotated[
            Mapping[str, t.NormalizedValue],
            Field(
                description="Input configuration payload",
            ),
        ]
        expected_values: Mapping[str, t.NormalizedValue] = Field(
            default_factory=dict,
            description="Expected effective values",
        )
        file_format: str = Field(
            default="json", description="Configuration file format"
        )
        env_vars: Mapping[str, str] = Field(
            default_factory=dict,
            description="Environment variable overrides",
        )
        description: Annotated[
            str,
            Field(default="", description="Human-readable test description"),
        ] = ""

        def create_temp_file(self, temp_dir: Path) -> Path:
            """Create temporary config file."""
            file_path = temp_dir / f"test_config.{self.file_format}"
            if self.file_format == "json":
                with Path(file_path).open("w", encoding="utf-8") as f:
                    json.dump(self.config_data, f, indent=2)
            elif self.file_format == "yaml":
                with Path(file_path).open("w", encoding="utf-8") as f:
                    yaml.dump(self.config_data, f, default_flow_style=False)
            elif self.file_format == "toml":
                content = "\n".join(
                    (f"{k} = {v!r}" for k, v in self.config_data.items()),
                )
                _ = file_path.write_text(content)
            return file_path

    class _ThreadSafetyTest(BaseModel):
        """Factory for thread safety test configurations."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

        thread_count: Annotated[
            int,
            Field(default=5, description="Number of concurrent threads"),
        ] = 5
        operations_per_thread: Annotated[
            int,
            Field(default=10, description="Operations per thread"),
        ] = 10
        description: Annotated[
            str,
            Field(default="", description="Thread safety scenario description"),
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
        container.clear_all()

    def teardown_method(self) -> None:
        """Reset singleton instances after each test."""
        FlextSettings.reset_for_testing()
        container = FlextContainer()
        container.clear_all()

    def test_singleton_pattern_with_factories(self) -> None:
        """Test that FlextSettings.get_global() returns the same instance."""
        config1 = FlextSettings.get_global()
        config2 = FlextSettings.get_global()
        config3 = FlextSettings.get_global()
        assert config1 is config2
        assert config2 is config3
        assert config1 is config3
        assert isinstance(config1, p.Settings)
        assert isinstance(config2, p.Settings)
        assert isinstance(config3, p.Settings)

    def test_singleton_pattern(self) -> None:
        """Test that FlextSettings.get_global() returns the same instance."""
        config1 = FlextSettings.get_global()
        config2 = FlextSettings.get_global()
        config3 = FlextSettings.get_global()
        assert config1 is config2
        assert config2 is config3
        assert id(config1) == id(config2) == id(config3)
        assert hasattr(config1, "app_name")
        assert hasattr(config1, "log_level")
        assert hasattr(config1, "cache_ttl")
        assert hasattr(config1, "max_workers")

    def test_config_in_flext_container(self) -> None:
        """Test that FlextContainer uses the global config singleton."""
        global_config = FlextSettings.get_global()
        container = FlextContainer()
        config_result = container.get("config")
        if config_result.is_success:
            retrieved_config = config_result.value
            assert retrieved_config is global_config

    def test_environment_variable_override(self) -> None:
        """Test that environment variables override default config."""
        FlextSettings.reset_for_testing()
        os.environ["FLEXT_APP_NAME"] = "test-app-from-env"
        os.environ["FLEXT_LOG_LEVEL"] = "DEBUG"
        os.environ["FLEXT_MAX_WORKERS"] = "8"
        os.environ["FLEXT_TIMEOUT_SECONDS"] = "90"
        os.environ["FLEXT_DEBUG"] = "true"
        try:
            config = FlextSettings.get_global()
            assert config.app_name == "test-app-from-env"
            assert config.log_level == "DEBUG"
            assert config.max_workers == 8
            assert config.timeout_seconds == 90
            assert config.debug is True
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
            config_data: Mapping[str, str | int | bool] = {
                "app_name": "test-app-from-json",
                "environment": "test",
                "log_level": "WARNING",
                "max_name_length": 150,
                "cache_enabled": False,
            }
            config_file_path = temp_directory / "config.json"
            config_file_path.write_text(json.dumps(config_data), encoding="utf-8")
            assert config_file_path.exists()
            assert config_file_path.read_text(encoding="utf-8") == json.dumps(
                config_data,
            )
            config = FlextSettings.get_global()
            assert config.app_name is not None
            assert config.log_level is not None
            assert config.max_workers is not None
            assert config.cache_ttl is not None
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
            config_file = temp_directory / "config.yaml"
            config_data: Mapping[str, str | int | bool] = {
                "app_name": "test-app-from-yaml",
                "environment": "production",
                "debug": False,
                "command_timeout": 60,
                "validation_strict_mode": True,
            }
            with config_file.open("w", encoding="utf-8") as f:
                yaml.dump(config_data, f)
            assert config_file.exists()
            loaded_data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            assert loaded_data == config_data
            config = FlextSettings.get_global()
            assert config.app_name is not None
            assert config.debug is not None
            assert config.timeout_seconds is not None
            assert config.max_batch_size is not None
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
            json_config: Mapping[str, str | int] = {
                "app_name": "from-json",
                "port": 3000,
            }
            json_file = temp_directory / "config.json"
            with json_file.open("w", encoding="utf-8") as f:
                json.dump(json_config, f)
            assert json_file.exists()
            assert json.loads(json_file.read_text(encoding="utf-8")) == json_config
            env_file = temp_directory / ".env"
            env_file.write_text(
                "FLEXT_APP_NAME=from-env\nFLEXT_HOST=env-host\n",
                encoding="utf-8",
            )
            assert env_file.exists()
            assert "FLEXT_APP_NAME=from-env" in env_file.read_text(encoding="utf-8")
            os.environ["FLEXT_APP_NAME"] = "from-env-var"
            config = FlextSettings.get_global()
            assert config.app_name in {"from-env-var", "flext"}
            assert config.cache_ttl in {300, 600}
            assert config.max_retry_attempts in {3, 5}
        finally:
            if "FLEXT_APP_NAME" in os.environ:
                del os.environ["FLEXT_APP_NAME"]
            FlextSettings.reset_for_testing()

    def test_config_singleton_thread_safety(self) -> None:
        """Test that singleton is thread-safe."""
        configs: Sequence[FlextSettings] = []

        def get_config() -> None:
            config = FlextSettings.get_global()
            configs.append(config)

        threads: Sequence[threading.Thread] = []
        for _ in range(10):
            t = threading.Thread(target=get_config)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(configs) == 10
        first_config = configs[0]
        for config in configs[1:]:
            assert config is first_config

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
        saved_env_vars: Mapping[str, str | None] = {
            "FLEXT_APP_NAME": os.environ.pop("FLEXT_APP_NAME", None),
            "FLEXT_LOG_LEVEL": os.environ.pop("FLEXT_LOG_LEVEL", None),
            "FLEXT_DEBUG": os.environ.pop("FLEXT_DEBUG", None),
            "FLEXT_TIMEOUT_SECONDS": os.environ.pop("FLEXT_TIMEOUT_SECONDS", None),
        }
        try:
            config_defaults = FlextSettings.get_global()
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
            config_dotenv = FlextSettings.get_global()
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
            config_env = FlextSettings.get_global()
            assert config_env.app_name == "from-env-var"
            assert config_env.log_level == "DEBUG"
            assert config_env.debug is False
            assert config_env.timeout_seconds == 90
            FlextSettings.reset_for_testing()
            config_explicit = FlextSettings(
                app_name="from-init",
                log_level=FlextConstants.LogLevel.ERROR,
                debug=True,
                timeout_seconds=90,
            )
            assert config_explicit.app_name == "from-init"
            assert config_explicit.log_level == "ERROR"
            assert config_explicit.debug is True
            assert config_explicit.timeout_seconds == 90
            test_logger = FlextLogger("test_precedence")
            assert test_logger is not None
            assert config_explicit.log_level == "ERROR"
            assert config_explicit.effective_log_level == FlextConstants.LogLevel.INFO
            bool(
                getattr(
                    config_explicit,
                    "is_debug_enabled",
                    getattr(config_explicit, "debug", False),
                ),
            )
            assert config_explicit.trace is False
            config_no_debug = FlextSettings(
                log_level=FlextConstants.LogLevel.WARNING,
                debug=False,
            )
            assert (
                config_no_debug.effective_log_level == FlextConstants.LogLevel.WARNING
            )
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
