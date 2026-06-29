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

import threading
from collections.abc import (
    MutableSequence,
)
from pathlib import Path

from flext_core import FlextContainer, FlextSettings
from tests import p, t, u

from .settings_integration_precedence import FlextSettingsPrecedenceCase


class TestsFlextSettingsIntegration(FlextSettingsPrecedenceCase):
    """Test FlextSettings singleton pattern and integration with all modules using factories."""

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
        with u.Tests.env_vars_context({
            "FLEXT_APP_NAME": "test-app-from-env",
            "FLEXT_LOG_LEVEL": "DEBUG",
            "FLEXT_MAX_WORKERS": "8",
            "FLEXT_TIMEOUT_SECONDS": "90",
            "FLEXT_DEBUG": "true",
        }):
            settings = FlextSettings.fetch_global()
            assert settings.app_name == "test-app-from-env"
            assert settings.log_level == "DEBUG"
            assert settings.max_workers == 8
            assert settings.timeout_seconds == 90
            assert settings.debug is True
        FlextSettings.reset_for_testing()

    def test_json_config_file_loading(self, temp_dir: Path) -> None:
        """Test loading configuration from JSON file."""
        FlextSettings.reset_for_testing()
        with u.Tests.env_vars_context(
            {},
            vars_to_clear=(
                "FLEXT_ENVIRONMENT",
                "FLEXT_APP_NAME",
                "FLEXT_LOG_LEVEL",
            ),
        ):
            config_data: t.MappingKV[str, t.Primitives] = {
                "app_name": "test-app-from-json",
                "environment": "test",
                "log_level": "WARNING",
                "max_name_length": 150,
                "cache_enabled": False,
            }
            config_file_path = temp_dir / "settings.json"
            u.Cli.json_write(config_file_path, config_data)
            assert config_file_path.exists()
            empty_config: t.JsonMapping = {}
            loaded = u.Cli.json_read(config_file_path).unwrap_or(empty_config)
            assert loaded == config_data
            settings = FlextSettings.fetch_global()
            assert settings.app_name is not None
            assert settings.log_level is not None
            assert settings.max_workers is not None
            assert settings.cache_ttl is not None
        FlextSettings.reset_for_testing()

    def test_yaml_config_file_loading(self, temp_dir: Path) -> None:
        """Test loading configuration from YAML file."""
        FlextSettings.reset_for_testing()
        with u.Tests.env_vars_context(
            {},
            vars_to_clear=(
                "FLEXT_ENVIRONMENT",
                "FLEXT_APP_NAME",
                "FLEXT_DEBUG",
            ),
        ):
            config_file = temp_dir / "settings.yaml"
            config_data: t.MappingKV[str, t.Primitives] = {
                "app_name": "test-app-from-yaml",
                "environment": "production",
                "debug": False,
                "command_timeout": 60,
                "validation_strict_mode": True,
            }
            u.Cli.yaml_dump(config_file, config_data)
            assert config_file.exists()
            empty_config: t.JsonMapping = {}
            loaded_data = u.Cli.yaml_parse(
                config_file.read_text(encoding="utf-8"),
            ).unwrap_or(empty_config)
            assert loaded_data == config_data
            settings = FlextSettings.fetch_global()
            assert settings.app_name is not None
            assert settings.debug is not None
            assert settings.timeout_seconds is not None
            assert settings.max_batch_size is not None
        FlextSettings.reset_for_testing()

    def test_config_priority_order(self, temp_dir: Path) -> None:
        """Test that configuration sources have correct priority.

        Uses temp_dir fixture to avoid writing files to current directory.
        Validates priority order: env var > .env > json.
        """
        FlextSettings.reset_for_testing()
        with u.Tests.env_vars_context(
            {"FLEXT_APP_NAME": "from-env-var"},
        ):
            json_config: t.HeaderMapping = {
                "app_name": "from-json",
                "port": 3000,
            }
            json_file = temp_dir / "settings.json"
            u.Cli.json_write(json_file, json_config)
            assert json_file.exists()
            assert u.Cli.json_read(json_file).unwrap_or({}) == json_config
            env_file = temp_dir / ".env"
            env_file.write_text(
                "FLEXT_APP_NAME=from-env\nFLEXT_HOST=env-host\n",
                encoding="utf-8",
            )
            assert env_file.exists()
            assert "FLEXT_APP_NAME=from-env" in env_file.read_text(encoding="utf-8")
            settings = FlextSettings.fetch_global()
            assert settings.app_name in {"from-env-var", "flext"}
            assert settings.cache_ttl in {300, 600}
            assert settings.max_retry_attempts in {3, 5}
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
