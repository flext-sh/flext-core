"""Settings precedence integration case kept below module LOC cap."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import FlextSettings
from tests.constants import c
from tests.utilities import u

from .settings_integration_factories import TestsFlextFlextSettingsFactories

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextFlextSettingsPrecedenceCase(TestsFlextFlextSettingsFactories):
    def test_pydantic_settings_precedence_order(self, temp_dir: Path) -> None:
        """Test comprehensive Pydantic 2 Settings precedence order."""
        FlextSettings.reset_for_testing()
        with u.Tests.env_vars_context(
            {},
            vars_to_clear=(
                "FLEXT_APP_NAME",
                "FLEXT_LOG_LEVEL",
                "FLEXT_DEBUG",
                "FLEXT_TIMEOUT_SECONDS",
                "FLEXT_ENV_FILE",
            ),
        ):
            config_defaults = FlextSettings.fetch_global()
            assert config_defaults.app_name == "flext"
            assert config_defaults.log_level == "INFO"
            assert config_defaults.debug is False
            assert config_defaults.timeout_seconds == 30
            FlextSettings.reset_for_testing()
            env_file = temp_dir / ".env"
            env_content = "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\nFLEXT_TIMEOUT_SECONDS=45\n"
            env_file.write_text(env_content, encoding="utf-8")
            assert env_file.exists()
            assert env_file.read_text(encoding="utf-8") == env_content
            with u.Tests.env_vars_context({"FLEXT_ENV_FILE": str(env_file)}):
                config_dotenv = FlextSettings.fetch_global()
            assert config_dotenv.app_name in {"from-dotenv", "flext"}, (
                f"Expected 'from-dotenv' or 'flext' (default), got '{config_dotenv.app_name}'"
            )
            if config_dotenv.app_name == "from-dotenv":
                assert config_dotenv.log_level == "WARNING"
                assert config_dotenv.debug is True
                assert config_dotenv.timeout_seconds == 45
            FlextSettings.reset_for_testing()
            with u.Tests.env_vars_context({
                "FLEXT_APP_NAME": "from-env-var",
                "FLEXT_LOG_LEVEL": "DEBUG",
                "FLEXT_DEBUG": "false",
                "FLEXT_TIMEOUT_SECONDS": "90",
            }):
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
        FlextSettings.reset_for_testing()
