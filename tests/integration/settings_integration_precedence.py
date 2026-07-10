"""Settings precedence integration case kept below module LOC cap."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import FlextSettings
from tests.utilities import u

from .settings_integration_factories import TestsFlextFlextSettingsFactories

if TYPE_CHECKING:
    from pathlib import Path

_FLEXT_ENV_KEYS = (
    "FLEXT_LOG_LEVEL",
    "FLEXT_DEBUG",
    "FLEXT_TRACE",
    "FLEXT_TIMEZONE",
    "FLEXT_ENV_FILE",
)


class TestsFlextFlextSettingsPrecedenceCase(TestsFlextFlextSettingsFactories):
    """Assert Pydantic settings precedence across the current field surface."""

    def test_pydantic_settings_precedence_order(self, temp_dir: Path) -> None:
        """Defaults < .env file < env var < explicit init for universal fields."""
        with u.Tests.env_vars_context({}, vars_to_clear=_FLEXT_ENV_KEYS):
            FlextSettings.reset_for_testing()
            defaults = FlextSettings()
            assert defaults.log_level == "INFO"
            assert defaults.debug is False
            assert defaults.trace is False
            assert defaults.timezone == "UTC"

            FlextSettings.reset_for_testing()
            env_file = temp_dir / ".env"
            env_content = "FLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n"
            _ = env_file.write_text(env_content, encoding="utf-8")
            assert env_file.read_text(encoding="utf-8") == env_content
            with u.Tests.env_vars_context({"FLEXT_ENV_FILE": str(env_file)}):
                from_dotenv = FlextSettings(_env_file=str(env_file))
                assert from_dotenv.log_level == "WARNING"
                assert from_dotenv.debug is True

            FlextSettings.reset_for_testing()
            with u.Tests.env_vars_context({
                "FLEXT_LOG_LEVEL": "DEBUG",
                "FLEXT_DEBUG": "true",
                "FLEXT_TRACE": "true",
            }):
                from_env = FlextSettings()
                assert from_env.log_level == "DEBUG"
                assert from_env.debug is True
                assert from_env.trace is True

            FlextSettings.reset_for_testing()
            explicit = FlextSettings(
                log_level="ERROR",
                debug=True,
                trace=False,
                timezone="America/Sao_Paulo",
            )
            assert explicit.log_level == "ERROR"
            assert explicit.debug is True
            assert explicit.trace is False
            assert explicit.timezone == "America/Sao_Paulo"
        FlextSettings.reset_for_testing()
