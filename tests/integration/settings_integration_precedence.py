"""Settings precedence integration case kept below module LOC cap."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import FlextSettings
from flext_tests import tm
from tests import u

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
            tm.that(defaults.log_level, eq="INFO")
            tm.that(defaults.debug, eq=False)
            tm.that(defaults.trace, eq=False)
            tm.that(defaults.timezone, eq="UTC")

            FlextSettings.reset_for_testing()
            env_file = temp_dir / ".env"
            env_content = "FLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n"
            _ = env_file.write_text(env_content, encoding="utf-8")
            tm.that(env_file.read_text(encoding="utf-8"), eq=env_content)
            with u.Tests.env_vars_context({"FLEXT_ENV_FILE": str(env_file)}):
                from_dotenv = FlextSettings(_env_file=str(env_file))
                tm.that(from_dotenv.log_level, eq="WARNING")
                tm.that(from_dotenv.debug, eq=True)

            FlextSettings.reset_for_testing()
            with u.Tests.env_vars_context({
                "FLEXT_LOG_LEVEL": "DEBUG",
                "FLEXT_DEBUG": "true",
                "FLEXT_TRACE": "true",
            }):
                from_env = FlextSettings()
                tm.that(from_env.log_level, eq="DEBUG")
                tm.that(from_env.debug, eq=True)
                tm.that(from_env.trace, eq=True)

            FlextSettings.reset_for_testing()
            explicit = FlextSettings(
                log_level="ERROR", debug=True, trace=False, timezone="America/Sao_Paulo"
            )
            tm.that(explicit.log_level, eq="ERROR")
            tm.that(explicit.debug, eq=True)
            tm.that(explicit.trace, eq=False)
            tm.that(explicit.timezone, eq="America/Sao_Paulo")
        FlextSettings.reset_for_testing()
