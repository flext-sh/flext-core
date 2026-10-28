"""Settings precedence integration case kept below module LOC cap.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from typing import TYPE_CHECKING

from flext_core import FlextSettings
from tests import c, tm
from tests.integration.settings_integration_factories import (
    TestsFlextFlextSettingsFactories,
)
from tests.utilities import u

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

    @staticmethod
    def test_pydantic_settings_precedence_order(temp_dir: Path) -> None:
        """Defaults < .env file < env var < explicit init for universal fields."""
        with u.Tests.env_vars_context({}, vars_to_clear=_FLEXT_ENV_KEYS):
            FlextSettings.reset_for_testing()
            defaults = FlextSettings()
            tm.that(defaults.log_level, eq="INFO")
            tm.fail(defaults.debug)
            tm.fail(defaults.trace)
            tm.that(defaults.timezone, eq="UTC")

            FlextSettings.reset_for_testing()
            env_file = temp_dir / ".env"
            env_content = "FLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n"
            _ = env_file.write_text(env_content, encoding="utf-8")
            tm.that(env_file.read_text(encoding="utf-8"), eq=env_content)
            with u.Tests.env_vars_context({"FLEXT_ENV_FILE": str(env_file)}):
                from_dotenv = FlextSettings()
                tm.that(from_dotenv.log_level, eq="WARNING")
                tm.ok(from_dotenv.debug)

            FlextSettings.reset_for_testing()
            with u.Tests.env_vars_context({
                "FLEXT_LOG_LEVEL": "DEBUG",
                "FLEXT_DEBUG": c.ParserBooleanToken.TRUE,
                "FLEXT_TRACE": c.ParserBooleanToken.TRUE,
            }):
                from_env = FlextSettings()
                tm.that(from_env.log_level, eq="DEBUG")
                tm.ok(from_env.debug)
                tm.ok(from_env.trace)

            FlextSettings.reset_for_testing()
            explicit = FlextSettings(
                log_level="ERROR",
                debug=True,
                trace=False,
                timezone="America/Sao_Paulo",
            )
            tm.that(explicit.log_level, eq="ERROR")
            tm.ok(explicit.debug)
            tm.fail(explicit.trace)
            tm.that(explicit.timezone, eq="America/Sao_Paulo")
        FlextSettings.reset_for_testing()
