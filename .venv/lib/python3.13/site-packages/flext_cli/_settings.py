"""FLEXT CLI settings — flat, env-sourceable scalars only (§2.6).

Settings carry ONLY simple scalars (``str | int | bool | float``) loadable
from ``.env`` / environment / CLI flags. Composed, nested, or validated
structures belong in ``config`` (``_config.py``); behavior (such as
test-runtime detection) lives in ``_utilities/settings.py`` behind ``u.Cli``
— never on the settings model.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic_settings import SettingsConfigDict

from flext_cli._constants.settings import FlextCliConstantsSettings
from flext_core import FlextSettings, m


class FlextCliSettings(FlextSettings):
    """CLI settings: flat scalars under the ``FLEXT_CLI_`` env prefix."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="FLEXT_CLI_", extra="ignore"
    )

    cli_verbose: Annotated[bool, m.Field(description="Verbose output")] = (
        FlextCliConstantsSettings.CLI_DEFAULT_VERBOSE
    )
    cli_quiet: Annotated[bool, m.Field(description="Quiet output")] = (
        FlextCliConstantsSettings.CLI_DEFAULT_QUIET
    )
    cli_app_name: Annotated[str, m.Field(description="CLI application name")] = (
        FlextCliConstantsSettings.FLEXT_CLI
    )
    cli_log_verbosity: Annotated[
        str, m.Field(description="Log format (compact, detailed, full)")
    ] = FlextCliConstantsSettings.CLI_DEFAULT_LOG_VERBOSITY
    cli_log_level: Annotated[str, m.Field(description="CLI log level")] = (
        FlextCliConstantsSettings.CLI_DEFAULT_LOG_LEVEL
    )
    cli_no_color: Annotated[bool, m.Field(description="Disable colored output")] = (
        FlextCliConstantsSettings.CLI_DEFAULT_NO_COLOR
    )
    cli_output_format: Annotated[
        str, m.Field(description="Output format (table, json, yaml, csv, plain)")
    ] = FlextCliConstantsSettings.CLI_DEFAULT_OUTPUT_FORMAT
    cli_config_file: Annotated[
        str | None, m.Field(description="Path to settings file")
    ] = None
    cli_token_file: Annotated[
        str | None, m.Field(description="Path to auth token file")
    ] = None
    cli_ci: Annotated[
        bool, m.Field(description="Whether the current runtime is a CI environment.")
    ] = False
    cli_pytest_current_test: Annotated[
        str | None, m.Field(description="Current pytest test identifier.")
    ] = None
    cli_shell_command: Annotated[
        str | None,
        m.Field(description="Current shell command propagated by the runtime."),
    ] = None


settings: FlextCliSettings = FlextCliSettings.fetch_global()
"""Process-wide CLI settings singleton (env-bound, validated at fetch time)."""


__all__: list[str] = ["FlextCliSettings", "settings"]
