"""FLEXT CLI configuration and registry constants."""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Final

from flext_cli._constants.enums import FlextCliConstantsEnums as ce
from flext_core import c, t


class FlextCliConstantsSettings:
    """CLI defaults, messages, registries, and output configuration."""

    OUTPUT_FORMATS: ClassVar[t.StrSequence] = tuple(
        output_format.value for output_format in ce.OutputFormats
    )
    LOG_LEVELS: ClassVar[t.StrSequence] = tuple(item.value for item in c.LogLevel)
    MESSAGE_TYPES: ClassVar[t.StrSequence] = tuple(
        item.value for item in ce.MessageTypes
    )

    CLI_DEFAULT_NO_COLOR: Final[bool] = False
    CLI_DEFAULT_VERBOSE: Final[bool] = False
    CLI_DEFAULT_QUIET: Final[bool] = False
    # NOTE (multi-agent): canonical scalar defaults consumed by _settings.py —
    # the settings foundation imports this PURE private module directly
    # (no facade cycle) so defaults stay SSOT with the enums (§1.8/§2.5).
    CLI_DEFAULT_LOG_VERBOSITY: Final[str] = ce.LogVerbosity.COMPACT.value
    CLI_DEFAULT_LOG_LEVEL: Final[str] = c.LogLevel.INFO.value
    CLI_DEFAULT_OUTPUT_FORMAT: Final[str] = ce.OutputFormats.TABLE.value
    ENV_DEFAULT_CI: Final[bool] = False
    ENV_VAR_CI: Final[str] = "CI"
    ENV_VAR_PYTEST_CURRENT_TEST: Final[str] = "PYTEST_CURRENT_TEST"
    ENV_VAR_SHELL_COMMAND: Final[str] = "_"

    FLEXT_CLI: Final[str] = "flext-cli"
    CLI_VERSION: Final[str] = "2.0.0"
    OPTIONAL_UNION_ARG_COUNT: Final[int] = 2
    CLI_SCALAR_TYPES_TUPLE: ClassVar[
        tuple[type[str], type[int], type[float], type[bool]]
    ] = t.PRIMITIVES_TYPES

    CLI_PARAM_SHORT_FLAG_VERBOSE: Final[str] = "v"
    CLI_PARAM_SHORT_FLAG_QUIET: Final[str] = "q"
    CLI_PARAM_SHORT_FLAG_DEBUG: Final[str] = "d"
    CLI_PARAM_SHORT_FLAG_TRACE: Final[str] = "t"
    CLI_PARAM_SHORT_FLAG_LOG_LEVEL: Final[str] = "L"
    CLI_PARAM_SHORT_FLAG_OUTPUT_FORMAT: Final[str] = "o"
    CLI_PARAM_SHORT_FLAG_CONFIG_FILE: Final[str] = "c"
    CLI_PARAM_PRIORITY_VERBOSE: Final[int] = 1
    CLI_PARAM_PRIORITY_QUIET: Final[int] = 2
    CLI_PARAM_PRIORITY_DEBUG: Final[int] = 3
    CLI_PARAM_PRIORITY_TRACE: Final[int] = 4
    CLI_PARAM_PRIORITY_LOG_LEVEL: Final[int] = 5
    CLI_PARAM_PRIORITY_LOG_FORMAT: Final[int] = 6
    CLI_PARAM_PRIORITY_OUTPUT_FORMAT: Final[int] = 7
    CLI_PARAM_PRIORITY_NO_COLOR: Final[int] = 8
    CLI_PARAM_PRIORITY_CONFIG_FILE: Final[int] = 9
    CLI_PARAM_KEY_SHORT: Final[str] = "short"
    CLI_PARAM_KEY_DEFAULT: Final[str] = "default"
    CLI_PARAM_KEY_PRIORITY: Final[str] = "priority"
    CLI_PARAM_KEY_CHOICES: Final[str] = "choices"
    CLI_PARAM_KEY_CASE_SENSITIVE: Final[str] = "case_sensitive"
    CLI_PARAM_KEY_FIELD_NAME_OVERRIDE: Final[str] = "field_name_override"
    CLI_PARAM_LOG_FORMAT_OVERRIDE: Final[str] = "log-format"
    CLI_PARAM_CASE_INSENSITIVE: Final[bool] = False
    CLI_VALID_LOG_FORMATS: ClassVar[t.StrSequence] = tuple(
        item.value for item in ce.LogVerbosity
    )

    COMMANDS_DEFAULT_NAME: Final[str] = "flext"
    COMMANDS_DEFAULT_DESCRIPTION: Final[str] = "FLEXT CLI"

    CLI_PARAM_REGISTRY: ClassVar[
        t.MappingKV[str, t.MappingKV[str, t.Scalar | t.StrSequence]]
    ] = MappingProxyType({
        "verbose": {
            CLI_PARAM_KEY_SHORT: CLI_PARAM_SHORT_FLAG_VERBOSE,
            CLI_PARAM_KEY_PRIORITY: CLI_PARAM_PRIORITY_VERBOSE,
        },
        "quiet": {
            CLI_PARAM_KEY_SHORT: CLI_PARAM_SHORT_FLAG_QUIET,
            CLI_PARAM_KEY_PRIORITY: CLI_PARAM_PRIORITY_QUIET,
        },
        "debug": {
            CLI_PARAM_KEY_SHORT: CLI_PARAM_SHORT_FLAG_DEBUG,
            CLI_PARAM_KEY_PRIORITY: CLI_PARAM_PRIORITY_DEBUG,
        },
        "trace": {
            CLI_PARAM_KEY_SHORT: CLI_PARAM_SHORT_FLAG_TRACE,
            CLI_PARAM_KEY_PRIORITY: CLI_PARAM_PRIORITY_TRACE,
        },
        "cli_log_level": {
            CLI_PARAM_KEY_SHORT: CLI_PARAM_SHORT_FLAG_LOG_LEVEL,
            CLI_PARAM_KEY_PRIORITY: CLI_PARAM_PRIORITY_LOG_LEVEL,
            CLI_PARAM_KEY_CHOICES: LOG_LEVELS,
            CLI_PARAM_KEY_CASE_SENSITIVE: CLI_PARAM_CASE_INSENSITIVE,
            CLI_PARAM_KEY_FIELD_NAME_OVERRIDE: "log_level",
        },
        "log_verbosity": {
            CLI_PARAM_KEY_PRIORITY: CLI_PARAM_PRIORITY_LOG_FORMAT,
            CLI_PARAM_KEY_CHOICES: LOG_LEVELS,
            CLI_PARAM_KEY_CASE_SENSITIVE: CLI_PARAM_CASE_INSENSITIVE,
            CLI_PARAM_KEY_FIELD_NAME_OVERRIDE: CLI_PARAM_LOG_FORMAT_OVERRIDE,
        },
        "output_format": {
            CLI_PARAM_KEY_SHORT: CLI_PARAM_SHORT_FLAG_OUTPUT_FORMAT,
            CLI_PARAM_KEY_PRIORITY: CLI_PARAM_PRIORITY_OUTPUT_FORMAT,
            CLI_PARAM_KEY_CHOICES: list(OUTPUT_FORMATS),
            CLI_PARAM_KEY_CASE_SENSITIVE: CLI_PARAM_CASE_INSENSITIVE,
        },
        "no_color": {CLI_PARAM_KEY_PRIORITY: CLI_PARAM_PRIORITY_NO_COLOR},
        "config_file": {
            CLI_PARAM_KEY_SHORT: CLI_PARAM_SHORT_FLAG_CONFIG_FILE,
            CLI_PARAM_KEY_PRIORITY: CLI_PARAM_PRIORITY_CONFIG_FILE,
        },
    })
