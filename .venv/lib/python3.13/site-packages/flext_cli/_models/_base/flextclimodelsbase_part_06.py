"""CLI Pydantic domain models."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_cli import t
from flext_core import m


class FlextCliModelsBase:
    """Implementation part for FlextCliModelsBase."""

    class CliParamsConfig(m.Value):
        """CLI parameters configuration for command-line parsing.

        Maps directly to CLI flags: --verbose, --quiet, --debug, --trace, etc.
        All fields are optional (None) to allow partial updates.
        Inherits frozen=True and extra="forbid" from m.Value.
        """

        verbose: Annotated[
            bool | None, m.Field(description="Enable verbose output")
        ] = None
        quiet: Annotated[
            bool | None, m.Field(description="Suppress non-essential output")
        ] = None
        debug: Annotated[
            bool | None, m.Field(None, description="Enable debug mode")
        ] = None
        trace: Annotated[
            bool | None, m.Field(description="Enable trace logging (requires debug)")
        ] = None
        log_level: Annotated[
            str | None, m.Field(description="Log level (DEBUG, INFO, WARNING, ERROR)")
        ] = None
        log_format: Annotated[
            str | None, m.Field(description="Log format (compact, detailed, full)")
        ] = None
        output_format: Annotated[
            str | None, m.Field(description="Output format (table, json, yaml, csv)")
        ] = None
        no_color: Annotated[
            bool | None, m.Field(description="Disable colored output")
        ] = None

        @property
        def params(self) -> t.JsonMapping:
            """Parameters mapping - required by CliParamsConfig."""
            return {
                "verbose": self.verbose or False,
                "quiet": self.quiet or False,
                "debug": self.debug or False,
                "trace": self.trace or False,
                "log_level": self.log_level or "",
                "log_format": self.log_format or "",
                "output_format": self.output_format or "",
                "no_color": self.no_color or False,
            }

    class OptionMetadata(m.BaseModel):
        """Validated option-registry metadata for Typer option generation."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="ignore", frozen=True)

        help: Annotated[
            str, m.Field("", description="Option help text", strict=True)
        ] = ""
        short: Annotated[
            str,
            m.Field(
                "",
                description="Single-letter short flag without leading dash",
                strict=True,
            ),
        ] = ""
        default: Annotated[
            t.Cli.CliValue | None,
            m.Field(None, description="Option default value when explicitly provided"),
        ] = None
        field_name_override: Annotated[
            str | None,
            m.Field(
                None, description="Override for the CLI-facing option name", strict=True
            ),
        ] = None

    class OptionSpec(m.Value):
        """Framework-neutral command option consumed by the private adapter."""

        declarations: Annotated[
            t.StrSequence,
            m.Field(description="Ordered long and short option declarations"),
        ]
        help_text: Annotated[str, m.Field(description="Human-readable option help")]
        default: Annotated[
            t.Cli.CliValue | None,
            m.Field(None, description="Validated optional default value"),
        ] = None
        required: Annotated[
            bool, m.Field(False, description="Require an explicit option value")
        ] = False

    class InvocationResult(m.Value):
        """Framework-neutral result of one real CLI invocation."""

        exit_code: Annotated[int, m.Field(description="Process-compatible exit code")]
        stdout: Annotated[str, m.Field(description="Captured standard output")] = ""
        stderr: Annotated[str, m.Field(description="Captured standard error")] = ""


__all__: list[str] = ["FlextCliModelsBase"]
