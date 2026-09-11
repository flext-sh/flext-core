"""FLEXT CLI - Unified Typer abstraction service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from inspect import Parameter

from flext_cli import m, p, settings, t, u
from flext_cli.services._cli_parts.flextclicli_part_01 import (
    FlextCliCli as FlextCliCliPart01,
)
from flext_cli.services.cli_params import FlextCliCommonParams


class FlextCliCli(FlextCliCliPart01):
    """Implementation part for FlextCliCli."""

    def _apply_common_params_to_config(self, *, params: m.Cli.CliParamsConfig) -> None:
        """Apply global CLI flags to the shared settings singleton."""
        resolved_log_level: str = (
            params.log_level if params.log_level is not None else settings.cli_log_level
        )
        next_params = params.model_copy(update={"log_level": resolved_log_level})
        result = FlextCliCommonParams.apply_to_config(settings, params=next_params)
        if result.failure:
            u.fetch_logger(__name__).warning(
                "failed to apply cli params", error=result.error or ""
            )
            return

        updated_settings = result.value
        # NOTE (multi-agent): the ``settings`` singleton is always the
        # concrete FlextCliSettings, which provably satisfies the Settings
        # protocol — the old isinstance guard was dead code (pyright
        # reportUnnecessaryIsInstance). Settings are flat scalars (§2.6):
        # field-level diff drives ``update_global`` directly.
        if updated_settings is settings:
            return
        overrides: dict[str, t.SettingsOverride | None] = {}
        if updated_settings.debug != settings.debug:
            overrides["debug"] = updated_settings.debug
        if updated_settings.trace != settings.trace:
            overrides["trace"] = updated_settings.trace
        for field in (
            "cli_verbose",
            "cli_quiet",
            "cli_no_color",
            "cli_log_level",
            "cli_log_verbosity",
            "cli_output_format",
        ):
            updated_value = getattr(updated_settings, field)
            if updated_value != getattr(settings, field):
                overrides[field] = updated_value
        if overrides:
            settings.update_global(**overrides)

    def create_app_with_common_params(
        self, *, name: str, help_text: str, add_completion: bool = True
    ) -> p.Cli.Application:
        """Create a Typer app with the shared global FLEXT CLI parameters."""
        app = u.Cli.framework_create_app(
            name=name, help_text=help_text, add_completion=add_completion
        )

        def apply_common_params(params: m.Cli.CliParamsConfig) -> bool:
            self._apply_common_params_to_config(params=params)
            return True

        field_names = ("debug", "trace", "verbose", "quiet", "log_level")
        parameters: t.MutableSequenceOf[Parameter] = []
        annotations: t.Cli.CliAnnotations = {"return": bool}
        for field_name in field_names:
            parameter, annotation = self._build_model_parameter(
                field_name, m.Cli.CliParamsConfig.model_fields[field_name], None
            )
            parameters.append(parameter)
            annotations[field_name] = annotation
        global_callback: FlextCliCli._ModelCommand[m.Cli.CliParamsConfig] = (
            self._ModelCommand(
                handler=apply_common_params,
                model_cls=m.Cli.CliParamsConfig,
                parameters=parameters,
            )
        )
        global_callback.__annotations__ = dict(annotations)
        u.Cli.framework_register_callback(app, global_callback)
        return app

    @staticmethod
    def add_group(
        app: p.Cli.Application, *, name: str, group: p.Cli.Application
    ) -> None:
        """Attach a subcommand group to an application."""
        u.Cli.framework_add_group(app, name=name, group=group)

    @staticmethod
    def create_group(*, help_text: str, name: str | None = None) -> p.Cli.Application:
        """Create a Typer command group without re-registering global params."""
        return u.Cli.framework_create_app(
            name=name, help_text=help_text, add_completion=True
        )


__all__: list[str] = ["FlextCliCli"]
