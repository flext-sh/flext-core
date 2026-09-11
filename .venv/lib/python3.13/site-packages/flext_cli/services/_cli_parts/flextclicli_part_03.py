"""FLEXT CLI - Unified Typer abstraction service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from inspect import Parameter
from typing import TYPE_CHECKING

from flext_cli import c, m, p, r, t, u
from flext_cli.services._cli_parts.flextclicli_part_02 import (
    FlextCliCli as FlextCliCliPart02,
)

if TYPE_CHECKING:
    # mro-j47u (codex): the earlier MRO part is referenced only by annotation;
    # inspect.Parameter remains runtime because it constructs the CLI signature.
    from flext_cli.services._cli_parts.flextclicli_part_01 import (
        FlextCliCli as FlextCliCliPart01,
    )


class FlextCliCli(FlextCliCliPart02):
    """Implementation part for FlextCliCli."""

    @classmethod
    def model_command[M: t.Cli.ModelLike](
        cls,
        model_cls: t.ModelClass[M],
        handler: p.Cli.ModelCommandHandler[M],
        settings: t.Cli.ModelLike | None = None,
    ) -> t.Cli.CliCommand:
        """Build a Typer command directly from a Pydantic request model."""
        parameters: t.MutableSequenceOf[Parameter] = []
        annotations: t.Cli.CliAnnotations = {"return": type(None)}
        fields = getattr(model_cls, "model_fields", {})
        for field_name, field_info in fields.items():
            if getattr(field_info, "exclude", None) is True:
                continue
            parameter, annotation = cls._build_model_parameter(
                field_name, field_info, settings
            )
            parameters.append(parameter)
            annotations[field_name] = annotation
        command: FlextCliCliPart01._ModelCommand[M] = cls._ModelCommand(
            handler=handler, model_cls=model_cls, parameters=parameters
        )
        command.__annotations__ = dict(annotations)
        return command

    @classmethod
    def derive_model[M: t.Cli.ModelLike](
        cls,
        model_cls: type[M],
        *sources: t.Cli.ModelSource,
        overrides: t.ScalarMapping | None = None,
    ) -> M:
        """Derive a target Pydantic model from ordered model/mapping sources."""
        merged: t.MutableJsonMapping = {}
        for source in sources:
            merged.update(u.Cli.model_source_data(model_cls, source))
        if overrides is not None:
            merged.update(u.Cli.model_source_data(model_cls, overrides))
        validated: M = model_cls.model_validate(merged)
        return validated

    @staticmethod
    def invoke_app(
        app: p.Cli.Application,
        *,
        args: t.StrSequence | None = None,
        charset: str = c.Cli.ENCODING_DEFAULT,
        env: t.StrMapping | None = None,
    ) -> p.Result[m.Cli.InvocationResult]:
        """Invoke an application through the private real-framework boundary."""
        try:
            invocation = u.Cli.framework_invoke(
                app, args=args, charset=charset, env=env
            )
        except (TypeError, ValueError) as exc:
            return r[m.Cli.InvocationResult].fail(str(exc))
        return r[m.Cli.InvocationResult].ok(invocation)


__all__: list[str] = ["FlextCliCli"]
