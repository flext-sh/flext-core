"""FLEXT CLI - Unified Typer abstraction service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import Parameter, Signature
from types import GenericAlias

from flext_cli import m, p, t, u


class FlextCliCli:
    """Implementation part for FlextCliCli."""

    class _ModelCommand[M: t.Cli.ModelLike]:
        """Callable wrapper with explicit signature for Typer introspection.

        Note: __annotations__ uses MutableMapping[str, type] because Typer reads
        it via inspect at runtime. __call__ uses t.Scalar kwargs because Typer
        may pass scalar values or repeated option lists.
        """

        __name__: str
        __signature__: Signature
        _handler: p.Cli.ModelCommandHandler[M]
        _model_cls: t.ModelClass[M]

        def __init__(
            self,
            *,
            handler: p.Cli.ModelCommandHandler[M],
            model_cls: t.ModelClass[M],
            parameters: t.SequenceOf[Parameter],
        ) -> None:
            self.__name__ = getattr(handler, "__name__", model_cls.__name__)
            self.__signature__ = Signature(parameters)
            self._handler = handler
            self._model_cls = model_cls

        def __call__(self, **kwargs: t.Cli.CliValue) -> t.JsonValue:
            model = self._model_cls.model_validate(kwargs)
            return self._handler(model)

    @classmethod
    def _build_model_parameter(
        cls, field_name: str, field_info: m.FieldInfo, settings: t.Cli.ModelLike | None
    ) -> tuple[Parameter, type | GenericAlias]:
        """Build a keyword-only Typer option from a Pydantic field."""
        alias = getattr(field_info, "alias", None)
        cli_name = alias or field_name
        option_name = f"--{cli_name.replace('_', '-')}"
        annotation = u.Cli.resolve_typer_annotation(
            getattr(field_info, "annotation", None) or str
        )
        is_required = field_info.is_required()
        default_value: t.Cli.CliValue | None = (
            None
            if is_required
            else u.Cli.field_default(field_name, field_info, settings)
        )
        option_decls = [option_name]
        extra = getattr(field_info, "json_schema_extra", None)
        custom_param_decls: list[str] | None = None
        if isinstance(extra, Mapping):
            declared = extra.get("typer_param_decls")
            if isinstance(declared, Sequence) and not isinstance(declared, str):
                custom_param_decls = [str(item) for item in declared]
        if annotation is bool and isinstance(default_value, bool):
            dashed_name = cli_name.replace("_", "-")
            option_decls = [f"--{dashed_name}/--no-{dashed_name}"]
        if custom_param_decls is not None:
            option_decls = custom_param_decls
        spec = m.Cli.OptionSpec(
            declarations=tuple(option_decls),
            help_text=getattr(field_info, "description", None) or "",
            default=default_value,
            required=is_required,
        )
        return (
            u.Cli.framework_build_parameter(field_name, annotation, spec),
            annotation,
        )


__all__: list[str] = ["FlextCliCli"]
