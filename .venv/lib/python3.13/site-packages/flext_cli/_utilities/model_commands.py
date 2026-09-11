"""Thin model-command adapters shared through ``u.Cli``.

NOTE (multi-agent): handlers are annotated as the structural type
``Callable[[M], t.JsonValue]`` (the exact contract of
``p.Cli.ModelCommandHandler``) because mypy degrades inherited nested
protocol classes reached through the ``p.Cli`` facade MRO to ``Any``
(pyrefly/pyright resolve them). Structural typing is the FLEXT default and
keeps all three checkers precise without casts.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Protocol, cast

from flext_cli import p, settings, t
from flext_core import m


class _SignatureCarrier(Protocol):
    """A callable whose CLI parameters typer/click read from `__signature__`.

    `__signature__` is a documented runtime attribute of function objects, but
    it is absent from the static `FunctionType`, so assigning it directly is
    rejected by the type checkers. Declaring it here states the contract the
    generated command actually satisfies, with no suppression.
    """

    __signature__: inspect.Signature
    __annotations__: dict[str, object]


class FlextCliUtilitiesModelCommands:
    """Model command methods exposed directly on ``u.Cli``."""

    class Builder[M: t.Cli.ModelLike]:
        """Thin builder for direct model-backed command callables."""

        def __init__(
            self,
            model_class: t.ModelClass[M],
            handler: Callable[[M], t.JsonValue],
            settings: t.Cli.ModelLike | None = None,
        ) -> None:
            """Store the canonical inputs for deferred command construction."""
            super().__init__()
            self.model_class = model_class
            self.handler = handler
            self.settings = settings

        def _resolve_default(self, field_info: m.FieldInfo) -> t.Cli.CliValue | type:
            if field_info.is_required():
                return inspect.Parameter.empty
            # NOTE (multi-agent): ``FieldInfo.get_default`` is typed ``Any``
            # in pydantic; the declared union is the real runtime contract.
            return cast(
                "t.Cli.CliValue | type",
                field_info.get_default(call_default_factory=True),
            )

        def build(self) -> t.Cli.CliCommand:
            """Build a direct callable with a real runtime signature."""
            model_fields = getattr(self.model_class, "model_fields", {})
            parameters = [
                inspect.Parameter(
                    name=field_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=self._resolve_default(field_info),
                    annotation=getattr(field_info, "annotation", None) or str,
                )
                for field_name, field_info in model_fields.items()
                if getattr(field_info, "exclude", None) is not True
            ]
            signature = inspect.Signature(parameters)

            # NOTE (multi-agent): ``self.settings`` (per-command, may be any
            # model or None) falls back to the module settings singleton.
            # Overrides apply only when the effective settings is a full
            # Settings protocol (has update_global) — a plain model skips them.
            effective_settings = (
                self.settings if self.settings is not None else settings
            )

            def command(**kwargs: t.Cli.CliValue) -> t.JsonValue:
                if isinstance(effective_settings, p.Cli.Settings):
                    settings_fields = effective_settings.model_dump()
                    applicable_overrides = {
                        field_name: field_value
                        for field_name, field_value in kwargs.items()
                        if field_name in settings_fields
                    }
                    if applicable_overrides:
                        effective_settings.update_global(**applicable_overrides)
                model = self.model_class.model_validate(kwargs)
                return self.handler(model)

            # typer/click read the CLI parameters off `__signature__`; the
            # protocol above declares that contract so no cast is suppressed.
            typed_command = cast("_SignatureCarrier", command)
            typed_command.__signature__ = signature
            typed_command.__annotations__ = {
                parameter.name: parameter.annotation for parameter in parameters
            }
            typed_command.__annotations__["return"] = t.JsonValue
            return command

    @staticmethod
    def model_source_data(
        model_cls: t.ModelClass[t.Cli.ModelLike], source: t.Cli.ModelSource
    ) -> t.JsonMapping:
        """Extract only target-compatible fields from a model or mapping source."""
        raw_source: t.JsonMapping | t.ScalarMapping
        if isinstance(source, Mapping):
            raw_source = source
        else:
            raw_source = source.model_dump(exclude_none=True)
        filtered_payload = {
            field_name: raw_source[field_name]
            for field_name in model_cls.model_fields
            if field_name in raw_source and raw_source[field_name] is not None
        }
        return t.Cli.JSON_MAPPING_ADAPTER.validate_python(filtered_payload)

    @classmethod
    def derive_model[M: t.Cli.ModelLike](
        cls,
        model_cls: type[M],
        *sources: t.Cli.ModelSource,
        overrides: t.ScalarMapping | None = None,
    ) -> M:
        """Derive a target model from ordered model/mapping sources."""
        merged: t.MutableJsonMapping = {}
        for source in sources:
            merged.update(cls.model_source_data(model_cls, source))
        if overrides is not None:
            merged.update(cls.model_source_data(model_cls, overrides))
        validated: M = model_cls.model_validate(merged)
        return validated

    @staticmethod
    def build_model_command[M: t.Cli.ModelLike](
        model_class: t.ModelClass[M],
        handler: Callable[[M], t.JsonValue],
        settings: t.Cli.ModelLike | None = None,
    ) -> t.Cli.CliCommand:
        """Build a model command through the canonical CLI service."""
        # NOTE (multi-agent): All model-class ingress uses t.ModelClass.
        return FlextCliUtilitiesModelCommands.Builder(
            model_class=model_class, handler=handler, settings=settings
        ).build()


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesModelCommands"]
