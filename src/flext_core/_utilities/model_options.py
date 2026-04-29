"""Runtime bootstrap option resolution.

Houses :meth:`resolve_runtime_options` and its helpers. Split from
``model_runtime.py`` so each layer stays under the 200-LOC cap while
the runtime DSL keeps composing via MRO inheritance.
"""

from __future__ import annotations

from collections.abc import Mapping

from flext_core import (
    FlextUtilitiesModel,
    m,
    p,
    t,
)


class FlextUtilitiesModelOptions(FlextUtilitiesModel):
    """Bootstrap-options resolver — RuntimeBootstrapOptions normalization."""

    @classmethod
    def resolve_runtime_options(
        cls,
        source: (
            m.RuntimeBootstrapOptions | Mapping[str, t.JsonPayload] | p.Base | None
        ) = None,
        **overrides: t.JsonPayload,
    ) -> m.RuntimeBootstrapOptions:
        """Resolve runtime options from models, mappings, or service instances."""
        resolved = m.RuntimeBootstrapOptions()
        match source:
            case None:
                pass
            case m.RuntimeBootstrapOptions() as runtime_options:
                resolved = runtime_options
            case Mapping() as source_mapping:
                resolved = m.RuntimeBootstrapOptions.model_validate(source_mapping)
            case _:
                options_resolver = getattr(source, "_runtime_bootstrap_options", None)
                raw_options = options_resolver() if callable(options_resolver) else None
                if raw_options is not None:
                    resolved = cls.resolve_runtime_options(raw_options)
                source_updates: dict[str, t.JsonPayload] = {
                    field_name: value
                    for field_name in m.RuntimeBootstrapOptions.model_fields
                    if (
                        value := getattr(
                            source,
                            "runtime_settings"
                            if field_name == "settings"
                            else "initial_context"
                            if field_name == "context"
                            else field_name,
                            None,
                        )
                    )
                    is not None
                }
                for attr_name, field_name in (
                    ("runtime_dispatcher", "dispatcher"),
                    ("runtime_registry", "registry"),
                ):
                    value = getattr(source, attr_name, None)
                    if value is not None:
                        source_updates[field_name] = value
                if source_updates:
                    resolved = resolved.model_copy(update=source_updates)
        if not overrides:
            return resolved
        override_updates = dict(
            cls.dump(
                cls.resolve_runtime_options(overrides),
                exclude_none=True,
            )
        )
        return (
            resolved.model_copy(update=override_updates)
            if override_updates
            else resolved
        )


__all__: list[str] = ["FlextUtilitiesModelOptions"]
