"""Runtime bootstrap option resolution.

Houses :meth:`resolve_runtime_options`. Split from ``model_runtime.py`` so each
layer stays under the 200-LOC cap while the runtime DSL keeps composing via MRO
inheritance.
"""

from __future__ import annotations

from flext_core import m, p

from .model import FlextUtilitiesModel


class FlextUtilitiesModelOptions(FlextUtilitiesModel):
    """Bootstrap-options resolver — RuntimeBootstrapOptions normalization."""

    @classmethod
    def resolve_runtime_options(
        cls, source: m.RuntimeBootstrapOptions | p.MixinsInfrastructure | None = None
    ) -> m.RuntimeBootstrapOptions:
        """Resolve the runtime options an options model or a component declares.

        A component contributes the options its service base declares through
        ``p.RuntimeBootstrapProvider`` and its own runtime seeds; a seed set on
        the instance wins over the declared value. Every value is validated by
        ``m.RuntimeBootstrapOptions``, so a non-conforming port raises
        ``ValidationError`` here instead of reaching the runtime.
        """
        match source:
            case None:
                return m.RuntimeBootstrapOptions()
            case m.RuntimeBootstrapOptions():
                return source
            case p.MixinsInfrastructure():
                declared = (
                    m.RuntimeBootstrapOptions.model_validate(
                        source.runtime_bootstrap_options(), from_attributes=True
                    )
                    if isinstance(source, p.RuntimeBootstrapProvider)
                    else m.RuntimeBootstrapOptions()
                )
                seeds = m.RuntimeBootstrapOptions(
                    settings=source.runtime_settings,
                    settings_type=source.settings_type,
                    settings_overrides=source.settings_overrides,
                    context=source.initial_context,
                )
                return m.RuntimeBootstrapOptions.model_validate({
                    **dict(declared),
                    **{name: value for name, value in seeds if value is not None},
                })
            case _:
                msg = f"unknown runtime bootstrap source: {source!r}"
                raise TypeError(msg)


__all__: list[str] = ["FlextUtilitiesModelOptions"]
