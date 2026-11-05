"""Runtime bootstrap option resolution.

Houses :meth:`resolve_runtime_options`. Split from ``model_runtime.py`` so each
layer stays under the 200-LOC cap while the runtime DSL keeps composing via MRO
inheritance.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import flext_core


class FlextUtilitiesModelOptions:
    """Bootstrap-options resolver — RuntimeBootstrapOptions normalization."""

    @classmethod
    def resolve_runtime_options(
        cls,
        source: flext_core.p.RuntimeBootstrapProvider
        | flext_core.p.RuntimeBootstrapOptions
        | flext_core.p.MixinsInfrastructure
        | None = None,
    ) -> flext_core.p.RuntimeBootstrapOptions:
        """Resolve the runtime options an options model or a component declares.

        A component contributes the options its service base declares through
        ``p.RuntimeBootstrapProvider`` and its own runtime seeds; a seed set on
        the instance wins over the declared value. Every value is validated by
        ``m.RuntimeBootstrapOptions``, so a non-conforming port raises
        ``ValidationError`` here instead of reaching the runtime.

        Returns:
            The resulting ``p.RuntimeBootstrapOptions``.

        Raises:
            TypeError: If the runtime bootstrap source has no supported contract.

        """
        match source:
            case None:
                return flext_core.m.RuntimeBootstrapOptions()
            case flext_core.m.RuntimeBootstrapOptions():
                return source
            case flext_core.p.MixinsInfrastructure():
                declared = (
                    flext_core.m.RuntimeBootstrapOptions.model_validate(
                        source.runtime_bootstrap_options(),
                        from_attributes=True,
                    )
                    if isinstance(source, flext_core.p.RuntimeBootstrapProvider)
                    else flext_core.m.RuntimeBootstrapOptions()
                )
                seeds = flext_core.m.RuntimeBootstrapOptions(
                    settings=source.runtime_settings,
                    settings_type=source.settings_type,
                    settings_overrides=source.settings_overrides,
                    context=source.initial_context,
                )
                return flext_core.m.RuntimeBootstrapOptions.model_validate({
                    **dict(declared),
                    **{name: value for name, value in seeds if value is not None},
                })
            case _:
                msg = f"unknown runtime bootstrap source: {source!r}"
                raise TypeError(msg)


__all__: list[str] = ["FlextUtilitiesModelOptions"]
