"""Runtime bootstrap option resolution.

Houses :meth:`resolve_runtime_options`. Split from ``model_runtime.py`` so each
layer stays under the 200-LOC cap while the runtime DSL keeps composing via MRO
inheritance.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from flext_core import m, p
from flext_core._utilities import FlextUtilitiesModel


class FlextUtilitiesModelOptions(FlextUtilitiesModel):
    """Bootstrap-options resolver — RuntimeBootstrapOptions normalization."""

    @classmethod
    def resolve_runtime_options(
        cls,
        source: m.RuntimeBootstrapOptions | p.MixinsInfrastructure | None = None,
    ) -> m.RuntimeBootstrapOptions:
        """Resolve the runtime options an options model or a component declares.

        A component contributes the options its service base declares through
        ``p.RuntimeBootstrapProvider`` and its own runtime seeds; a seed set on
        the instance wins over the declared value. Every value is validated by
        ``m.RuntimeBootstrapOptions``, so a non-conforming port raises
        ``ValidationError`` here instead of reaching the runtime.

        Returns:
            The resulting ``m.RuntimeBootstrapOptions``.

        Raises:
            TypeError: If unknown runtime bootstrap source.

        """
        match source:
            case None:
                return m.RuntimeBootstrapOptions()
            case m.RuntimeBootstrapOptions():
                return source
            case _:
                candidate = cast(
                    "m.RuntimeBootstrapOptions | p.MixinsInfrastructure",
                    source,
                )
                if not isinstance(candidate, p.MixinsInfrastructure):
                    msg = f"unknown runtime bootstrap source: {candidate!r}"
                    raise TypeError(msg)
                declared = (
                    m.RuntimeBootstrapOptions.model_validate(
                        candidate.runtime_bootstrap_options(),
                        from_attributes=True,
                    )
                    if isinstance(candidate, p.RuntimeBootstrapProvider)
                    else m.RuntimeBootstrapOptions()
                )
                seeds = m.RuntimeBootstrapOptions(
                    settings=candidate.runtime_settings,
                    settings_type=candidate.settings_type,
                    settings_overrides=candidate.settings_overrides,
                    context=candidate.initial_context,
                )
                return m.RuntimeBootstrapOptions.model_validate({
                    **dict(declared),
                    **{name: value for name, value in seeds if value is not None},
                })


__all__: list[str] = ["FlextUtilitiesModelOptions"]
