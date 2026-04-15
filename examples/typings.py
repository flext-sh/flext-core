"""Public examples typing facade for flext-core."""

from __future__ import annotations

from examples import p
from flext_core import t


class ExamplesFlextCoreTypes(t):
    """Examples-specific type aliases built from canonical flext-core contracts."""

    class Examples:
        """Examples namespace for shared aliases."""

        type ExampleRenderable = (
            t.ValueOrModel
            | t.ConfigMap
            | p.ResultLike[t.RuntimeAtomic]
            | p.ResultLike[t.Container]
        )


t = ExamplesFlextCoreTypes

__all__: list[str] = ["ExamplesFlextCoreTypes", "t"]
