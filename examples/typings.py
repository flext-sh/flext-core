"""Public examples typing facade for flext-core."""

from __future__ import annotations

from flext_core import m, p, t


class ExamplesFlextCoreTypes(t):
    """Examples-specific type aliases built from canonical flext-core contracts."""

    class Examples:
        """Examples namespace for shared aliases."""

        type ExampleRenderable = (
            t.RuntimeData
            | t.ScalarOrModel
            | m.ConfigMap
            | p.ResultLike[t.RuntimeData]
            | p.ResultLike[t.Container]
        )


t = ExamplesFlextCoreTypes

__all__: list[str] = ["ExamplesFlextCoreTypes", "t"]
