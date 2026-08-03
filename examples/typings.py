"""Public examples typing facade for flext-core."""

from __future__ import annotations

from flext_core import FlextTypes, m, p


class ExamplesFlextTypes(FlextTypes):
    """Examples-specific type aliases built from canonical flext-core contracts."""

    class Examples:
        """Examples namespace for shared aliases."""

        type ExampleRenderable = (
            t.JsonPayload
            | t.ScalarOrModel
            | m.ConfigMap
            | p.Result[t.JsonPayload]
            | p.Result[t.JsonValue]
        )


t = ExamplesFlextTypes

__all__: list[str] = ["ExamplesFlextTypes", "t"]
