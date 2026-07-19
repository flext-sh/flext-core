"""Public examples typing facade for flext-core."""

from __future__ import annotations

from flext_core import FlextTypes


class ExamplesFlextTypes(FlextTypes):
    """Examples-specific type aliases built from canonical flext-core contracts."""


t = ExamplesFlextTypes

__all__: list[str] = ["ExamplesFlextTypes", "t"]
