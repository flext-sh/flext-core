"""API facade for flext-core."""

from __future__ import annotations


class FlextApi:
    """API facade for flext-core."""


core: FlextApi = FlextApi()
"""Process-wide ``FlextApi`` facade singleton for the foundation package."""

__all__: list[str] = ["FlextApi", "core"]
