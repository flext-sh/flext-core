"""Public examples protocols facade for flext-core."""

from __future__ import annotations

from flext_core import FlextProtocols

p: type[FlextProtocols] = FlextProtocols

__all__: list[str] = ["p"]
