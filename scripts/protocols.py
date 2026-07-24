"""Protocol definitions for flext."""

from __future__ import annotations

from flext_cli import p


class ScriptsFlextProtocols(p):
    """Protocol definitions for flext."""


p = ScriptsFlextProtocols

__all__: list[str] = ["ScriptsFlextProtocols", "p"]
