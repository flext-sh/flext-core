# AUTO-GENERATED FILE — Regenerate with: make gen
"""Scripts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core import d, e, h, r, s, x
    from scripts.constants import ScriptsFlextConstants, c
    from scripts.models import ScriptsFlextModels, m
    from scripts.protocols import ScriptsFlextProtocols, p
    from scripts.typings import ScriptsFlextTypes, t
    from scripts.utilities import ScriptsFlextUtilities, u
_LAZY_IMPORTS = build_lazy_import_map({
    ".constants": ("ScriptsFlextConstants", "c"),
    ".models": ("ScriptsFlextModels", "m"),
    ".protocols": ("ScriptsFlextProtocols", "p"),
    ".typings": ("ScriptsFlextTypes", "t"),
    ".utilities": ("ScriptsFlextUtilities", "u"),
    "flext_core": ("d", "e", "h", "r", "s", "x"),
})


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__: list[str] = [
    "ScriptsFlextConstants",
    "ScriptsFlextModels",
    "ScriptsFlextProtocols",
    "ScriptsFlextTypes",
    "ScriptsFlextUtilities",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "u",
    "x",
]
