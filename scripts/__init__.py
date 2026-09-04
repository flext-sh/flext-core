# AUTO-GENERATED FILE — Regenerate with: make gen
"""Scripts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_cli import d, e, h, r, s, x

    from .constants import ScriptsFlextConstants, ScriptsFlextConstants as c
    from .models import ScriptsFlextModels, ScriptsFlextModels as m
    from .protocols import ScriptsFlextProtocols, ScriptsFlextProtocols as p
    from .typings import ScriptsFlextTypes, ScriptsFlextTypes as t
    from .utilities import ScriptsFlextUtilities, ScriptsFlextUtilities as u
__all__: tuple[str, ...] = (
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
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".constants": ("ScriptsFlextConstants", "c"),
            ".models": ("ScriptsFlextModels", "m"),
            ".protocols": ("ScriptsFlextProtocols", "p"),
            ".typings": ("ScriptsFlextTypes", "t"),
            ".utilities": ("ScriptsFlextUtilities", "u"),
            "flext_cli": ("d", "e", "h", "r", "s", "x"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
