# AUTO-GENERATED FILE — Regenerate with: make gen
"""Scripts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core import (
        config,
        core,
        d,
        e,
        h,
        lazy,
        lazy_attribute,
        normalize_lazy_imports,
        r,
        s,
        settings,
        x,
    )

    from .constants import ScriptsFlextConstants, c
    from .models import ScriptsFlextModels, m
    from .protocols import ScriptsFlextProtocols, p
    from .typings import ScriptsFlextTypes, t
    from .utilities import ScriptsFlextUtilities, u


__all__: tuple[str, ...] = (
    "ScriptsFlextConstants",
    "ScriptsFlextModels",
    "ScriptsFlextProtocols",
    "ScriptsFlextTypes",
    "ScriptsFlextUtilities",
    "c",
    "config",
    "core",
    "d",
    "e",
    "h",
    "lazy",
    "lazy_attribute",
    "m",
    "normalize_lazy_imports",
    "p",
    "r",
    "s",
    "settings",
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
            "flext_core": (
                "config",
                "core",
                "d",
                "e",
                "h",
                "lazy",
                "lazy_attribute",
                "normalize_lazy_imports",
                "r",
                "s",
                "settings",
                "x",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
