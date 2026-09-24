# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core import d, e, h, r, s, x

    from . import _models, _shared_parts
    from .constants import c
    from .models import ExamplesFlextModels, m
    from .protocols import p
    from .shared import ExamplesFlextShared
    from .typings import t
    from .utilities import u


__all__: tuple[str, ...] = (
    "ExamplesFlextModels",
    "ExamplesFlextShared",
    "_models",
    "_shared_parts",
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
            "._models": ("_models",),
            "._shared_parts": ("_shared_parts",),
            ".constants": ("c",),
            ".models": ("ExamplesFlextModels", "m"),
            ".protocols": ("p",),
            ".shared": ("ExamplesFlextShared",),
            ".typings": ("t",),
            ".utilities": ("u",),
            "flext_core": ("d", "e", "h", "r", "s", "x"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
