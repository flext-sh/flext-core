# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples. Shared Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core import c, d, e, h, m, p, r, s, t, u, x

    from .shared_part_01 import ExamplesFlextSharedBase
    from .shared_part_02 import ExamplesFlextShared
__all__: tuple[str, ...] = (
    "ExamplesFlextShared",
    "ExamplesFlextSharedBase",
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
            ".shared_part_01": ("ExamplesFlextSharedBase",),
            ".shared_part_02": ("ExamplesFlextShared",),
            "flext_core": ("c", "d", "e", "h", "m", "p", "r", "s", "t", "u", "x"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
