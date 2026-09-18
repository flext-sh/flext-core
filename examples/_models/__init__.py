# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples. Models package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core import c, d, e, h, m, p, r, s, t, u, x

    from .handle import ExamplesFlextSharedHandle
    from .person import ExamplesFlextSharedPerson
__all__: tuple[str, ...] = (
    "ExamplesFlextSharedHandle",
    "ExamplesFlextSharedPerson",
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
            ".handle": ("ExamplesFlextSharedHandle",),
            ".person": ("ExamplesFlextSharedPerson",),
            "flext_core": ("c", "d", "e", "h", "m", "p", "r", "s", "t", "u", "x"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
