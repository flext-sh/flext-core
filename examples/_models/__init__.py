# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples. Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core import c, d, e, h, p, r, s, t, u, x

    from .errors import m
    from .shared import ExamplesFlextSharedHandle, ExamplesFlextSharedPerson
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
            ".errors": ("m",),
            ".shared": ("ExamplesFlextSharedHandle", "ExamplesFlextSharedPerson"),
            "flext_core": ("c", "d", "e", "h", "p", "r", "s", "t", "u", "x"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
