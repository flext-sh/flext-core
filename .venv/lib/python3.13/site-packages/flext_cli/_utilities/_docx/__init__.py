# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Utilities. Docx package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._reader import FlextCliUtilitiesDocxReader
    from ._renderer import FlextCliUtilitiesDocxRenderer
__all__: tuple[str, ...] = (
    "FlextCliUtilitiesDocxReader",
    "FlextCliUtilitiesDocxRenderer",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._reader": ("FlextCliUtilitiesDocxReader",),
            "._renderer": ("FlextCliUtilitiesDocxRenderer",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
