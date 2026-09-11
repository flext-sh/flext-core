# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Utilities. Pptx package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._reader import FlextCliUtilitiesPptxReader
    from ._renderer import FlextCliUtilitiesPptxRenderer
    from ._serializer import FlextCliUtilitiesPptxSerializer
    from ._types import FlextCliUtilitiesPptxTypes
__all__: tuple[str, ...] = (
    "FlextCliUtilitiesPptxReader",
    "FlextCliUtilitiesPptxRenderer",
    "FlextCliUtilitiesPptxSerializer",
    "FlextCliUtilitiesPptxTypes",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._reader": ("FlextCliUtilitiesPptxReader",),
            "._renderer": ("FlextCliUtilitiesPptxRenderer",),
            "._serializer": ("FlextCliUtilitiesPptxSerializer",),
            "._types": ("FlextCliUtilitiesPptxTypes",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
