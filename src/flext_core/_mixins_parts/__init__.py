# AUTO-GENERATED FILE — Regenerate with: make gen
"""Mixins Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._mixins_parts.flextmixins_part_02 import FlextMixins as FlextMixins
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextmixins_part_02": ("FlextMixins",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
