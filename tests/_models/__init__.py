# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from _models.mixins import TestsFlextCoreModelsMixins
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".mixins": ("TestsFlextCoreModelsMixins",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__ = [
    "TestsFlextCoreModelsMixins",
]
