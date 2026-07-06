# AUTO-GENERATED FILE — Regenerate with: make gen
"""Context Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._context_parts.flextcontext_part_02 import FlextContext
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextcontext_part_02": ("FlextContext",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
