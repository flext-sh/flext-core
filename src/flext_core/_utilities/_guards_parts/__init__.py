# AUTO-GENERATED FILE — Regenerate with: make gen
"""Guards Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._guards_parts.guards_part_02 import FlextUtilitiesGuards
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".guards_part_02": ("FlextUtilitiesGuards",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
