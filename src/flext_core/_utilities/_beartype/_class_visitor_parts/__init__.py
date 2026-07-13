# AUTO-GENERATED FILE — Regenerate with: make gen
"""Class Visitor Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._class_visitor_parts.class_visitor_part_03 import (
        FlextUtilitiesBeartypeClassVisitor as FlextUtilitiesBeartypeClassVisitor,
    )
_LAZY_IMPORTS = build_lazy_import_map({
    "._parts": ("_parts",),
    ".class_visitor_part_03": ("FlextUtilitiesBeartypeClassVisitor",),
})


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
