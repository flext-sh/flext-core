# AUTO-GENERATED FILE — Regenerate with: make gen
"""Factories Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions._factories_parts.flextexceptionsfactories_part_04 import (
        FlextExceptionsFactories as FlextExceptionsFactories,
    )
_LAZY_IMPORTS = build_lazy_import_map({
    ".flextexceptionsfactories_part_04": ("FlextExceptionsFactories",)
})


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
