# AUTO-GENERATED FILE — Regenerate with: make gen
"""Context Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._context_parts.flextprotocolscontext_part_03 import (
        FlextProtocolsContext as FlextProtocolsContext,
    )
_LAZY_IMPORTS = build_lazy_import_map({
    ".flextprotocolscontext_part_03": ("FlextProtocolsContext",)
})


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
