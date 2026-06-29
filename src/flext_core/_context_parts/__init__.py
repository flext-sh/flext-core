# AUTO-GENERATED FILE — Regenerate with: make gen
"""Context Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._context_parts.flextcontext_part_02 import (
        FlextContext as FlextContext,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextcontext_part_02": ("FlextContext",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
