# AUTO-GENERATED FILE — Regenerate with: make gen
"""Logging Context Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._utilities._logging_context_parts.logging_context_part_02 import (
        FlextUtilitiesLoggingContext as FlextUtilitiesLoggingContext,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".logging_context_part_02": ("FlextUtilitiesLoggingContext",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
