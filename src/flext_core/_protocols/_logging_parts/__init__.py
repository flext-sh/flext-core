# AUTO-GENERATED FILE — Regenerate with: make gen
"""Logging Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._protocols._logging_parts.flextprotocolslogging_part_03 import (
        FlextProtocolsLogging as FlextProtocolsLogging,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextprotocolslogging_part_03": ("FlextProtocolsLogging",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
