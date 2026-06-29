# AUTO-GENERATED FILE — Regenerate with: make gen
"""Container Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._protocols._container_parts.flextprotocolscontainer_part_03 import (
        FlextProtocolsContainer as FlextProtocolsContainer,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextprotocolscontainer_part_03": ("FlextProtocolsContainer",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
