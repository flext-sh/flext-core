# AUTO-GENERATED FILE — Regenerate with: make gen
"""Generators Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._utilities._generators_parts.generators_part_02 import (
        FlextUtilitiesGenerators as FlextUtilitiesGenerators,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".generators_part_02": ("FlextUtilitiesGenerators",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
