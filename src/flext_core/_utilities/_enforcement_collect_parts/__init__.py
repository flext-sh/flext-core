# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement Collect Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._utilities._enforcement_collect_parts.enforcement_collect_part_02 import (
        FlextUtilitiesEnforcementCollect as FlextUtilitiesEnforcementCollect,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".enforcement_collect_part_02": ("FlextUtilitiesEnforcementCollect",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
