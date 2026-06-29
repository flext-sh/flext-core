# AUTO-GENERATED FILE — Regenerate with: make gen
"""Base Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._settings._base_parts.flextsettingsbase_part_02 import (
        FlextSettingsBase as FlextSettingsBase,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextsettingsbase_part_02": ("FlextSettingsBase",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
