# AUTO-GENERATED FILE — Regenerate with: make gen
"""Project Metadata Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._utilities._project_metadata_parts.project_metadata_part_03 import (
        FlextUtilitiesProjectMetadata as FlextUtilitiesProjectMetadata,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".project_metadata_part_03": ("FlextUtilitiesProjectMetadata",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
