# AUTO-GENERATED FILE — Regenerate with: make gen
"""Project Metadata Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._project_metadata_parts.project_metadata_part_04 import (
        FlextUtilitiesProjectMetadata,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".project_metadata_part_04": ("FlextUtilitiesProjectMetadata",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
