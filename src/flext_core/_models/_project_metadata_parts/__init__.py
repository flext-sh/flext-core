# AUTO-GENERATED FILE — Regenerate with: make gen
"""Project Metadata Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
        FlextModelsProjectMetadata as FlextModelsProjectMetadata,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextmodelsprojectmetadata_part_04": ("FlextModelsProjectMetadata",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
