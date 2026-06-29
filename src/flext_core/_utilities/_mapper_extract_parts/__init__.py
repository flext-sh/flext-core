# AUTO-GENERATED FILE — Regenerate with: make gen
"""Mapper Extract Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._utilities._mapper_extract_parts.mapper_extract_part_02 import (
        FlextUtilitiesMapperExtract as FlextUtilitiesMapperExtract,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".mapper_extract_part_02": ("FlextUtilitiesMapperExtract",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
