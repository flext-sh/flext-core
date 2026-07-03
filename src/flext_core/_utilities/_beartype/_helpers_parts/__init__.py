# AUTO-GENERATED FILE — Regenerate with: make gen
"""Helpers Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._helpers_parts.helpers_part_03 import (
        FlextUtilitiesBeartypeHelpers as FlextUtilitiesBeartypeHelpers,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".helpers_part_03": ("FlextUtilitiesBeartypeHelpers",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
