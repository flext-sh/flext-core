# AUTO-GENERATED FILE — Regenerate with: make gen
"""Context package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._context.__scope_parts.flextmodelscontextscope_part_03 import (
        FlextModelsContextScope as FlextModelsContextScope,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".__scope_parts": ("__scope_parts",),
        ".__scope_parts.flextmodelscontextscope_part_03": ("FlextModelsContextScope",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
