# AUTO-GENERATED FILE — Regenerate with: make gen
"""Container Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._container_parts.flextmodelscontainer_part_04 import (
        FlextModelsContainer as FlextModelsContainer,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextmodelscontainer_part_04": ("FlextModelsContainer",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
