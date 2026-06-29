# AUTO-GENERATED FILE — Regenerate with: make gen
"""Base Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._models._base_parts.flextmodelsbase_part_03 import (
        FlextModelsBase as FlextModelsBase,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextmodelsbase_part_03": ("FlextModelsBase",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
