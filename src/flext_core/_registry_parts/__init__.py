# AUTO-GENERATED FILE — Regenerate with: make gen
"""Registry Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._registry_parts.flextregistry_part_04 import (
        FlextRegistry as FlextRegistry,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextregistry_part_04": ("FlextRegistry",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
