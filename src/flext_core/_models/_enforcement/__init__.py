# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._enforcement._base import (
        EnforcementModelBase,
        FlextModelsEnforcementBase,
    )
    from flext_core._models._enforcement._catalog import FlextModelsEnforcementCatalog
    from flext_core._models._enforcement._params import FlextModelsEnforcementParams
    from flext_core._models._enforcement._sources import FlextModelsEnforcementSources
_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._base": (
            "EnforcementModelBase",
            "FlextModelsEnforcementBase",
        ),
        "._catalog": ("FlextModelsEnforcementCatalog",),
        "._params": ("FlextModelsEnforcementParams",),
        "._sources": ("FlextModelsEnforcementSources",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
