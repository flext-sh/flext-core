# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Enforcement package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._base import EnforcementModelBase, FlextModelsEnforcementBase
    from ._catalog import FlextModelsEnforcementCatalog
    from ._inspection import FlextModelsEnforcementInspection
    from ._params import FlextModelsEnforcementParams
    from ._resolution import FlextModelsEnforcementResolution
    from ._sources import FlextModelsEnforcementSources


__all__: tuple[str, ...] = (
    "EnforcementModelBase",
    "FlextModelsEnforcementBase",
    "FlextModelsEnforcementCatalog",
    "FlextModelsEnforcementInspection",
    "FlextModelsEnforcementParams",
    "FlextModelsEnforcementResolution",
    "FlextModelsEnforcementSources",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._base": ("EnforcementModelBase", "FlextModelsEnforcementBase"),
            "._catalog": ("FlextModelsEnforcementCatalog",),
            "._inspection": ("FlextModelsEnforcementInspection",),
            "._params": ("FlextModelsEnforcementParams",),
            "._resolution": ("FlextModelsEnforcementResolution",),
            "._sources": ("FlextModelsEnforcementSources",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
