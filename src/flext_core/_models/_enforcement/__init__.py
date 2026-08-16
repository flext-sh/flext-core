# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Enforcement package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._base import EnforcementModelBase, FlextModelsEnforcementBase
    from ._catalog import FlextModelsEnforcementCatalog
    from ._params import FlextModelsEnforcementParams
    from ._sources import FlextModelsEnforcementSources
__all__: tuple[str, ...] = (
    "EnforcementModelBase",
    "FlextModelsEnforcementBase",
    "FlextModelsEnforcementCatalog",
    "FlextModelsEnforcementParams",
    "FlextModelsEnforcementSources",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType(
        build_lazy_import_map(
            MappingProxyType({
                "._base": ("EnforcementModelBase", "FlextModelsEnforcementBase"),
                "._catalog": ("FlextModelsEnforcementCatalog",),
                "._params": ("FlextModelsEnforcementParams",),
                "._sources": ("FlextModelsEnforcementSources",),
            }),
            alias_groups=MappingProxyType({}),
            sort_keys=False,
        )
    ),
    public_exports=__all__,
)
