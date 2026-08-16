# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Enforcement package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._base import EnforcementModelBase, FlextModelsEnforcementBase
    from ._catalog import FlextModelsEnforcementCatalog
    from ._params import FlextModelsEnforcementParams
    from ._sources import FlextModelsEnforcementSources

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    "._base": ("EnforcementModelBase", "FlextModelsEnforcementBase"),
    "._catalog": ("FlextModelsEnforcementCatalog",),
    "._params": ("FlextModelsEnforcementParams",),
    "._sources": ("FlextModelsEnforcementSources",),
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
)

__all__: tuple[str, ...] = (
    "EnforcementModelBase",
    "FlextModelsEnforcementBase",
    "FlextModelsEnforcementCatalog",
    "FlextModelsEnforcementParams",
    "FlextModelsEnforcementSources",
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
