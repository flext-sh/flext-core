# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra. Enforcement package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .collection_base import (
        FlextInfraEnforcementCollectionBase,
        FlextInfraEnforcementEvaluation,
    )
    from .collection_sources import FlextInfraEnforcementSourceCollectors
    from .engine import FlextInfraEnforcementEngine
    from .metadata import FlextInfraEnforcementMetadata
    from .selection import FlextInfraEnforcementSelection
__all__: tuple[str, ...] = (
    "FlextInfraEnforcementCollectionBase",
    "FlextInfraEnforcementEngine",
    "FlextInfraEnforcementEvaluation",
    "FlextInfraEnforcementMetadata",
    "FlextInfraEnforcementSelection",
    "FlextInfraEnforcementSourceCollectors",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".collection_base": (
                "FlextInfraEnforcementCollectionBase",
                "FlextInfraEnforcementEvaluation",
            ),
            ".collection_sources": ("FlextInfraEnforcementSourceCollectors",),
            ".engine": ("FlextInfraEnforcementEngine",),
            ".metadata": ("FlextInfraEnforcementMetadata",),
            ".selection": ("FlextInfraEnforcementSelection",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
