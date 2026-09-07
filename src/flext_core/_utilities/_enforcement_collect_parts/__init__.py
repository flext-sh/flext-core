# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Enforcement Collect Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .enforcement_collect_part_02 import FlextUtilitiesEnforcementCollect
__all__: tuple[str, ...] = ("FlextUtilitiesEnforcementCollect",)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".enforcement_collect_part_02": ("FlextUtilitiesEnforcementCollect",)
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
