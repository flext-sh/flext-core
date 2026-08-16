# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Checker Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .checker_part_03 import FlextUtilitiesChecker
__all__: tuple[str, ...] = ("FlextUtilitiesChecker",)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType(
        build_lazy_import_map(
            MappingProxyType({".checker_part_03": ("FlextUtilitiesChecker",)}),
            alias_groups=MappingProxyType({}),
            sort_keys=False,
        )
    ),
    public_exports=__all__,
)
