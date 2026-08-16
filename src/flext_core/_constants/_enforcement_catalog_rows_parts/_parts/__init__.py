# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Enforcement Catalog Rows Parts. Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flextconstantsenforcementcatalogrows_part_01_a import INFRA_DETECTOR_ROWS_CORE
    from .flextconstantsenforcementcatalogrows_part_01_b import (
        INFRA_DETECTOR_ROWS_PATTERNS,
    )

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    ".flextconstantsenforcementcatalogrows_part_01_a": ("INFRA_DETECTOR_ROWS_CORE",),
    ".flextconstantsenforcementcatalogrows_part_01_b": (
        "INFRA_DETECTOR_ROWS_PATTERNS",
    ),
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
)

__all__: tuple[str, ...] = ("INFRA_DETECTOR_ROWS_CORE", "INFRA_DETECTOR_ROWS_PATTERNS")

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
