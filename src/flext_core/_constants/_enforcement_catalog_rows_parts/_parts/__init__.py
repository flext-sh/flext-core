# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Constants. Enforcement Catalog Rows Parts. Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flextconstantsenforcementcatalogrows_part_01_a import INFRA_DETECTOR_ROWS_CORE
    from .flextconstantsenforcementcatalogrows_part_01_b import (
        INFRA_DETECTOR_ROWS_PATTERNS,
    )
__all__: tuple[str, ...] = ("INFRA_DETECTOR_ROWS_CORE", "INFRA_DETECTOR_ROWS_PATTERNS")

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".flextconstantsenforcementcatalogrows_part_01_a": (
                "INFRA_DETECTOR_ROWS_CORE",
            ),
            ".flextconstantsenforcementcatalogrows_part_01_b": (
                "INFRA_DETECTOR_ROWS_PATTERNS",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
