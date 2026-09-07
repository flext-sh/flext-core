# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Parser Targets Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .parser_targets_part_02 import FlextUtilitiesParserTargets
__all__: tuple[str, ...] = ("FlextUtilitiesParserTargets",)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({".parser_targets_part_02": ("FlextUtilitiesParserTargets",)}),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
