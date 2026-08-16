# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype. Class Visitor Parts. Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import class_visitor_part_02_01 as class_visitor_part_02_01
    from . import class_visitor_part_02_02 as class_visitor_part_02_02

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    ".class_visitor_part_02_01": ("class_visitor_part_02_01",),
    ".class_visitor_part_02_02": ("class_visitor_part_02_02",),
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
)

__all__: tuple[str, ...] = ("class_visitor_part_02_01", "class_visitor_part_02_02")

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
