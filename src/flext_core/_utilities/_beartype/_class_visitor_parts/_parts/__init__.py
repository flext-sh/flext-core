# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype. Class Visitor Parts. Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import class_visitor_part_02_01 as class_visitor_part_02_01
    from . import class_visitor_part_02_02 as class_visitor_part_02_02
__all__: tuple[str, ...] = ("class_visitor_part_02_01", "class_visitor_part_02_02")

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType(
        build_lazy_import_map(
            MappingProxyType({
                ".class_visitor_part_02_01": ("class_visitor_part_02_01",),
                ".class_visitor_part_02_02": ("class_visitor_part_02_02",),
            }),
            alias_groups=MappingProxyType({}),
            sort_keys=False,
        )
    ),
    public_exports=__all__,
)
