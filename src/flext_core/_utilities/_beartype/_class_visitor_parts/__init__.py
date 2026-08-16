# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype. Class Visitor Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _parts as _parts
    from .class_visitor_part_03 import FlextUtilitiesBeartypeClassVisitor
__all__: tuple[str, ...] = ("FlextUtilitiesBeartypeClassVisitor", "_parts")

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType(
        build_lazy_import_map(
            MappingProxyType({
                "._parts": ("_parts",),
                ".class_visitor_part_03": ("FlextUtilitiesBeartypeClassVisitor",),
            }),
            alias_groups=MappingProxyType({}),
            sort_keys=False,
        )
    ),
    public_exports=__all__,
)
