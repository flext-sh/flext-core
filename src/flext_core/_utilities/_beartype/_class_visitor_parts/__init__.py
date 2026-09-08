# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype. Class Visitor Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _parts as _parts
    from ._parts.class_visitor_part_02_01 import alias_first_violation
    from ._parts.class_visitor_part_02_02 import (
        redundant_inner_violation,
        self_ref_violation,
    )
    from .class_visitor_part_03 import FlextUtilitiesBeartypeClassVisitor
__all__: tuple[str, ...] = (
    "FlextUtilitiesBeartypeClassVisitor",
    "_parts",
    "alias_first_violation",
    "redundant_inner_violation",
    "self_ref_violation",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._parts": ("_parts",),
            "._parts.class_visitor_part_02_01": ("alias_first_violation",),
            "._parts.class_visitor_part_02_02": (
                "redundant_inner_violation",
                "self_ref_violation",
            ),
            ".class_visitor_part_03": ("FlextUtilitiesBeartypeClassVisitor",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
