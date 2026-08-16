# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions. Base Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flextexceptionsbase_part_01 import FlextBaseErrorMetadataMixin
    from .flextexceptionsbase_part_02 import FlextBaseErrorStateMixin
    from .flextexceptionsbase_part_03 import FlextExceptionsBase

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    ".flextexceptionsbase_part_01": ("FlextBaseErrorMetadataMixin",),
    ".flextexceptionsbase_part_02": ("FlextBaseErrorStateMixin",),
    ".flextexceptionsbase_part_03": ("FlextExceptionsBase",),
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
)

__all__: tuple[str, ...] = (
    "FlextBaseErrorMetadataMixin",
    "FlextBaseErrorStateMixin",
    "FlextExceptionsBase",
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
