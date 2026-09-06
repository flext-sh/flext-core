# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions. Base Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flextexceptionsbase_part_01 import FlextBaseErrorMetadataMixin
    from .flextexceptionsbase_part_02 import FlextBaseErrorStateMixin
    from .flextexceptionsbase_part_03 import FlextExceptionsBase
__all__: tuple[str, ...] = (
    "FlextBaseErrorMetadataMixin",
    "FlextBaseErrorStateMixin",
    "FlextExceptionsBase",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".flextexceptionsbase_part_01": ("FlextBaseErrorMetadataMixin",),
            ".flextexceptionsbase_part_02": ("FlextBaseErrorStateMixin",),
            ".flextexceptionsbase_part_03": ("FlextExceptionsBase",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
