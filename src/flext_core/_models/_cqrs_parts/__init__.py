# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Cqrs Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flextmodelscqrs_part_01 import CqrsPagination
__all__: tuple[str, ...] = ("CqrsPagination",)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({".flextmodelscqrs_part_01": ("CqrsPagination",)}),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
