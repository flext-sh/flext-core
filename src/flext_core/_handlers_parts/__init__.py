# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Handlers Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flexthandlers_part_07 import FlextHandlers
__all__: tuple[str, ...] = ("FlextHandlers",)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({".flexthandlers_part_07": ("FlextHandlers",)}),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
