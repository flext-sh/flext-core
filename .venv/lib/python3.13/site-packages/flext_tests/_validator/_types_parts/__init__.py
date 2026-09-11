# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Validator. Types Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .types_part_02 import FlextValidatorTypes
__all__: tuple[str, ...] = ("FlextValidatorTypes",)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({".types_part_02": ("FlextValidatorTypes",)}),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
