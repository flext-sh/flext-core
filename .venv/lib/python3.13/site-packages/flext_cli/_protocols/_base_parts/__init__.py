# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Protocols. Base Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .flextcliprotocolsbase_part_05 import FlextCliProtocolsBase
__all__: tuple[str, ...] = ("FlextCliProtocolsBase",)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".flextcliprotocolsbase_part_05": ("FlextCliProtocolsBase",)
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
