# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Logging Config Parts package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .logging_config_part_03 import FlextUtilitiesLoggingConfig
__all__: tuple[str, ...] = ("FlextUtilitiesLoggingConfig",)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({".logging_config_part_03": ("FlextUtilitiesLoggingConfig",)}),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
