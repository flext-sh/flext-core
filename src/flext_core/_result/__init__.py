# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Result package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .base import FlextResultBase
    from .behavior import FlextResultBehavior
    from .composition import FlextResultComposition
    from .construction import FlextResultConstruction
    from .transforms import FlextResultTransforms
    from .unwrap import FlextResultUnwrap
__all__: tuple[str, ...] = (
    "FlextResultBase",
    "FlextResultBehavior",
    "FlextResultComposition",
    "FlextResultConstruction",
    "FlextResultTransforms",
    "FlextResultUnwrap",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".base": ("FlextResultBase",),
            ".behavior": ("FlextResultBehavior",),
            ".composition": ("FlextResultComposition",),
            ".construction": ("FlextResultConstruction",),
            ".transforms": ("FlextResultTransforms",),
            ".unwrap": ("FlextResultUnwrap",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
