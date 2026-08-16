# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Result package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .base import FlextResultBase
    from .behavior import FlextResultBehavior
    from .composition import FlextResultComposition
    from .construction import FlextResultConstruction
    from .transforms import FlextResultTransforms
    from .unwrap import FlextResultUnwrap

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    ".base": ("FlextResultBase",),
    ".behavior": ("FlextResultBehavior",),
    ".composition": ("FlextResultComposition",),
    ".construction": ("FlextResultConstruction",),
    ".transforms": ("FlextResultTransforms",),
    ".unwrap": ("FlextResultUnwrap",),
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
)

__all__: tuple[str, ...] = (
    "FlextResultBase",
    "FlextResultBehavior",
    "FlextResultComposition",
    "FlextResultConstruction",
    "FlextResultTransforms",
    "FlextResultUnwrap",
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
