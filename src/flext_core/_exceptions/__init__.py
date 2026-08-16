# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _base_parts as _base_parts
    from . import _factories_parts as _factories_parts
    from ._base_parts.flextexceptionsbase_part_01 import FlextBaseErrorMetadataMixin
    from ._base_parts.flextexceptionsbase_part_02 import FlextBaseErrorStateMixin
    from .base import FlextExceptionsBase
    from .factories import FlextExceptionsFactories
    from .helpers import FlextExceptionsHelpers
    from .metrics import FlextExceptionsMetrics
    from .template import FlextExceptionsTemplate
    from .types import FlextExceptionsTypes

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    "._base_parts": ("_base_parts",),
    "._base_parts.flextexceptionsbase_part_01": ("FlextBaseErrorMetadataMixin",),
    "._base_parts.flextexceptionsbase_part_02": ("FlextBaseErrorStateMixin",),
    "._factories_parts": ("_factories_parts",),
    ".base": ("FlextExceptionsBase",),
    ".factories": ("FlextExceptionsFactories",),
    ".helpers": ("FlextExceptionsHelpers",),
    ".metrics": ("FlextExceptionsMetrics",),
    ".template": ("FlextExceptionsTemplate",),
    ".types": ("FlextExceptionsTypes",),
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
)

__all__: tuple[str, ...] = (
    "FlextBaseErrorMetadataMixin",
    "FlextBaseErrorStateMixin",
    "FlextExceptionsBase",
    "FlextExceptionsFactories",
    "FlextExceptionsHelpers",
    "FlextExceptionsMetrics",
    "FlextExceptionsTemplate",
    "FlextExceptionsTypes",
    "_base_parts",
    "_factories_parts",
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
