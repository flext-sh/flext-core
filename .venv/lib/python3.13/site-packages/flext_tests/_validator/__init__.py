# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Validator package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _types_parts as _types_parts
    from .bypass import FlextValidatorBypass
    from .imports import FlextValidatorImports
    from .layer import FlextValidatorLayer
    from .markdown import FlextValidatorMarkdown
    from .settings import FlextValidatorSettings
    from .tests import FlextValidatorTests
    from .types import FlextValidatorTypes
__all__: tuple[str, ...] = (
    "FlextValidatorBypass",
    "FlextValidatorImports",
    "FlextValidatorLayer",
    "FlextValidatorMarkdown",
    "FlextValidatorSettings",
    "FlextValidatorTests",
    "FlextValidatorTypes",
    "_types_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._types_parts": ("_types_parts",),
            ".bypass": ("FlextValidatorBypass",),
            ".imports": ("FlextValidatorImports",),
            ".layer": ("FlextValidatorLayer",),
            ".markdown": ("FlextValidatorMarkdown",),
            ".settings": ("FlextValidatorSettings",),
            ".tests": ("FlextValidatorTests",),
            ".types": ("FlextValidatorTypes",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
