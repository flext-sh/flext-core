# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _class_visitor_parts as _class_visitor_parts
    from . import _helpers_parts as _helpers_parts
    from ._alias_visitor import FlextUtilitiesBeartypeAliasVisitor
    from ._library_visitor import FlextUtilitiesBeartypeLibraryVisitor
    from .attr_visitor import FlextUtilitiesBeartypeAttrVisitor
    from .class_visitor import FlextUtilitiesBeartypeClassVisitor
    from .deprecated_visitor import FlextUtilitiesBeartypeDeprecatedVisitor
    from .field_visitor import FlextUtilitiesBeartypeFieldVisitor
    from .helpers import FlextUtilitiesBeartypeHelpers
    from .import_visitor import FlextUtilitiesBeartypeImportVisitor
    from .method_visitor import FlextUtilitiesBeartypeMethodVisitor
    from .module_visitor import FlextUtilitiesBeartypeModuleVisitor
__all__: tuple[str, ...] = (
    "FlextUtilitiesBeartypeAliasVisitor",
    "FlextUtilitiesBeartypeAttrVisitor",
    "FlextUtilitiesBeartypeClassVisitor",
    "FlextUtilitiesBeartypeDeprecatedVisitor",
    "FlextUtilitiesBeartypeFieldVisitor",
    "FlextUtilitiesBeartypeHelpers",
    "FlextUtilitiesBeartypeImportVisitor",
    "FlextUtilitiesBeartypeLibraryVisitor",
    "FlextUtilitiesBeartypeMethodVisitor",
    "FlextUtilitiesBeartypeModuleVisitor",
    "_class_visitor_parts",
    "_helpers_parts",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType(
        build_lazy_import_map(
            MappingProxyType({
                "._alias_visitor": ("FlextUtilitiesBeartypeAliasVisitor",),
                "._class_visitor_parts": ("_class_visitor_parts",),
                "._helpers_parts": ("_helpers_parts",),
                "._library_visitor": ("FlextUtilitiesBeartypeLibraryVisitor",),
                ".attr_visitor": ("FlextUtilitiesBeartypeAttrVisitor",),
                ".class_visitor": ("FlextUtilitiesBeartypeClassVisitor",),
                ".deprecated_visitor": ("FlextUtilitiesBeartypeDeprecatedVisitor",),
                ".field_visitor": ("FlextUtilitiesBeartypeFieldVisitor",),
                ".helpers": ("FlextUtilitiesBeartypeHelpers",),
                ".import_visitor": ("FlextUtilitiesBeartypeImportVisitor",),
                ".method_visitor": ("FlextUtilitiesBeartypeMethodVisitor",),
                ".module_visitor": ("FlextUtilitiesBeartypeModuleVisitor",),
            }),
            alias_groups=MappingProxyType({}),
            sort_keys=False,
        )
    ),
    public_exports=__all__,
)
