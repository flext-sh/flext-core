# AUTO-GENERATED FILE — Regenerate with: make gen
"""Beartype package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._alias_visitor import (
        FlextUtilitiesBeartypeAliasVisitor as FlextUtilitiesBeartypeAliasVisitor,
    )
    from flext_core._utilities._beartype._class_visitor_parts.class_visitor_part_03 import (
        FlextUtilitiesBeartypeClassVisitor as FlextUtilitiesBeartypeClassVisitor,
    )
    from flext_core._utilities._beartype._helpers_parts.helpers_part_03 import (
        FlextUtilitiesBeartypeHelpers as FlextUtilitiesBeartypeHelpers,
    )
    from flext_core._utilities._beartype._library_visitor import (
        FlextUtilitiesBeartypeLibraryVisitor as FlextUtilitiesBeartypeLibraryVisitor,
    )
    from flext_core._utilities._beartype.attr_visitor import (
        FlextUtilitiesBeartypeAttrVisitor as FlextUtilitiesBeartypeAttrVisitor,
    )
    from flext_core._utilities._beartype.deprecated_visitor import (
        FlextUtilitiesBeartypeDeprecatedVisitor as FlextUtilitiesBeartypeDeprecatedVisitor,
    )
    from flext_core._utilities._beartype.field_visitor import (
        FlextUtilitiesBeartypeFieldVisitor as FlextUtilitiesBeartypeFieldVisitor,
    )
    from flext_core._utilities._beartype.import_visitor import (
        FlextUtilitiesBeartypeImportVisitor as FlextUtilitiesBeartypeImportVisitor,
    )
    from flext_core._utilities._beartype.method_visitor import (
        FlextUtilitiesBeartypeMethodVisitor as FlextUtilitiesBeartypeMethodVisitor,
    )
    from flext_core._utilities._beartype.module_visitor import (
        FlextUtilitiesBeartypeModuleVisitor as FlextUtilitiesBeartypeModuleVisitor,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._alias_visitor": ("FlextUtilitiesBeartypeAliasVisitor",),
        "._class_visitor_parts": ("_class_visitor_parts",),
        "._class_visitor_parts.class_visitor_part_03": (
            "FlextUtilitiesBeartypeClassVisitor",
        ),
        "._helpers_parts": ("_helpers_parts",),
        "._helpers_parts.helpers_part_03": ("FlextUtilitiesBeartypeHelpers",),
        "._library_visitor": ("FlextUtilitiesBeartypeLibraryVisitor",),
        ".attr_visitor": ("FlextUtilitiesBeartypeAttrVisitor",),
        ".deprecated_visitor": ("FlextUtilitiesBeartypeDeprecatedVisitor",),
        ".field_visitor": ("FlextUtilitiesBeartypeFieldVisitor",),
        ".import_visitor": ("FlextUtilitiesBeartypeImportVisitor",),
        ".method_visitor": ("FlextUtilitiesBeartypeMethodVisitor",),
        ".module_visitor": ("FlextUtilitiesBeartypeModuleVisitor",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
