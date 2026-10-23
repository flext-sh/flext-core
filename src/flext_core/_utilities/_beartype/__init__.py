# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype import (
        _class_visitor as _class_visitor,
        _helpers as _helpers,
    )
    from flext_core._utilities._beartype._alias_visitor import (
        FlextUtilitiesBeartypeAliasVisitor,
    )
    from flext_core._utilities._beartype._class_visitor import alias_first_violation
    from flext_core._utilities._beartype._field_union_scan import (
        FlextUtilitiesBeartypeFieldUnionScan,
    )
    from flext_core._utilities._beartype._import_blacklist_visitor import (
        FlextUtilitiesBeartypeImportBlacklistVisitor,
    )
    from flext_core._utilities._beartype._library_visitor import (
        FlextUtilitiesBeartypeLibraryVisitor,
    )
    from flext_core._utilities._beartype._module_source_deferred import (
        FlextUtilitiesBeartypeModuleSourceDeferred,
    )
    from flext_core._utilities._beartype._module_source_provenance import (
        FlextUtilitiesBeartypeModuleSourceProvenance,
    )
    from flext_core._utilities._beartype._module_source_scan import (
        FlextUtilitiesBeartypeModuleSourceScan,
    )
    from flext_core._utilities._beartype.attr_visitor import (
        FlextUtilitiesBeartypeAttrVisitor,
    )
    from flext_core._utilities._beartype.class_visitor import (
        FlextUtilitiesBeartypeClassVisitor,
    )
    from flext_core._utilities._beartype.deprecated_visitor import (
        FlextUtilitiesBeartypeDeprecatedVisitor,
    )
    from flext_core._utilities._beartype.field_visitor import (
        FlextUtilitiesBeartypeFieldVisitor,
    )
    from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers
    from flext_core._utilities._beartype.import_visitor import (
        FlextUtilitiesBeartypeImportVisitor,
    )
    from flext_core._utilities._beartype.method_visitor import (
        FlextUtilitiesBeartypeMethodVisitor,
    )
    from flext_core._utilities._beartype.module_source import (
        FlextUtilitiesBeartypeModuleSource,
    )
    from flext_core._utilities._beartype.module_visitor import (
        FlextUtilitiesBeartypeModuleVisitor,
    )
    from flext_core._utilities._beartype.type_aliases import (
        FlextUtilitiesBeartypeTypeAliases,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesBeartypeAliasVisitor",
    "FlextUtilitiesBeartypeAttrVisitor",
    "FlextUtilitiesBeartypeClassVisitor",
    "FlextUtilitiesBeartypeDeprecatedVisitor",
    "FlextUtilitiesBeartypeFieldUnionScan",
    "FlextUtilitiesBeartypeFieldVisitor",
    "FlextUtilitiesBeartypeHelpers",
    "FlextUtilitiesBeartypeImportBlacklistVisitor",
    "FlextUtilitiesBeartypeImportVisitor",
    "FlextUtilitiesBeartypeLibraryVisitor",
    "FlextUtilitiesBeartypeMethodVisitor",
    "FlextUtilitiesBeartypeModuleSource",
    "FlextUtilitiesBeartypeModuleSourceDeferred",
    "FlextUtilitiesBeartypeModuleSourceProvenance",
    "FlextUtilitiesBeartypeModuleSourceScan",
    "FlextUtilitiesBeartypeModuleVisitor",
    "FlextUtilitiesBeartypeTypeAliases",
    "_class_visitor",
    "_helpers",
    "alias_first_violation",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeAliasVisitor": "._alias_visitor",
        "FlextUtilitiesBeartypeAttrVisitor": ".attr_visitor",
        "FlextUtilitiesBeartypeClassVisitor": ".class_visitor",
        "FlextUtilitiesBeartypeDeprecatedVisitor": ".deprecated_visitor",
        "FlextUtilitiesBeartypeFieldUnionScan": "._field_union_scan",
        "FlextUtilitiesBeartypeFieldVisitor": ".field_visitor",
        "FlextUtilitiesBeartypeHelpers": ".helpers",
        "FlextUtilitiesBeartypeImportBlacklistVisitor": "._import_blacklist_visitor",
        "FlextUtilitiesBeartypeImportVisitor": ".import_visitor",
        "FlextUtilitiesBeartypeLibraryVisitor": "._library_visitor",
        "FlextUtilitiesBeartypeMethodVisitor": ".method_visitor",
        "FlextUtilitiesBeartypeModuleSource": ".module_source",
        "FlextUtilitiesBeartypeModuleSourceDeferred": "._module_source_deferred",
        "FlextUtilitiesBeartypeModuleSourceProvenance": "._module_source_provenance",
        "FlextUtilitiesBeartypeModuleSourceScan": "._module_source_scan",
        "FlextUtilitiesBeartypeModuleVisitor": ".module_visitor",
        "FlextUtilitiesBeartypeTypeAliases": ".type_aliases",
        "_class_visitor": "._class_visitor",
        "_helpers": "._helpers",
        "alias_first_violation": ("._class_visitor.class_visitor_part_02_01"),
    }),
    public_exports=__all__,
)[0]
