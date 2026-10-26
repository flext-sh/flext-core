# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Beartype package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._alias_visitor import (
        FlextUtilitiesBeartypeAliasVisitor,
    )
    from flext_core._utilities._beartype._class_visitor.alias_rules import (
        alias_first_violation,
    )
    from flext_core._utilities._beartype._class_visitor.visitor import (
        FlextUtilitiesBeartypeClassVisitorPart03,
    )
    from flext_core._utilities._beartype._class_visitor.visitor_base import (
        FlextUtilitiesBeartypeClassVisitorPart01,
    )
    from flext_core._utilities._beartype._class_visitor.visitor_checks import (
        FlextUtilitiesBeartypeClassVisitorPart02,
    )
    from flext_core._utilities._beartype._field_union_scan import (
        FlextUtilitiesBeartypeFieldUnionScan,
    )
    from flext_core._utilities._beartype._helpers.helpers_part_01 import (
        FlextUtilitiesBeartypeHelpersPart01,
    )
    from flext_core._utilities._beartype._helpers.helpers_part_02 import (
        FlextUtilitiesBeartypeHelpersPart02,
    )
    from flext_core._utilities._beartype._helpers.helpers_part_03 import (
        FlextUtilitiesBeartypeHelpersPart03,
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
    "FlextUtilitiesBeartypeClassVisitorPart01",
    "FlextUtilitiesBeartypeClassVisitorPart02",
    "FlextUtilitiesBeartypeClassVisitorPart03",
    "FlextUtilitiesBeartypeDeprecatedVisitor",
    "FlextUtilitiesBeartypeFieldUnionScan",
    "FlextUtilitiesBeartypeFieldVisitor",
    "FlextUtilitiesBeartypeHelpers",
    "FlextUtilitiesBeartypeHelpersPart01",
    "FlextUtilitiesBeartypeHelpersPart02",
    "FlextUtilitiesBeartypeHelpersPart03",
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
    "alias_first_violation",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeAliasVisitor": "._alias_visitor",
        "FlextUtilitiesBeartypeAttrVisitor": ".attr_visitor",
        "FlextUtilitiesBeartypeClassVisitor": ".class_visitor",
        "FlextUtilitiesBeartypeClassVisitorPart01": "._class_visitor.visitor_base",
        "FlextUtilitiesBeartypeClassVisitorPart02": "._class_visitor.visitor_checks",
        "FlextUtilitiesBeartypeClassVisitorPart03": "._class_visitor.visitor",
        "FlextUtilitiesBeartypeDeprecatedVisitor": ".deprecated_visitor",
        "FlextUtilitiesBeartypeFieldUnionScan": "._field_union_scan",
        "FlextUtilitiesBeartypeFieldVisitor": ".field_visitor",
        "FlextUtilitiesBeartypeHelpers": ".helpers",
        "FlextUtilitiesBeartypeHelpersPart01": "._helpers.helpers_part_01",
        "FlextUtilitiesBeartypeHelpersPart02": "._helpers.helpers_part_02",
        "FlextUtilitiesBeartypeHelpersPart03": "._helpers.helpers_part_03",
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
        "alias_first_violation": "._class_visitor.alias_rules",
    }),
    public_exports=__all__,
)
