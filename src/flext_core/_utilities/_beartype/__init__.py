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
    from flext_core._utilities._beartype import _class_visitor_parts, _helpers_parts
    from flext_core._utilities._beartype._alias_visitor import (
        FlextUtilitiesBeartypeAliasVisitor,
    )
    from flext_core._utilities._beartype._class_visitor_parts._parts.class_visitor_part_02_01 import (
        alias_first_violation,
    )
    from flext_core._utilities._beartype._class_visitor_parts._parts.class_visitor_part_02_02 import (
        redundant_inner_violation,
        self_ref_violation,
    )
    from flext_core._utilities._beartype._library_visitor import (
        FlextUtilitiesBeartypeLibraryVisitor,
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
    "FlextUtilitiesBeartypeFieldVisitor",
    "FlextUtilitiesBeartypeHelpers",
    "FlextUtilitiesBeartypeImportVisitor",
    "FlextUtilitiesBeartypeLibraryVisitor",
    "FlextUtilitiesBeartypeMethodVisitor",
    "FlextUtilitiesBeartypeModuleSource",
    "FlextUtilitiesBeartypeModuleVisitor",
    "FlextUtilitiesBeartypeTypeAliases",
    "_class_visitor_parts",
    "_helpers_parts",
    "alias_first_violation",
    "redundant_inner_violation",
    "self_ref_violation",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesBeartypeAliasVisitor": "._alias_visitor",
        "FlextUtilitiesBeartypeAttrVisitor": ".attr_visitor",
        "FlextUtilitiesBeartypeClassVisitor": ".class_visitor",
        "FlextUtilitiesBeartypeDeprecatedVisitor": ".deprecated_visitor",
        "FlextUtilitiesBeartypeFieldVisitor": ".field_visitor",
        "FlextUtilitiesBeartypeHelpers": ".helpers",
        "FlextUtilitiesBeartypeImportVisitor": ".import_visitor",
        "FlextUtilitiesBeartypeLibraryVisitor": "._library_visitor",
        "FlextUtilitiesBeartypeMethodVisitor": ".method_visitor",
        "FlextUtilitiesBeartypeModuleSource": ".module_source",
        "FlextUtilitiesBeartypeModuleVisitor": ".module_visitor",
        "FlextUtilitiesBeartypeTypeAliases": ".type_aliases",
        "_class_visitor_parts": "._class_visitor_parts",
        "_helpers_parts": "._helpers_parts",
        "alias_first_violation": (
            "._class_visitor_parts._parts.class_visitor_part_02_01"
        ),
        "redundant_inner_violation": (
            "._class_visitor_parts._parts.class_visitor_part_02_02"
        ),
        "self_ref_violation": "._class_visitor_parts._parts.class_visitor_part_02_02",
    }),
    public_exports=__all__,
)
