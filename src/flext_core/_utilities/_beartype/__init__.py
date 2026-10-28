# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for beartype.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype.alias import FlextBeartypeAlias
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
    "FlextBeartypeAlias",
    "FlextUtilitiesBeartypeAttrVisitor",
    "FlextUtilitiesBeartypeClassVisitor",
    "FlextUtilitiesBeartypeDeprecatedVisitor",
    "FlextUtilitiesBeartypeFieldVisitor",
    "FlextUtilitiesBeartypeHelpers",
    "FlextUtilitiesBeartypeImportVisitor",
    "FlextUtilitiesBeartypeMethodVisitor",
    "FlextUtilitiesBeartypeModuleSource",
    "FlextUtilitiesBeartypeModuleVisitor",
    "FlextUtilitiesBeartypeTypeAliases",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextBeartypeAlias": ".alias",
        "FlextUtilitiesBeartypeAttrVisitor": ".attr_visitor",
        "FlextUtilitiesBeartypeClassVisitor": ".class_visitor",
        "FlextUtilitiesBeartypeDeprecatedVisitor": ".deprecated_visitor",
        "FlextUtilitiesBeartypeFieldVisitor": ".field_visitor",
        "FlextUtilitiesBeartypeHelpers": ".helpers",
        "FlextUtilitiesBeartypeImportVisitor": ".import_visitor",
        "FlextUtilitiesBeartypeMethodVisitor": ".method_visitor",
        "FlextUtilitiesBeartypeModuleSource": ".module_source",
        "FlextUtilitiesBeartypeModuleVisitor": ".module_visitor",
        "FlextUtilitiesBeartypeTypeAliases": ".type_aliases",
    }),
    public_exports=__all__,
)
