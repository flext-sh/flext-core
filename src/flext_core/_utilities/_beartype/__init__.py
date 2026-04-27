"""Beartype-driven enforcement engine mixins — data-driven rule visitors.

Visitors are organized by domain via MRO composition:
- FlextUtilitiesBeartypeFieldVisitor: field annotation governance
- FlextUtilitiesBeartypeAttrVisitor: class-attribute checks
- FlextUtilitiesBeartypeMethodVisitor: method shape + static requirements
- FlextUtilitiesBeartypeClassVisitor: class placement + MRO + protocol trees
- FlextUtilitiesBeartypeModuleVisitor: module-level introspection
- FlextUtilitiesBeartypeImportVisitor: import discipline checks
- FlextUtilitiesBeartypeDeprecatedVisitor: bytecode-based deprecated syntax

Root engine (FlextUtilitiesBeartypeEngine) composes via MRO for
canonical dispatch via apply(kind, params, *args).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._beartype.field_visitor import (
    FlextUtilitiesBeartypeFieldVisitor,
)
from flext_core._utilities._beartype.attr_visitor import (
    FlextUtilitiesBeartypeAttrVisitor,
)
from flext_core._utilities._beartype.method_visitor import (
    FlextUtilitiesBeartypeMethodVisitor,
)
from flext_core._utilities._beartype.class_visitor import (
    FlextUtilitiesBeartypeClassVisitor,
)
from flext_core._utilities._beartype.module_visitor import (
    FlextUtilitiesBeartypeModuleVisitor,
)
from flext_core._utilities._beartype.import_visitor import (
    FlextUtilitiesBeartypeImportVisitor,
)
from flext_core._utilities._beartype.deprecated_visitor import (
    FlextUtilitiesBeartypeDeprecatedVisitor,
)

__all__ = [
    "FlextUtilitiesBeartypeFieldVisitor",
    "FlextUtilitiesBeartypeAttrVisitor",
    "FlextUtilitiesBeartypeMethodVisitor",
    "FlextUtilitiesBeartypeClassVisitor",
    "FlextUtilitiesBeartypeModuleVisitor",
    "FlextUtilitiesBeartypeImportVisitor",
    "FlextUtilitiesBeartypeDeprecatedVisitor",
]
