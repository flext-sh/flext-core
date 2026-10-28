"""Extracted facade class for beartype.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._beartype import (
    FlextBeartypeAlias,
    FlextUtilitiesBeartypeAttrVisitor,
    FlextUtilitiesBeartypeClassVisitor,
    FlextUtilitiesBeartypeDeprecatedVisitor,
    FlextUtilitiesBeartypeFieldVisitor,
    FlextUtilitiesBeartypeHelpers,
    FlextUtilitiesBeartypeImportVisitor,
    FlextUtilitiesBeartypeMethodVisitor,
    FlextUtilitiesBeartypeModuleSource,
    FlextUtilitiesBeartypeModuleVisitor,
    FlextUtilitiesBeartypeTypeAliases,
)


class FlextUtilitiesBeartype(
    FlextBeartypeAlias,
    FlextUtilitiesBeartypeAttrVisitor,
    FlextUtilitiesBeartypeClassVisitor,
    FlextUtilitiesBeartypeDeprecatedVisitor,
    FlextUtilitiesBeartypeFieldVisitor,
    FlextUtilitiesBeartypeHelpers,
    FlextUtilitiesBeartypeImportVisitor,
    FlextUtilitiesBeartypeMethodVisitor,
    FlextUtilitiesBeartypeModuleSource,
    FlextUtilitiesBeartypeModuleVisitor,
    FlextUtilitiesBeartypeTypeAliases,
):
    """Facade for beartype."""


__all__: list[str] = ["FlextUtilitiesBeartype"]
