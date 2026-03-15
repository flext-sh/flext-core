# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Code generation services.

Provides code generation tools for workspace standardization,
including lazy-import init file generation (PEP 562).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_infra.codegen.census import FlextInfraCodegenCensus
    from flext_infra.codegen.constants_quality_gate import (
        FlextInfraCodegenConstantsQualityGate,
    )
    from flext_infra.codegen.fixer import FlextInfraCodegenFixer
    from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
    from flext_infra.codegen.py_typed import FlextInfraCodegenPyTyped
    from flext_infra.codegen.scaffolder import FlextInfraCodegenScaffolder
    from flext_infra.codegen.transforms import FlextInfraCodegenTransforms

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraCodegenCensus": ("flext_infra.codegen.census", "FlextInfraCodegenCensus"),
    "FlextInfraCodegenConstantsQualityGate": ("flext_infra.codegen.constants_quality_gate", "FlextInfraCodegenConstantsQualityGate"),
    "FlextInfraCodegenFixer": ("flext_infra.codegen.fixer", "FlextInfraCodegenFixer"),
    "FlextInfraCodegenLazyInit": ("flext_infra.codegen.lazy_init", "FlextInfraCodegenLazyInit"),
    "FlextInfraCodegenPyTyped": ("flext_infra.codegen.py_typed", "FlextInfraCodegenPyTyped"),
    "FlextInfraCodegenScaffolder": ("flext_infra.codegen.scaffolder", "FlextInfraCodegenScaffolder"),
    "FlextInfraCodegenTransforms": ("flext_infra.codegen.transforms", "FlextInfraCodegenTransforms"),
}

__all__ = [
    "FlextInfraCodegenCensus",
    "FlextInfraCodegenConstantsQualityGate",
    "FlextInfraCodegenFixer",
    "FlextInfraCodegenLazyInit",
    "FlextInfraCodegenPyTyped",
    "FlextInfraCodegenScaffolder",
    "FlextInfraCodegenTransforms",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
