# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Validator extensions for FLEXT architecture validation.

Internal module providing specialized validation methods.
Use via FlextTestsValidator (tv) in validator.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_tests._validator.bypass import FlextValidatorBypass
    from flext_tests._validator.imports import FlextValidatorImports
    from flext_tests._validator.layer import FlextValidatorLayer
    from flext_tests._validator.models import (
        FlextValidatorModels,
        FlextValidatorModels as m,
        vm,
    )
    from flext_tests._validator.settings import FlextValidatorSettings
    from flext_tests._validator.tests import FlextValidatorTests
    from flext_tests._validator.types import (
        FlextValidatorTypes,
        FlextValidatorTypes as t,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextValidatorBypass": ("flext_tests._validator.bypass", "FlextValidatorBypass"),
    "FlextValidatorImports": (
        "flext_tests._validator.imports",
        "FlextValidatorImports",
    ),
    "FlextValidatorLayer": ("flext_tests._validator.layer", "FlextValidatorLayer"),
    "FlextValidatorModels": ("flext_tests._validator.models", "FlextValidatorModels"),
    "FlextValidatorSettings": (
        "flext_tests._validator.settings",
        "FlextValidatorSettings",
    ),
    "FlextValidatorTests": ("flext_tests._validator.tests", "FlextValidatorTests"),
    "FlextValidatorTypes": ("flext_tests._validator.types", "FlextValidatorTypes"),
    "m": ("flext_tests._validator.models", "FlextValidatorModels"),
    "t": ("flext_tests._validator.types", "FlextValidatorTypes"),
    "vm": ("flext_tests._validator.models", "vm"),
}

__all__ = [
    "FlextValidatorBypass",
    "FlextValidatorImports",
    "FlextValidatorLayer",
    "FlextValidatorModels",
    "FlextValidatorSettings",
    "FlextValidatorTests",
    "FlextValidatorTypes",
    "m",
    "t",
    "vm",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
