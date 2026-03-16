# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Internal typings subpackage for FlextTypes MRO composition.

This package contains domain-specific typing classes composed by the facade.
External code should import from ``flext_core.t`` instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.containers import FlextTypingContainers
    from flext_core._typings.core import FlextTypesCore
    from flext_core._typings.generics import (
        TV,
        EnumT,
        MessageT_contra,
        P,
        R,
        ResultT,
        T,
        T_co,
        T_contra,
        T_Model,
        T_Namespace,
        T_Settings,
        TRuntime,
        TV_co,
        U,
        ValidatedParams,
        ValidatedReturn,
    )
    from flext_core._typings.services import FlextTypesServices
    from flext_core._typings.validation import FlextTypesValidation
    from flext_core.typings import FlextTypes

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EnumT": ("flext_core._typings.generics", "EnumT"),
    "FlextTypesCore": ("flext_core._typings.core", "FlextTypesCore"),
    "FlextTypesServices": ("flext_core._typings.services", "FlextTypesServices"),
    "FlextTypesValidation": ("flext_core._typings.validation", "FlextTypesValidation"),
    "FlextTypingBase": ("flext_core._typings.base", "FlextTypingBase"),
    "FlextTypingContainers": (
        "flext_core._typings.containers",
        "FlextTypingContainers",
    ),
    "MessageT_contra": ("flext_core._typings.generics", "MessageT_contra"),
    "P": ("flext_core._typings.generics", "P"),
    "R": ("flext_core._typings.generics", "R"),
    "ResultT": ("flext_core._typings.generics", "ResultT"),
    "T": ("flext_core._typings.generics", "T"),
    "TRuntime": ("flext_core._typings.generics", "TRuntime"),
    "TV": ("flext_core._typings.generics", "TV"),
    "TV_co": ("flext_core._typings.generics", "TV_co"),
    "T_Model": ("flext_core._typings.generics", "T_Model"),
    "T_Namespace": ("flext_core._typings.generics", "T_Namespace"),
    "T_Settings": ("flext_core._typings.generics", "T_Settings"),
    "T_co": ("flext_core._typings.generics", "T_co"),
    "T_contra": ("flext_core._typings.generics", "T_contra"),
    "U": ("flext_core._typings.generics", "U"),
    "ValidatedParams": ("flext_core._typings.generics", "ValidatedParams"),
    "ValidatedReturn": ("flext_core._typings.generics", "ValidatedReturn"),
}

__all__ = [
    "TV",
    "EnumT",
    "FlextTypesCore",
    "FlextTypesServices",
    "FlextTypesValidation",
    "FlextTypingBase",
    "FlextTypingContainers",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "T",
    "TRuntime",
    "TV_co",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "U",
    "ValidatedParams",
    "ValidatedReturn",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
