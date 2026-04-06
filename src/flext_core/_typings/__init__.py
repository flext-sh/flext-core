# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Typings package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import flext_core._typings.annotateds as _flext_core__typings_annotateds

    annotateds = _flext_core__typings_annotateds
    import flext_core._typings.base as _flext_core__typings_base
    from flext_core._typings.annotateds import FlextTypesAnnotateds

    base = _flext_core__typings_base
    import flext_core._typings.containers as _flext_core__typings_containers
    from flext_core._typings.base import FlextTypingBase

    containers = _flext_core__typings_containers
    import flext_core._typings.core as _flext_core__typings_core
    from flext_core._typings.containers import FlextTypingContainers

    core = _flext_core__typings_core
    import flext_core._typings.generics as _flext_core__typings_generics
    from flext_core._typings.core import FlextTypesCore

    generics = _flext_core__typings_generics
    import flext_core._typings.services as _flext_core__typings_services
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
    )

    services = _flext_core__typings_services
    import flext_core._typings.typeadapters as _flext_core__typings_typeadapters
    from flext_core._typings.services import FlextTypesServices

    typeadapters = _flext_core__typings_typeadapters
    import flext_core._typings.validation as _flext_core__typings_validation
    from flext_core._typings.typeadapters import FlextTypesTypeAdapters

    validation = _flext_core__typings_validation
    from flext_core._typings.validation import FlextTypesValidation
_LAZY_IMPORTS = {
    "EnumT": ("flext_core._typings.generics", "EnumT"),
    "FlextTypesAnnotateds": ("flext_core._typings.annotateds", "FlextTypesAnnotateds"),
    "FlextTypesCore": ("flext_core._typings.core", "FlextTypesCore"),
    "FlextTypesServices": ("flext_core._typings.services", "FlextTypesServices"),
    "FlextTypesTypeAdapters": (
        "flext_core._typings.typeadapters",
        "FlextTypesTypeAdapters",
    ),
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
    "annotateds": "flext_core._typings.annotateds",
    "base": "flext_core._typings.base",
    "containers": "flext_core._typings.containers",
    "core": "flext_core._typings.core",
    "generics": "flext_core._typings.generics",
    "services": "flext_core._typings.services",
    "typeadapters": "flext_core._typings.typeadapters",
    "validation": "flext_core._typings.validation",
}

__all__ = [
    "TV",
    "EnumT",
    "FlextTypesAnnotateds",
    "FlextTypesCore",
    "FlextTypesServices",
    "FlextTypesTypeAdapters",
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
    "annotateds",
    "base",
    "containers",
    "core",
    "generics",
    "services",
    "typeadapters",
    "validation",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
