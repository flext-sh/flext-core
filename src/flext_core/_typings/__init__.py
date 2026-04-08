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
    import flext_core._typings.services as _flext_core__typings_services
    from flext_core._typings.core import FlextTypesCore

    services = _flext_core__typings_services
    import flext_core._typings.typeadapters as _flext_core__typings_typeadapters
    from flext_core._typings.services import FlextTypesServices

    typeadapters = _flext_core__typings_typeadapters
    import flext_core._typings.validation as _flext_core__typings_validation
    from flext_core._typings.typeadapters import FlextTypesTypeAdapters

    validation = _flext_core__typings_validation
    from flext_core._typings.validation import FlextTypesValidation
    from flext_core.typings import (
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
_LAZY_IMPORTS = {
    "EnumT": ("flext_core.typings", "EnumT"),
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
    "MessageT_contra": ("flext_core.typings", "MessageT_contra"),
    "P": ("flext_core.typings", "P"),
    "R": ("flext_core.typings", "R"),
    "ResultT": ("flext_core.typings", "ResultT"),
    "T": ("flext_core.typings", "T"),
    "TRuntime": ("flext_core.typings", "TRuntime"),
    "TV": ("flext_core.typings", "TV"),
    "TV_co": ("flext_core.typings", "TV_co"),
    "T_Model": ("flext_core.typings", "T_Model"),
    "T_Namespace": ("flext_core.typings", "T_Namespace"),
    "T_Settings": ("flext_core.typings", "T_Settings"),
    "T_co": ("flext_core.typings", "T_co"),
    "T_contra": ("flext_core.typings", "T_contra"),
    "U": ("flext_core.typings", "U"),
    "annotateds": "flext_core._typings.annotateds",
    "base": "flext_core._typings.base",
    "containers": "flext_core._typings.containers",
    "core": "flext_core._typings.core",
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
    "services",
    "typeadapters",
    "validation",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
