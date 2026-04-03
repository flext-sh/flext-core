# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Typings package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import flext_core._typings.base as _flext_core__typings_base

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
    import flext_core._typings.validation as _flext_core__typings_validation
    from flext_core._typings.services import FlextTypesServices

    validation = _flext_core__typings_validation
    from flext_core._typings.validation import FlextTypesValidation
_LAZY_IMPORTS = {
    "EnumT": "flext_core._typings.generics",
    "FlextTypesCore": "flext_core._typings.core",
    "FlextTypesServices": "flext_core._typings.services",
    "FlextTypesValidation": "flext_core._typings.validation",
    "FlextTypingBase": "flext_core._typings.base",
    "FlextTypingContainers": "flext_core._typings.containers",
    "MessageT_contra": "flext_core._typings.generics",
    "P": "flext_core._typings.generics",
    "R": "flext_core._typings.generics",
    "ResultT": "flext_core._typings.generics",
    "T": "flext_core._typings.generics",
    "TRuntime": "flext_core._typings.generics",
    "TV": "flext_core._typings.generics",
    "TV_co": "flext_core._typings.generics",
    "T_Model": "flext_core._typings.generics",
    "T_Namespace": "flext_core._typings.generics",
    "T_Settings": "flext_core._typings.generics",
    "T_co": "flext_core._typings.generics",
    "T_contra": "flext_core._typings.generics",
    "U": "flext_core._typings.generics",
    "base": "flext_core._typings.base",
    "containers": "flext_core._typings.containers",
    "core": "flext_core._typings.core",
    "generics": "flext_core._typings.generics",
    "services": "flext_core._typings.services",
    "validation": "flext_core._typings.validation",
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
    "base",
    "containers",
    "core",
    "generics",
    "services",
    "validation",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
