# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Typings package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_core._typings import (
        base,
        containers,
        core,
        generics,
        services,
        validation,
    )
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
    )
    from flext_core._typings.services import FlextTypesServices
    from flext_core._typings.validation import FlextTypesValidation

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = {
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
