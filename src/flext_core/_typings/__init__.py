# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Internal typings subpackage for FlextTypes MRO composition.

This package contains domain-specific typing classes composed by the facade.
External code should import from ``flext_core.t`` instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

# L0-OVERRIDE — inline lazy to avoid circular: _typings -> _utilities.lazy -> typings -> _typings
from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

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
    )
    from flext_core._typings.services import FlextTypesServices
    from flext_core._typings.validation import FlextTypesValidation

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "EnumT": ["flext_core._typings.generics", "EnumT"],
    "FlextTypesCore": ["flext_core._typings.core", "FlextTypesCore"],
    "FlextTypesServices": ["flext_core._typings.services", "FlextTypesServices"],
    "FlextTypesValidation": ["flext_core._typings.validation", "FlextTypesValidation"],
    "FlextTypingBase": ["flext_core._typings.base", "FlextTypingBase"],
    "FlextTypingContainers": ["flext_core._typings.containers", "FlextTypingContainers"],
    "MessageT_contra": ["flext_core._typings.generics", "MessageT_contra"],
    "P": ["flext_core._typings.generics", "P"],
    "R": ["flext_core._typings.generics", "R"],
    "ResultT": ["flext_core._typings.generics", "ResultT"],
    "T": ["flext_core._typings.generics", "T"],
    "TRuntime": ["flext_core._typings.generics", "TRuntime"],
    "TV": ["flext_core._typings.generics", "TV"],
    "TV_co": ["flext_core._typings.generics", "TV_co"],
    "T_Model": ["flext_core._typings.generics", "T_Model"],
    "T_Namespace": ["flext_core._typings.generics", "T_Namespace"],
    "T_Settings": ["flext_core._typings.generics", "T_Settings"],
    "T_co": ["flext_core._typings.generics", "T_co"],
    "T_contra": ["flext_core._typings.generics", "T_contra"],
    "U": ["flext_core._typings.generics", "U"],
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
]


def __getattr__(name: str) -> type:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> Sequence[str]:
    return sorted(__all__)


_current = sys.modules.get(__name__)
if _current is not None:
    _parts = __name__.split(".")
    for _mod_path, _ in _LAZY_IMPORTS.values():
        if _mod_path:
            _mp = _mod_path.split(".")
            if len(_mp) > len(_parts) and _mp[: len(_parts)] == _parts:
                _sub = getattr(_current, _mp[len(_parts)], None)
                if _sub is not None and isinstance(_sub, type(sys)):
                    delattr(_current, _mp[len(_parts)])
