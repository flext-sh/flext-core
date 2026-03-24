"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import BaseModel

from flext_core import (
    TV,
    EnumT,
    FlextTypesCore,
    FlextTypesServices,
    FlextTypesValidation,
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


class FlextTypes(
    FlextTypesCore,
    FlextTypesServices,
    FlextTypesValidation,
):
    """Type system foundation for FLEXT ecosystem.

    Strictly tiered layers - Primitives subset Scalar subset Container.
    ``t.NormalizedValue`` and ``Any`` are strictly forbidden in domain state.
    ``None`` is **never** baked into definitions.
    """


t = FlextTypes

__all__ = [
    "TV",
    "BaseModel",
    "EnumT",
    "FlextTypes",
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
    "t",
]
