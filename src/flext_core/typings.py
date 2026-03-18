"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from pydantic import BaseModel

from flext_core._typings import (
    FlextTypesCore,
    FlextTypesServices,
    FlextTypesValidation,
)
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

if TYPE_CHECKING:
    from flext_core import p


class FlextTypes(
    FlextTypesCore,
    FlextTypesServices,
    FlextTypesValidation,
):
    """Type system foundation for FLEXT ecosystem.

    Strictly tiered layers - Primitives subset Scalar subset Container.
    ``object`` and ``Any`` are strictly forbidden in domain state.
    ``None`` is **never** baked into definitions.
    """

    # --- Parameter specifications and type variables for validated functions ---
    ValidatedParams = ParamSpec("ValidatedParams")
    ValidatedReturn = TypeVar("ValidatedReturn")

    type RuntimeAtomic = FlextTypesCore.Container | BaseModel
    type RuntimeData = (
        FlextTypesCore.NormalizedValue
        | FlextTypesServices.MetadataValue
        | FlextTypesCore.ContainerValue
        | Mapping[str, FlextTypesCore.NormalizedValue | BaseModel]
        | Sequence[FlextTypesCore.NormalizedValue]
        | p.HasModelDump
    )

    # --- Dispatcher type aliases ---
    type DispatchableHandler = (
        Callable[
            ...,
            p.ResultLike[FlextTypesCore.Container | BaseModel]
            | FlextTypesCore.Container
            | BaseModel
            | None,
        ]
        | p.DispatchMessage
        | p.Handle
        | p.Execute
    )
    type ResolvedHandlerCallable = Callable[
        [p.Routable],
        p.ResultLike[FlextTypesCore.Container | BaseModel]
        | FlextTypesCore.Container
        | BaseModel
        | None,
    ]
    type RegisteredHandler = tuple[DispatchableHandler, ResolvedHandlerCallable]
    type AutoHandlerRegistration = tuple[
        DispatchableHandler,
        ResolvedHandlerCallable,
        tuple[FlextTypesServices.MessageTypeSpecifier, ...],
    ]


t = FlextTypes

# Re-export class attributes at module level for backward compatibility
ValidatedParams = FlextTypes.ValidatedParams
ValidatedReturn = FlextTypes.ValidatedReturn

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
    "ValidatedParams",
    "ValidatedReturn",
    "t",
]
