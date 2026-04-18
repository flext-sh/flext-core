"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from flext_core import (
    FlextModelsNamespace,
    FlextModelsPydantic,
    FlextTypesAnnotateds,
    FlextTypesCore,
    FlextTypesPydantic,
    FlextTypesServices,
    FlextTypesTypeAdapters,
    FlextTypingProjectMetadata,
)

if TYPE_CHECKING:
    from flext_core import FlextSettings

EnumT = TypeVar("EnumT", bound=StrEnum)
"""Type variable bounded to ``StrEnum`` implementations."""
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
"""Contravariant message type for dispatcher and handler protocols."""
P = ParamSpec("P")
"""Parameter specification for preserving callable signatures."""
R = TypeVar("R")
"""Generic return type variable for callables and result helpers."""
RootValueT = TypeVar("RootValueT")
"""Root value type for wrapped Pydantic root models."""
ResultT = TypeVar("ResultT")
"""Result payload type used across railway-style operations."""
T = TypeVar("T")
"""Unconstrained generic type variable."""
T_co = TypeVar("T_co", covariant=True)
"""Covariant generic type variable for read-only positions."""
T_contra = TypeVar("T_contra", contravariant=True)
"""Contravariant generic type variable for input-only positions."""
T_DomainResult = TypeVar(
    "T_DomainResult",
    bound=FlextTypesServices.ValueOrModel | Sequence[FlextTypesServices.ValueOrModel],
)
"""Domain result constrained to FLEXT value-or-model payloads."""
T_Model = TypeVar("T_Model", bound=FlextModelsPydantic.BaseModel)
"""Generic type variable bounded to Pydantic base models."""
T_Namespace = TypeVar("T_Namespace")
"""Namespace type variable for facade and MRO composition helpers."""
T_Settings = TypeVar("T_Settings", bound="FlextSettings")
"""Settings type variable bounded to ``FlextSettings`` implementations."""
TRuntime = TypeVar("TRuntime")
"""Runtime state type variable for service and container orchestration."""
TV = TypeVar("TV")
"""Generic type variable for validated values."""
TV_co = TypeVar("TV_co", covariant=True)
"""Covariant validated-value type variable."""
U = TypeVar("U")
"""Secondary unconstrained generic type variable."""


class FlextTypes(
    FlextTypesAnnotateds,
    FlextTypesCore,
    FlextTypesPydantic,
    FlextTypesServices,
    FlextTypesTypeAdapters,
    FlextTypingProjectMetadata,
    FlextModelsNamespace,
):
    """Type system foundation for FLEXT ecosystem.

    Strictly tiered layers - Primitives subset Scalar subset Container.
    ``object`` and ``Any`` are strictly forbidden in domain state.
    ``None`` is **never** baked into definitions.
    """


t = FlextTypes

__all__: list[str] = [
    "TV",
    "EnumT",
    "FlextTypes",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "RootValueT",
    "T",
    "TRuntime",
    "TV_co",
    "T_DomainResult",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "U",
    "t",
]
