"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import ParamSpec, TypeVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from flext_core import (
    FlextModelsNamespace,
    FlextTypesAnnotateds,
    FlextTypesCore,
    FlextTypesServices,
    FlextTypesTypeAdapters,
    FlextTypesValidation,
)

EnumT = TypeVar("EnumT", bound=StrEnum)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
P = ParamSpec("P")
R = TypeVar("R")
ResultT = TypeVar("ResultT")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
T_Model = TypeVar("T_Model", bound=BaseModel)
T_Namespace = TypeVar("T_Namespace")
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
TRuntime = TypeVar("TRuntime")
TV = TypeVar("TV")
TV_co = TypeVar("TV_co", covariant=True)
U = TypeVar("U")


class FlextTypes(
    FlextTypesAnnotateds,
    FlextTypesCore,
    FlextTypesServices,
    FlextTypesValidation,
    FlextTypesTypeAdapters,
    FlextModelsNamespace,
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
