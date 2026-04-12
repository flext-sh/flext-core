"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from flext_core._models.namespace import FlextModelsNamespace
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._typings.annotateds import FlextTypesAnnotateds
from flext_core._typings.core import FlextTypesCore
from flext_core._typings.pydantic import FlextTypesPydantic
from flext_core._typings.services import FlextTypesServices
from flext_core._typings.typeadapters import FlextTypesTypeAdapters
from flext_core._typings.validation import FlextTypesValidation
from flext_core._utilities.pydantic import FlextUtilitiesPydantic

if TYPE_CHECKING:
    from flext_core.settings import FlextSettings

BaseModel = FlextModelsPydantic.BaseModel
TypeAdapter = FlextUtilitiesPydantic.TypeAdapter

EnumT = TypeVar("EnumT", bound=StrEnum)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
P = ParamSpec("P")
R = TypeVar("R")
ResultT = TypeVar("ResultT")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
T_DomainResult = TypeVar(
    "T_DomainResult",
    bound=FlextTypesServices.ValueOrModel | Sequence[FlextTypesServices.ValueOrModel],
)
T_Model = TypeVar("T_Model", bound=BaseModel)
T_Namespace = TypeVar("T_Namespace")
T_Settings = TypeVar("T_Settings", bound="FlextSettings")
TRuntime = TypeVar("TRuntime")
TV = TypeVar("TV")
TV_co = TypeVar("TV_co", covariant=True)
U = TypeVar("U")


class FlextTypes(
    FlextTypesAnnotateds,
    FlextTypesCore,
    FlextTypesPydantic,
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

    # Canonical settings-first names
    SettingsMap = FlextTypesCore.ConfigMap
    SettingsModelInput = FlextTypesServices.ConfigModelInput


t = FlextTypes

__all__: list[str] = [
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
    "T_DomainResult",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "TypeAdapter",
    "U",
    "t",
]
