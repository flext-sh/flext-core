"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from flext_core._protocols.result import FlextProtocolsResult as prt
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.config import FlextTypingConfig
from flext_core._typings.containers import FlextTypingContainers
from flext_core._typings.core import FlextTypesCore
from flext_core._typings.lazy import FlextTypesLazy
from flext_core._typings.project_metadata import FlextTypingProjectMetadata
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._typings.services import FlextTypesServices
from flext_core._typings.typeadapters import FlextTypesTypeAdapters


class FlextTypes(
    FlextTypingBase,
    FlextTypingConfig,
    FlextTypingContainers,
    FlextTypesCore,
    FlextTypesLazy,
    FlextTypesServices,
    FlextTypesTypeAdapters,
    FlextTypingProjectMetadata,
):
    """Type system foundation for FLEXT ecosystem.

    Strictly tiered layers - Primitives subset Scalar subset Container.
    ``object`` and ``Any`` are strictly forbidden in domain state.
    ``None`` is **never** baked into definitions.
    """


t = FlextTypes


type JsonMapping = Mapping[str, tp.JsonValue]

type JsonDict = dict[str, tp.JsonValue]

type ConfigModelInput = prt.HasModelDump | JsonMapping

T = TypeVar("T")

__all__: list[str] = ["FlextTypes", "t"]
