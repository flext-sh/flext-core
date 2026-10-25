# AUTO-GENERATED FILE — Regenerate with: make gen
"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from flext_core._typings import (
    FlextTypesAnnotateds,
    FlextTypesCore,
    FlextTypesLazy,
    FlextTypesPydantic,
    FlextTypesServices,
    FlextTypingBase,
    FlextTypingConfig,
)


class FlextTypes(
    FlextTypingBase,
    FlextTypesPydantic,
    FlextTypingConfig,
    FlextTypesCore,
    FlextTypesAnnotateds,
    FlextTypesLazy,
    FlextTypesServices,
):
    """Type system foundation for FLEXT ecosystem.

    Strictly tiered layers - Primitives subset Scalar subset Container.
    ``object`` and ``Any`` are strictly forbidden in domain state.
    ``None`` is **never** baked into definitions.
    """


t = FlextTypes


type JsonMapping = Mapping[str, FlextTypesPydantic.JsonValue]

type JsonDict = dict[str, FlextTypesPydantic.JsonValue]

T = TypeVar("T")


type ModuleGlobalValue = FlextTypesLazy.ModuleGlobalValue

type ModuleGlobals = FlextTypesLazy.ModuleGlobals

__all__: list[str] = ["FlextTypes", "t"]
