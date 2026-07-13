"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

from flext_core import FlextTypes as Types

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._typings.base import FlextTypingBase
from ._typings.config import FlextTypingConfig
from ._typings.containers import FlextTypingContainers
from ._typings.core import FlextTypesCore
from ._typings.lazy import FlextTypesLazy
from ._typings.project_metadata import FlextTypingProjectMetadata
from ._typings.services import FlextTypesServices
from ._typings.typeadapters import FlextTypesTypeAdapters


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


t_core = FlextTypes
t = FlextTypes

__all__: list[str] = ["FlextTypes", "t", "t_core"]
