"""Type aliases and generics for the FLEXT ecosystem - Thin MRO Facade.

from flext_core import FlextTypes as Types

Zero internal imports - depends only on stdlib, pydantic, pydantic-settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._typings.annotateds import FlextTypesAnnotateds
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.containers import FlextTypingContainers
from flext_core._typings.core import FlextTypesCore
from flext_core._typings.project_metadata import FlextTypingProjectMetadata
from flext_core._typings.pydantic import FlextTypesPydantic
from flext_core._typings.services import FlextTypesServices
from flext_core._typings.typeadapters import FlextTypesTypeAdapters


class FlextTypes(
    FlextTypingBase,
    FlextTypesAnnotateds,
    FlextTypingContainers,
    FlextTypesCore,
    FlextTypesPydantic,
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

__all__: list[str] = [
    "FlextTypes",
    "t",
]
