"""Pure state declarations for the operational public context facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core import t
from flext_core._models import (
    FlextModelsBase,
    FlextModelsContainers,
    FlextModelsPydantic,
)
from flext_core._models._context import FlextModelsContextScope


class FlextContextState(FlextModelsBase.ManagedModel):
    """Scope values, metadata and the shared service-container binding."""

    model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
        FlextModelsPydantic.ConfigDict(
            extra="forbid",
            validate_assignment=False,
            arbitrary_types_allowed=True,
        )
    )
    data: FlextModelsContainers.ConfigMap = FlextModelsPydantic.Field(
        default_factory=lambda: FlextModelsContainers.ConfigMap(root={}),
        description="Scoped key-value payload for this context instance.",
    )
    metadata: FlextModelsBase.Metadata = FlextModelsPydantic.Field(
        default_factory=FlextModelsBase.Metadata,
        description="Correlation and service metadata snapshot.",
    )
    _container_state: ClassVar[FlextModelsContextScope.ContextContainerState] = (
        FlextModelsContextScope.ContextContainerState()
    )


__all__: t.StrSequence = ("FlextContextState",)
