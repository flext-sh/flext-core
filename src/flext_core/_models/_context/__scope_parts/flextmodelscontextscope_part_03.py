"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import p
from flext_core._models._context.__scope_parts.flextmodelscontextscope_part_02 import (
    FlextModelsContextScope as FlextModelsContextScopePart02,
)
from flext_core._models.base import FlextModelsBase
from flext_core._models.pydantic import FlextModelsPydantic as mp


class FlextModelsContextScope(FlextModelsContextScopePart02):
    class ContextContainerState(FlextModelsBase.ArbitraryTypesModel):
        """Centralized container binding state for `FlextContext`."""

        container: Annotated[
            p.Container | None,
            mp.Field(
                default=None,
                description="Container configured for service namespace resolution",
            ),
        ] = None

        @mp.computed_field
        def configured(self) -> bool:
            """Whether a container is configured for service access.

            Returns:
                The resulting ``bool``.

            """
            return self.container is not None


__all__: list[str] = ["FlextModelsContextScope"]
