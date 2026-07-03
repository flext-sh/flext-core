"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Self

from flext_core._models.base import FlextModelsBase
from flext_core._utilities.pydantic import FlextUtilitiesPydantic
from flext_core.protocols import FlextProtocols as p

from .flextmodelscontextscope_part_02 import (
    FlextModelsContextScope as FlextModelsContextScopePart02,
)


class FlextModelsContextScope(FlextModelsContextScopePart02):
    class ContextContainerState(FlextModelsBase.ArbitraryTypesModel):
        """Centralized container binding state for `FlextContext`."""

        container: Annotated[
            p.Container | None,
            FlextUtilitiesPydantic.Field(
                default=None,
                description="Container configured for service namespace resolution",
            ),
        ] = None

        @FlextUtilitiesPydantic.computed_field()
        def configured(self) -> bool:
            """Whether a container is configured for service access."""
            return self.container is not None

        def with_container(self, container: p.Container | None) -> Self:
            """Replace the configured container immutably."""
            updated_state: Self = self.model_copy(update={"container": container})
            return updated_state


__all__: list[str] = ["FlextModelsContextScope"]
