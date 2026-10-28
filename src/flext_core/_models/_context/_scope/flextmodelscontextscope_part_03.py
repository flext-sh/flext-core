"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from typing import Annotated

from flext_core import p, t
from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._models._context._scope import FlextModelsContextScopePart02


class FlextModelsContextScopePart03(FlextModelsContextScopePart02):
    class ContextContainerState(FlextModelsBase.ArbitraryTypesModel):
        """Centralized container binding state for `FlextContext`."""

        container: Annotated[
            t.Port[p.Container] | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Container configured for service namespace resolution",
            ),
        ] = None

        @FlextModelsPydantic.computed_field
        def configured(self) -> bool:
            """Whether a container is configured for service access.

            Returns:
                The resulting ``bool``.

            """
            return self.container is not None


__all__: list[str] = ["FlextModelsContextScopePart03"]
