"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import ConfigDict, Field, computed_field

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._models import FlextModelsBase


class FlextModelsCqrsPart01:
    """First CQRS namespace part: the models the query contract references."""

    class Pagination(FlextModelsBase.FlexibleInternalModel):
        """Pagination model for query results.

        Declared in the first part so the query annotations of the next part
        reference it through an importable module global.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "Pagination",
                "description": (
                    "Pagination model for query results with computed fields"
                ),
            },
        )
        page: Annotated[
            flext_core.t.PositiveInt,
            Field(description="Page number (1-based indexing)", examples=[1, 2, 10]),
        ] = flext_core.c.FIRST_PAGE_INDEX
        size: Annotated[
            flext_core.t.PositiveInt,
            Field(
                ge=flext_core.c.MIN_PAGE_SIZE,
                le=flext_core.c.MAX_PAGE_SIZE,
                description="Number of items per page within the configured bounds",
                examples=[
                    flext_core.c.MIN_PAGE_SIZE,
                    flext_core.c.DEFAULT_PAGE_SIZE,
                    flext_core.c.MAX_PAGE_SIZE,
                ],
            ),
        ] = flext_core.c.DEFAULT_PAGE_SIZE

        @computed_field
        @property
        def limit(self) -> int:
            """Limit alias for size."""
            return self.size

        @computed_field
        @property
        def offset(self) -> int:
            """Offset from page and size."""
            return (self.page - flext_core.c.FIRST_PAGE_INDEX) * self.size


__all__: list[str] = ["FlextModelsCqrsPart01"]
