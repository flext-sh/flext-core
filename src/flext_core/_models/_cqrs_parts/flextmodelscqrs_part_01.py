"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field, computed_field

from flext_core import c, t

from ..base import FlextModelsBase as m


class CqrsPagination(m.FlexibleInternalModel):
    """Pagination model for query results.

    Defined at module level so it can be referenced in Query annotations
    without forward-reference issues (Pydantic can resolve it statically).
    Exposed as FlextModelsCqrs.Pagination.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={
            "title": "Pagination",
            "description": "Pagination model for query results with computed fields",
        }
    )
    page: Annotated[
        t.PositiveInt,
        Field(description="Page number (1-based indexing)", examples=[1, 2, 10, 100]),
    ] = c.DEFAULT_RETRY_DELAY_SECONDS
    size: Annotated[
        t.PositiveInt,
        Field(
            le=c.MAX_PAGE_SIZE,
            description="Number of items per page (max 1000)",
            examples=[10, 20, 50, 100],
        ),
    ] = c.DEFAULT_PAGE_SIZE

    @computed_field
    @property
    def limit(self) -> int:
        """Limit alias for size."""
        return self.size

    @computed_field
    @property
    def offset(self) -> int:
        """Offset from page and size."""
        return (self.page - 1) * self.size
