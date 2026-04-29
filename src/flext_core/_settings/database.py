"""FlextSettingsDatabase — database connectivity fields.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from flext_core import c, t


class FlextSettingsDatabase:
    """Database-related settings fields."""

    database_url: Annotated[str, Field(description="Database URL")] = c.DATABASE_URL
    database_pool_size: Annotated[
        t.PositiveInt,
        Field(
            description="Database pool size",
        ),
    ] = c.DEFAULT_PAGE_SIZE


__all__: list[str] = ["FlextSettingsDatabase"]
