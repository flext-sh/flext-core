"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated

from flext_core._models._context._data import FlextModelsContextData
from flext_core._models.base import FlextModelsBase
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities.pydantic import FlextUtilitiesPydantic
from flext_core.constants import FlextConstants as c

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes as t


class FlextModelsContextScope:
    """Namespace for context scope and statistics models."""

    class ContextStatistics(FlextModelsBase.ArbitraryTypesModel):
        """Statistics tracking for context operations.

        Enforcement exemption: counters and the ``operations`` map are
        incremented throughout the context lifecycle; fresh per-instance.
        """

        sets: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of set operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        gets: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of get operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        removes: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of remove operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        clears: Annotated[
            t.NonNegativeInt,
            FlextUtilitiesPydantic.Field(
                default=c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of clear operations",
            ),
        ] = c.DEFAULT_MAX_COMMAND_RETRIES
        operations: Annotated[
            t.JsonMapping,
            FlextModelsPydantic.BeforeValidator(
                lambda v: (
                    FlextModelsContextData.normalize_to_mapping(v)
                    if v is not None
                    else {}
                ),
            ),
            FlextUtilitiesPydantic.Field(
                description="Additional metric counters and timing values grouped by metric key.",
            ),
        ] = FlextUtilitiesPydantic.Field(default_factory=lambda: MappingProxyType({}))


__all__: list[str] = ["FlextModelsContextScope"]
