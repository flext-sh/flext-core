"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Annotated, ClassVar

import flext_core
from flext_core._models import FlextModelsBase, FlextModelsPydantic
from flext_core._models._context import FlextModelsContextData


class FlextModelsContextScopePart01:
    """Namespace for context scope and statistics models."""

    class ContextStatistics(FlextModelsBase.ArbitraryTypesModel):
        """Statistics tracking for context operations.

        Enforcement exemption: counters and the ``operations`` map are
        incremented throughout the context lifecycle; fresh per-instance.
        """

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True)
        )

        sets: Annotated[
            flext_core.t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=flext_core.c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of set operations",
            ),
        ] = flext_core.c.DEFAULT_MAX_COMMAND_RETRIES
        gets: Annotated[
            flext_core.t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=flext_core.c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of get operations",
            ),
        ] = flext_core.c.DEFAULT_MAX_COMMAND_RETRIES
        removes: Annotated[
            flext_core.t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=flext_core.c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of remove operations",
            ),
        ] = flext_core.c.DEFAULT_MAX_COMMAND_RETRIES
        clears: Annotated[
            flext_core.t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=flext_core.c.DEFAULT_MAX_COMMAND_RETRIES,
                description="Number of clear operations",
            ),
        ] = flext_core.c.DEFAULT_MAX_COMMAND_RETRIES
        operations: Annotated[
            flext_core.t.JsonMapping,
            FlextModelsPydantic.BeforeValidator(
                FlextModelsContextData.ContextData.validate_dict_serializable,
            ),
            FlextModelsPydantic.Field(
                description=(
                    "Additional metric counters and timing values grouped by metric "
                    "key."
                ),
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=lambda: MappingProxyType[str, flext_core.t.JsonValue]({}),
            validate_default=True,
        )


__all__: list[str] = ["FlextModelsContextScopePart01"]
