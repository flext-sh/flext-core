"""Public enforcement model namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models._enforcement._base import FlextModelsEnforcementBase
from flext_core._models._enforcement._catalog import FlextModelsEnforcementCatalog
from flext_core._models._enforcement._inspection import FlextModelsEnforcementInspection
from flext_core._models._enforcement._params import FlextModelsEnforcementParams
from flext_core._models._enforcement._sources import FlextModelsEnforcementSources
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._typings.base import FlextTypingBase


class FlextModelsEnforcement(
    FlextModelsEnforcementParams,
    FlextModelsEnforcementCatalog,
    FlextModelsEnforcementSources,
    FlextModelsEnforcementInspection,
    FlextModelsEnforcementBase,
):
    """Public facade for enforcement model namespaces."""

    class Report(FlextModelsEnforcementBase.Report):
        """Violations and source-proven deferrals, kept as distinct outcomes.

        The inherited sequence protocol describes violations only. ``complete``
        additionally answers whether every requested runtime inspection ran.
        """

        deferred: FlextTypingBase.SequenceOf[
            FlextModelsEnforcementInspection.DeferredInspection
        ] = ()

        @FlextModelsPydantic.computed_field
        @property
        def complete(self) -> bool:
            """Whether all requested alias values were available at runtime."""
            return not self.deferred


__all__: list[str] = ["FlextModelsEnforcement"]
