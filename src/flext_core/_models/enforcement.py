"""Public enforcement model namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .._typings.base import FlextTypingBase as t
from ._enforcement._base import FlextModelsEnforcementBase
from ._enforcement._catalog import FlextModelsEnforcementCatalog
from ._enforcement._inspection import FlextModelsEnforcementInspection
from ._enforcement._params import FlextModelsEnforcementParams
from ._enforcement._sources import FlextModelsEnforcementSources
from .pydantic import FlextModelsPydantic as mp


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

        deferred: t.SequenceOf[FlextModelsEnforcementInspection.DeferredInspection] = ()

        @mp.computed_field
        @property
        def complete(self) -> bool:
            """Whether all requested alias values were available at runtime."""
            return not self.deferred


__all__: list[str] = ["FlextModelsEnforcement"]
