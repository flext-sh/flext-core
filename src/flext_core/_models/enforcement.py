"""Public enforcement model namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._enforcement._base import FlextModelsEnforcementBase
from ._enforcement._catalog import FlextModelsEnforcementCatalog
from ._enforcement._params import FlextModelsEnforcementParams
from ._enforcement._sources import FlextModelsEnforcementSources


class FlextModelsEnforcement(
    FlextModelsEnforcementParams,
    FlextModelsEnforcementCatalog,
    FlextModelsEnforcementSources,
    FlextModelsEnforcementBase,
):
    """Public facade for enforcement model namespaces."""


__all__: list[str] = ["FlextModelsEnforcement"]
