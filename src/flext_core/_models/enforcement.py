"""Public enforcement model namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models._enforcement._base import FlextModelsEnforcementBase
from flext_core._models._enforcement._catalog import FlextModelsEnforcementCatalog
from flext_core._models._enforcement._params import FlextModelsEnforcementParams
from flext_core._models._enforcement._sources import FlextModelsEnforcementSources


class FlextModelsEnforcement(
    FlextModelsEnforcementParams,
    FlextModelsEnforcementCatalog,
    FlextModelsEnforcementSources,
    FlextModelsEnforcementBase,
):
    """Public facade for enforcement model namespaces."""


__all__: list[str] = ["FlextModelsEnforcement"]
