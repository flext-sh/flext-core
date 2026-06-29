"""Composed enforcement model namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models._enforcement._catalog import FlextModelsEnforcementCatalog
from flext_core._models._enforcement._params import FlextModelsEnforcementParams


class FlextModelsEnforcement(
    FlextModelsEnforcementCatalog,
    FlextModelsEnforcementParams,
):
    """Namespace for enforcement violation, predicate, and catalog models."""


__all__: list[str] = ["FlextModelsEnforcement"]
