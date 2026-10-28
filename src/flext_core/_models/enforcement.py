"""Extracted facade class for enforcement.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models._enforcement import (
    FlextModelsEnforcementCatalog,
    FlextModelsEnforcementInspection,
    FlextModelsEnforcementParams,
)


class FlextModelsEnforcement(
    FlextModelsEnforcementCatalog,
    FlextModelsEnforcementInspection,
    FlextModelsEnforcementParams,
):
    """Facade for enforcement."""


__all__: list[str] = ["FlextModelsEnforcement"]
