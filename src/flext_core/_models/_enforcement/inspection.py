"""Rule-level accounting for source-proven alias deferrals.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models._enforcement import (
    FlextModelsEnforcementBase,
    FlextModelsEnforcementResolution,
)


class FlextModelsEnforcementInspection(FlextModelsEnforcementResolution):
    """Bind rule context to an already-defined alias resolution contract."""

    class DeferredInspection(FlextModelsEnforcementBase):
        """A rule whose required alias value belongs to static type checking."""

        tag: str
        location: str
        alias: FlextModelsEnforcementResolution.DeferredAlias


__all__ = ["FlextModelsEnforcementInspection"]
