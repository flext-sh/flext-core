"""Typed outcomes of PEP 695 alias evaluation.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Literal

from flext_core import p, t
from flext_core._models._enforcement import FlextModelsEnforcementBase


class FlextModelsEnforcementResolution:
    """Value and source-proven deferral contracts available before consumers."""

    class ResolvedAlias(FlextModelsEnforcementBase):
        """A lazy alias evaluated successfully, retaining its runtime value."""

        status: Literal["resolved"] = "resolved"
        value: t.Port[p.AttributeProbe]

    class DeferredAlias(FlextModelsEnforcementBase):
        """Source-proven imports prevent runtime evaluation of a declared alias."""

        status: Literal["deferred"] = "deferred"
        module: str
        qualname: str
        file_path: str
        line_number: int
        unavailable_imports: tuple[str, ...]


__all__ = ["FlextModelsEnforcementResolution"]
