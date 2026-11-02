# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/scope
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols._context._scope import (
    FlextProtocolsContextScopeFlextmodelscontextscopePart01,
    FlextProtocolsContextScopeFlextmodelscontextscopePart02,
)
from flext_core._protocols._context._scope.flextmodelscontextscope_part_03 import (
    FlextProtocolsContextScopeFlextmodelscontextscopePart03,
)


class FlextProtocolsContextScope(
    FlextProtocolsContextScopeFlextmodelscontextscopePart03,
    FlextProtocolsContextScopeFlextmodelscontextscopePart02,
    FlextProtocolsContextScopeFlextmodelscontextscopePart01,
):
    """Own the structural contracts declared by its model source."""


__all__: list[str] = [
    "FlextProtocolsContextScope",
]
