# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/flext_context
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from flext_core.protocols import p


class FlextProtocolsFlextContext:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextState(Protocol):
        """Structural data contract projected from the owning model."""

        data: p.ConfigMap
        metadata: p.Metadata


__all__: list[str] = [
    "FlextProtocolsFlextContext",
]
