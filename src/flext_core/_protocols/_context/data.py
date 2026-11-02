# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/data
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core.protocols import p


class FlextProtocolsContextData:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextData(Protocol):
        """Structural data contract projected from the owning model."""

        data: Mapping[str, t.Scalar]
        metadata: p.Metadata | Mapping[str, t.Scalar] | None


__all__: list[str] = [
    "FlextProtocolsContextData",
]
