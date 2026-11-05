# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/data
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import flext_core
from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    pass


class FlextProtocolsContextData:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextData(FlextProtocolsBase.FlexibleInternalModel, Protocol):
        """Structural data contract projected from the owning model."""

        data: Mapping[str, flext_core.t.Scalar]

        @property
        def metadata(
            self,
        ) -> FlextProtocolsBase.Metadata | Mapping[str, flext_core.t.Scalar] | None:
            """Context metadata (creation info, source, etc.)."""
            ...


__all__: list[str] = [
    "FlextProtocolsContextData",
]
