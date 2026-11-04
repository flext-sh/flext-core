# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/data
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._typings import FlextTypingBase


class FlextProtocolsContextData:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ContextData(FlextProtocolsBase.FlexibleInternalModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def data(self) -> Mapping[str, FlextTypingBase.Scalar]:
            """Initial context data as key-value pairs."""
            ...

        @property
        def metadata(
            self,
        ) -> FlextProtocolsBase.Metadata | Mapping[str, FlextTypingBase.Scalar] | None:
            """Context metadata (creation info, source, etc.)."""
            ...


__all__: list[str] = [
    "FlextProtocolsContextData",
]
