# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_context/tokens
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from flext_core import t
from flext_core._protocols import FlextProtocolsEntity


class FlextProtocolsContextTokens:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class StructlogProxyToken(FlextProtocolsEntity.Value, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def key(self) -> t.NonEmptyStr:
            """Context variable key (alphanumeric, underscore)."""
            ...

        @property
        def previous_value(self) -> t.JsonPayload | datetime | None:
            """Previous value before set operation."""
            ...

    @runtime_checkable
    class Token(FlextProtocolsEntity.Value, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def key(self) -> t.NonEmptyStr:
            """Unique key for the context variable."""
            ...

        @property
        def old_value(self) -> t.JsonPayload | None:
            """Previous value before set operation."""
            ...


__all__: list[str] = [
    "FlextProtocolsContextTokens",
]
