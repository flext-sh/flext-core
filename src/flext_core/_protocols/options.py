# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/options
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import flext_core
from flext_core._protocols import FlextProtocolsPydantic

if TYPE_CHECKING:
    pass


class FlextProtocolsOptions:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class GenerateOptions(FlextProtocolsPydantic.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        prefix: str | None
        parts: tuple[flext_core.t.JsonValue, ...] | None
        length: int | None
        include_timestamp: bool
        separator: str


__all__: list[str] = [
    "FlextProtocolsOptions",
]
