# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/options
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._typings import FlextTypesPydantic


class FlextProtocolsOptions:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class GenerateOptions(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def prefix(self) -> str | None:
            """Custom ID prefix."""
            ...

        @property
        def parts(self) -> tuple[FlextTypesPydantic.JsonValue, ...] | None:
            """Optional parts inserted between prefix and random suffix."""
            ...

        @property
        def length(self) -> int | None:
            """Optional random suffix length override."""
            ...

        @property
        def include_timestamp(self) -> bool:
            """Whether to prepend a UTC timestamp to parts."""
            ...

        @property
        def separator(self) -> str:
            """Separator used for custom formatted IDs."""
            ...


__all__: list[str] = [
    "FlextProtocolsOptions",
]
