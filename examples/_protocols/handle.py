# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/handle
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import FlextProtocols


class ExamplesFlextProtocolsHandle:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Handle(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def value(self) -> int:
            """Opaque integer handle identifier."""
            ...

        @property
        def cleaned(self) -> bool:
            """Whether the handle has been released."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsHandle",
]
