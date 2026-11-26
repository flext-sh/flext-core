# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex11
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx11:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Payload(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def text(self) -> str:
            """Payload text content."""
            ...

    @runtime_checkable
    class ServiceHandlerConfig(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        enabled: bool

    @runtime_checkable
    class ServiceHandlerLike(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        message_type: type[p.Value]

    @runtime_checkable
    class ProcessorProtocolGood(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def status(self) -> str:
            """Processing outcome status."""
            ...

    @runtime_checkable
    class ProcessorProtocolBad(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def status(self) -> str:
            """Processing failure status."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx11",
]
