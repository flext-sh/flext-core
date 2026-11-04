# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex02
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from examples import c
    from flext_core import p


class ExamplesFlextProtocolsEx02:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class DatabaseService(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings(self) -> p.ConfigMap:
            """Database connection settings."""
            ...

        @property
        def status(self) -> c.Status:
            """Service connection status."""
            ...

    @runtime_checkable
    class CacheService(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings(self) -> p.ConfigMap:
            """Cache connection settings."""
            ...

        @property
        def status(self) -> c.Status:
            """Service connection status."""
            ...

    @runtime_checkable
    class EmailService(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings(self) -> p.ConfigMap:
            """Email service settings."""
            ...

        @property
        def status(self) -> c.Status:
            """Service connection status."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx02",
]
