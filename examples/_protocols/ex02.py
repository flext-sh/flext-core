# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex02
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols, c

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx02:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class DatabaseService(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings(self) -> p.ConfigMap:
            """Database connection settings."""
            ...

        @property
        def status(self) -> c.Status:
            """Service connection status."""
            ...

        @staticmethod
        def connect() -> p.Result[bool]:
            """Structural member projected from the owning model."""

        @staticmethod
        def query(sql: str) -> p.Result[p.ConfigMap]:
            """Structural member projected from the owning model."""

    @runtime_checkable
    class CacheService(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings(self) -> p.ConfigMap:
            """Cache connection settings."""
            ...

        @property
        def status(self) -> c.Status:
            """Service connection status."""
            ...

        @staticmethod
        def set(key: str, value: str) -> p.Result[bool]:
            """Structural member projected from the owning model."""

    @runtime_checkable
    class EmailService(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings(self) -> p.ConfigMap:
            """Email service settings."""
            ...

        @property
        def status(self) -> c.Status:
            """Service connection status."""
            ...

        @staticmethod
        def send(to: str, subject: str, body: str) -> p.Result[bool]:
            """Structural member projected from the owning model."""


__all__: list[str] = [
    "ExamplesFlextProtocolsEx02",
]
