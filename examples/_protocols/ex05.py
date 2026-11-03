# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex05
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from examples import c
from flext_core import FlextProtocols

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx05:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class UserModel(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """User display name."""
            ...

        @property
        def status(self) -> c.Examples.StatusEnum:
            """User account status."""
            ...

        @property
        def age(self) -> int:
            """User age in years."""
            ...

    @runtime_checkable
    class HandlerBad(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class HandlerLike(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def data(self) -> p.ConfigMap:
            """Handler payload map."""
            ...

    @runtime_checkable
    class GoodProcessor(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class BadProcessor(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""


__all__: list[str] = [
    "ExamplesFlextProtocolsEx05",
]
