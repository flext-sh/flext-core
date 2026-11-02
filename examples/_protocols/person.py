# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/person
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import FlextProtocols


class ExamplesFlextProtocolsPerson:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Person(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Given name of the person."""
            ...

        @property
        def age(self) -> int:
            """Age in whole years."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsPerson",
]
