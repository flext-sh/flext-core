# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/domain
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols


class TestsFlextProtocolsMixinsDomain:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class EmailResponse(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def status(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def message_id(self) -> str:
            """Structural member projected from the owning model."""
            ...

    @runtime_checkable
    class NoDict(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        value: int

    @runtime_checkable
    class ParseOptions(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def strip(self) -> bool:
            """Structural member projected from the owning model."""
            ...

        @property
        def remove_empty(self) -> bool:
            """Structural member projected from the owning model."""
            ...

        @property
        def validator(self) -> Callable[[str], bool] | None:
            """Structural member projected from the owning model."""
            ...


__all__: list[str] = [
    "TestsFlextProtocolsMixinsDomain",
]
