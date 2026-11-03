# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/container
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols

if TYPE_CHECKING:
    from tests import t


class TestsFlextProtocolsMixinsContainer:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ServiceScenario(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Service scenario name."""
            ...

        @property
        def service(self) -> t.Primitives:
            """Service value to register."""
            ...

        @property
        def description(self) -> str:
            """Scenario description."""
            ...

    @runtime_checkable
    class TypedRetrievalScenario(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Typed retrieval scenario name."""
            ...

        @property
        def service(self) -> str | int:
            """Registered service value."""
            ...

        @property
        def expected_type(self) -> type[str | int]:
            """Expected service type."""
            ...

        @property
        def should_pass(self) -> bool:
            """Whether typed retrieval should succeed."""
            ...

        @property
        def description(self) -> str:
            """Scenario description."""
            ...


__all__: list[str] = [
    "TestsFlextProtocolsMixinsContainer",
]
