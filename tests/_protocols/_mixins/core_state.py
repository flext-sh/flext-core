# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/core_state
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols


class TestsFlextProtocolsMixinsCoreState:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class SingletonClassForTest(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        name: str
        timeout: int

        @classmethod
        def fetch_global(
            cls,
        ) -> TestsFlextProtocolsMixinsCoreState.SingletonClassForTest:
            """Get global singleton instance."""
            ...

        @classmethod
        def reset_instance(cls) -> None:
            """Reset singleton instance for test isolation."""
            ...

    @runtime_checkable
    class _SampleEntity(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        unique_id: str
        name: str


__all__: list[str] = [
    "TestsFlextProtocolsMixinsCoreState",
]
