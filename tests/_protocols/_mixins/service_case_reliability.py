# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/service_case_reliability
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols

if TYPE_CHECKING:
    from flext_tests import p


class TestsFlextProtocolsMixinsServiceCaseReliability:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ReliabilityScenario(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Unique reliability scenario name."""
            ...

        @property
        def strategy(self) -> str:
            """Reliability strategy under test."""
            ...

        @property
        def settings(self) -> p.ConfigMap:
            """Reliability configuration payload."""
            ...

        @property
        def simulate_failures(self) -> int:
            """Number of failures to simulate."""
            ...

        @property
        def expected_state(self) -> str:
            """Expected strategy terminal state."""
            ...

        @property
        def should_succeed(self) -> bool:
            """Whether scenario expects successful outcome."""
            ...

        @property
        def description(self) -> str | None:
            """Human-readable scenario description."""
            ...


__all__: list[str] = [
    "TestsFlextProtocolsMixinsServiceCaseReliability",
]
