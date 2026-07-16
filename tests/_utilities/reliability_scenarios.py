"""Reliability scenario helpers for flext-core tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tests.models import m

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.typings import t


class TestsFlextUtilitiesReliabilityScenariosMixin:
    """Reliability scenario helpers."""

    class ReliabilityScenarios:
        """Centralized reliability scenarios - single source of truth."""

        _RETRY_BASE_SETTINGS: ClassVar[p.ConfigMap] = m.ConfigMap(
            root={"max_retries": 3, "backoff_type": "constant", "backoff_ms": 10},
        )
        _RETRY_EXHAUSTED_SETTINGS: ClassVar[p.ConfigMap] = m.ConfigMap(
            root={"max_retries": 2, "backoff_type": "constant", "backoff_ms": 10},
        )

        RETRY_SCENARIOS: ClassVar[Sequence[p.Tests.ReliabilityScenario]] = [
            m.Tests.ReliabilityScenario(
                name="retry_immediate_success",
                strategy="retry",
                settings=_RETRY_BASE_SETTINGS,
                simulate_failures=0,
                expected_state="success",
                should_succeed=True,
                description="Operation succeeds immediately",
            ),
            m.Tests.ReliabilityScenario(
                name="retry_after_one_failure",
                strategy="retry",
                settings=_RETRY_BASE_SETTINGS,
                simulate_failures=1,
                expected_state="success",
                should_succeed=True,
                description="Succeeds after one retry",
            ),
            m.Tests.ReliabilityScenario(
                name="retry_exhausted",
                strategy="retry",
                settings=_RETRY_EXHAUSTED_SETTINGS,
                simulate_failures=5,
                expected_state="exhausted",
                should_succeed=False,
                description="All retries exhausted",
            ),
        ]
        CIRCUIT_BREAKER_SCENARIOS: ClassVar[
            t.SequenceOf[p.Tests.ReliabilityScenario]
        ] = [
            m.Tests.ReliabilityScenario(
                name="circuit_initial_closed",
                strategy="circuit_breaker",
                settings=m.ConfigMap(root={"failure_threshold": 5, "timeout_ms": 1000}),
                simulate_failures=0,
                expected_state="closed",
                should_succeed=True,
                description="Circuit starts in closed state",
            ),
            m.Tests.ReliabilityScenario(
                name="circuit_open_on_threshold",
                strategy="circuit_breaker",
                settings=m.ConfigMap(root={"failure_threshold": 2, "timeout_ms": 1000}),
                simulate_failures=3,
                expected_state="open",
                should_succeed=False,
                description="Circuit opens after threshold",
            ),
        ]


__all__: list[str] = ["TestsFlextUtilitiesReliabilityScenariosMixin"]
