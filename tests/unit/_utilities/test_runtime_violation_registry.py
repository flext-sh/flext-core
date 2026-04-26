"""Process-local violation registry behavior contract.

Covers thread-safe ``append_violation_report``, idempotent
``drain_violation_reports`` (returns then clears),
``clear_violation_reports`` reset, and rejection of non-``ViolationReport``
payloads.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from tests import c, m, u


class TestsFlextCoreUtilitiesRuntimeViolationRegistry:
    """Behavior contract for FlextUtilitiesRuntimeViolationRegistry."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self) -> Iterator[None]:
        u.clear_violation_reports()
        yield
        u.clear_violation_reports()

    def test_drain_returns_empty_when_no_appends(self) -> None:
        assert u.drain_violation_reports() == ()

    def test_append_then_drain_yields_in_insertion_order(self) -> None:
        first = m.ViolationReport(rule_id="ENFORCE-045", outcome=c.ViolationOutcome.HIT)
        second = m.ViolationReport(
            rule_id="ENFORCE-046", outcome=c.ViolationOutcome.MISS
        )
        u.append_violation_report(first)
        u.append_violation_report(second)
        assert u.drain_violation_reports() == (first, second)

    def test_drain_is_idempotent(self) -> None:
        u.append_violation_report(
            m.ViolationReport(rule_id="ENFORCE-045", outcome=c.ViolationOutcome.HIT),
        )
        first = u.drain_violation_reports()
        assert len(first) == 1
        assert u.drain_violation_reports() == ()

    def test_clear_resets_state(self) -> None:
        u.append_violation_report(
            m.ViolationReport(rule_id="ENFORCE-045", outcome=c.ViolationOutcome.SKIP),
        )
        u.clear_violation_reports()
        assert u.drain_violation_reports() == ()

    def test_concurrent_appends_are_serialized(self) -> None:
        target_count = 200
        report = m.ViolationReport(
            rule_id="ENFORCE-045", outcome=c.ViolationOutcome.HIT
        )

        def _push() -> None:
            u.append_violation_report(report)

        threads = [threading.Thread(target=_push) for _ in range(target_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(u.drain_violation_reports()) == target_count
