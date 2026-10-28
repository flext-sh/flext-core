"""Behavioral tests for the runtime violation registry public contract.

Exercises the process-local buffer via its public classmethod surface
(``append_violation_report`` / ``drain_violation_reports`` /
``clear_violation_reports``) exposed by the public ``u`` facade. Every
assertion targets observable behavior a dispatcher caller relies on:
buffering, atomic drain-and-reset idempotence, FIFO ordering, content
fidelity, and silent clearing.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import m, tm, u


@pytest.mark.usefixtures("_isolated_buffer")
class TestsFlextCoreUtilitiesRuntimeViolationRegistry:
    """Public tests for the runtime violation registry buffer."""

    @staticmethod
    @pytest.fixture
    def _isolated_buffer() -> None:
        """Guarantee each test starts and ends with an empty buffer."""
        u.clear_violation_reports()

    @staticmethod
    def _report(message: str) -> m.Report:
        return m.Report(
            violations=[
                m.Violation(
                    qualname="tests.runtime.sample",
                    layer="Runtime",
                    severity="warning",
                    message=message,
                ),
            ],
        )

    @staticmethod
    def test_drain_on_empty_buffer_returns_empty_tuple() -> None:
        """Test drain on empty buffer returns empty tuple."""
        tm.that(u.drain_violation_reports(), eq=())

    def test_appended_report_is_returned_by_drain(self) -> None:
        """Test appended report is returned by drain."""
        report = self._report("captured violation")

        u.append_violation_report(report)
        drained = u.drain_violation_reports()

        tm.that(drained, eq=(report,))
        tm.that(drained[0].violations[0].message, eq="captured violation")

    def test_drain_resets_buffer_so_second_call_is_empty(self) -> None:
        """Test drain resets buffer so second call is empty."""
        u.append_violation_report(
            self._report("once"),
        )

        first = u.drain_violation_reports()
        second = u.drain_violation_reports()

        tm.that(len(first), eq=1)
        tm.that(second, eq=())

    def test_multiple_appends_drain_in_fifo_order(self) -> None:
        """Test multiple appends drain in fifo order."""
        reports = [self._report(f"v{index}") for index in range(3)]

        for report in reports:
            u.append_violation_report(report)
        drained = u.drain_violation_reports()

        tm.that([item.violations[0].message for item in drained], eq=["v0", "v1", "v2"])

    def test_clear_discards_buffered_reports_without_returning_them(self) -> None:
        """Test clear discards buffered reports without returning them."""
        u.append_violation_report(
            self._report("dropped"),
        )

        u.clear_violation_reports()

        tm.that(u.drain_violation_reports(), eq=())

    @staticmethod
    def test_clear_on_empty_buffer_is_safe_and_idempotent() -> None:
        """Test clear on empty buffer is safe and idempotent."""
        u.clear_violation_reports()
        u.clear_violation_reports()

        tm.that(u.drain_violation_reports(), eq=())

    @staticmethod
    def test_reports_with_empty_violations_are_buffered() -> None:
        """Test reports with empty violations are buffered."""
        empty_report = m.Report(violations=())

        u.append_violation_report(empty_report)
        drained = u.drain_violation_reports()

        tm.that(drained, eq=(empty_report,))
        tm.that(drained[0].violations, eq=())
