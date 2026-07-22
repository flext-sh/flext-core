"""Behavioral tests for the runtime violation registry public contract.

Exercises the process-local buffer via its public classmethod surface
(``append_violation_report`` / ``drain_violation_reports`` /
``clear_violation_reports``) exported at the ``flext_core`` root. Every
assertion targets observable behavior a dispatcher caller relies on:
buffering, atomic drain-and-reset idempotence, FIFO ordering, content
fidelity, and silent clearing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.models import m

if TYPE_CHECKING:
    # pyrefly cannot resolve the lazy root export; bind the concrete class for
    # static typing while the runtime import below exercises the public surface.
    from flext_core._utilities.runtime_violation_registry import (
        FlextUtilitiesRuntimeViolationRegistry as runtime_registry,
    )
else:
    from flext_core._utilities.runtime_violation_registry import (
        FlextUtilitiesRuntimeViolationRegistry as runtime_registry,
    )


class TestsFlextCoreUtilitiesRuntimeViolationRegistry:
    """Public tests for the runtime violation registry buffer."""

    @pytest.fixture
    def _isolated_buffer(self) -> None:
        """Guarantee each test starts and ends with an empty buffer."""
        runtime_registry.clear_violation_reports()

    @staticmethod
    def _report(message: str) -> m.Report:
        return m.Report(
            violations=[
                m.Violation(
                    qualname="tests.runtime.sample",
                    layer="Runtime",
                    severity="warning",
                    message=message,
                )
            ]
        )

    def test_drain_on_empty_buffer_returns_empty_tuple(
        self, _isolated_buffer: None
    ) -> None:
        assert runtime_registry.drain_violation_reports() == ()

    def test_appended_report_is_returned_by_drain(self, _isolated_buffer: None) -> None:
        report = self._report("captured violation")

        runtime_registry.append_violation_report(report)
        drained = runtime_registry.drain_violation_reports()

        assert drained == (report,)
        assert drained[0].violations[0].message == "captured violation"

    def test_drain_resets_buffer_so_second_call_is_empty(
        self, _isolated_buffer: None
    ) -> None:
        runtime_registry.append_violation_report(self._report("once"))

        first = runtime_registry.drain_violation_reports()
        second = runtime_registry.drain_violation_reports()

        assert len(first) == 1
        assert second == ()

    def test_multiple_appends_drain_in_fifo_order(self, _isolated_buffer: None) -> None:
        reports = [self._report(f"v{index}") for index in range(3)]

        for report in reports:
            runtime_registry.append_violation_report(report)
        drained = runtime_registry.drain_violation_reports()

        assert [item.violations[0].message for item in drained] == ["v0", "v1", "v2"]

    def test_clear_discards_buffered_reports_without_returning_them(
        self, _isolated_buffer: None
    ) -> None:
        runtime_registry.append_violation_report(self._report("dropped"))

        runtime_registry.clear_violation_reports()

        assert runtime_registry.drain_violation_reports() == ()

    def test_clear_on_empty_buffer_is_safe_and_idempotent(
        self, _isolated_buffer: None
    ) -> None:
        runtime_registry.clear_violation_reports()
        runtime_registry.clear_violation_reports()

        assert runtime_registry.drain_violation_reports() == ()

    def test_reports_with_empty_violations_are_buffered(
        self, _isolated_buffer: None
    ) -> None:
        empty_report = m.Report(violations=())

        runtime_registry.append_violation_report(empty_report)
        drained = runtime_registry.drain_violation_reports()

        assert drained == (empty_report,)
        assert drained[0].violations == ()
