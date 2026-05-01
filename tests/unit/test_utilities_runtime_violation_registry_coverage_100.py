"""Runtime violation registry coverage tests.

Exercises the thread-safe buffer directly via
``FlextUtilitiesRuntimeViolationRegistry`` (not on ``u.*``).
"""

from __future__ import annotations

from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._utilities.runtime_violation_registry import (
    FlextUtilitiesRuntimeViolationRegistry as rvr,
)


class TestsFlextRuntimeViolationRegistry:
    def setup_method(self) -> None:
        rvr.clear_violation_reports()

    def _make_report(self, message: str = "test msg") -> me.Report:
        v = me.Violation(
            qualname="test.func",
            layer="domain",
            severity="error",
            message=message,
        )
        return me.Report(violations=[v])

    def test_drain_empty_returns_empty_tuple(self) -> None:
        result = rvr.drain_violation_reports()
        assert result == ()

    def test_clear_on_empty_registry_is_safe(self) -> None:
        rvr.clear_violation_reports()
        assert rvr.drain_violation_reports() == ()

    def test_append_then_drain_returns_one_report(self) -> None:
        rpt = self._make_report("first")
        rvr.append_violation_report(rpt)
        drained = rvr.drain_violation_reports()
        assert len(drained) == 1
        assert drained[0].violations[0].message == "first"

    def test_drain_is_destructive(self) -> None:
        rvr.append_violation_report(self._make_report())
        rvr.drain_violation_reports()
        second = rvr.drain_violation_reports()
        assert second == ()

    def test_clear_discards_without_returning(self) -> None:
        rvr.append_violation_report(self._make_report())
        rvr.clear_violation_reports()
        assert rvr.drain_violation_reports() == ()

    def test_append_multiple_reports_preserves_order(self) -> None:
        for i in range(3):
            rvr.append_violation_report(self._make_report(f"msg-{i}"))
        drained = rvr.drain_violation_reports()
        assert len(drained) == 3
        assert drained[0].violations[0].message == "msg-0"
        assert drained[1].violations[0].message == "msg-1"
        assert drained[2].violations[0].message == "msg-2"

    def test_drain_returns_tuple(self) -> None:
        rvr.append_violation_report(self._make_report())
        result = rvr.drain_violation_reports()
        assert isinstance(result, tuple)

    def test_report_with_empty_violations_is_accepted(self) -> None:
        empty_report = me.Report(violations=())
        rvr.append_violation_report(empty_report)
        drained = rvr.drain_violation_reports()
        assert len(drained) == 1
        assert drained[0].violations == ()
