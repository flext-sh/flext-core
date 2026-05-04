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
    """Public tests for the runtime violation registry buffer."""

    def test_append_then_drain_is_idempotent(self) -> None:
        rvr.clear_violation_reports()
        report = me.Report(violations=())

        rvr.append_violation_report(report)

        drained = rvr.drain_violation_reports()
        assert len(drained) == 1
        assert drained[0] is report
        assert rvr.drain_violation_reports() == ()

    def test_clear_discards_buffered_reports(self) -> None:
        report = me.Report(violations=())
        rvr.append_violation_report(report)

        rvr.clear_violation_reports()

        assert rvr.drain_violation_reports() == ()
