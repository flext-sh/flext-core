"""Runtime violation registry behavior through the public root export."""

from __future__ import annotations

from flext_core import FlextUtilitiesRuntimeViolationRegistry as runtime_registry
from tests import m


class TestsFlextRuntimeViolationRegistry:
    """Public tests for the runtime violation registry buffer."""

    def test_append_then_drain_is_idempotent(self) -> None:
        runtime_registry.clear_violation_reports()
        report = m.Report(
            violations=[
                m.Violation(
                    qualname="tests.runtime.sample",
                    layer="Runtime",
                    severity="warning",
                    message="captured violation",
                )
            ]
        )

        runtime_registry.append_violation_report(report)

        drained = runtime_registry.drain_violation_reports()
        assert len(drained) == 1
        assert drained[0] is report
        assert drained[0].violations[0].message == "captured violation"
        assert runtime_registry.drain_violation_reports() == ()

    def test_clear_discards_buffered_reports(self) -> None:
        report = m.Report(violations=())
        runtime_registry.append_violation_report(report)

        runtime_registry.clear_violation_reports()

        assert runtime_registry.drain_violation_reports() == ()
