"""Public exception metrics tests."""

from __future__ import annotations

from flext_tests import e

from tests.models import m


class TestsFlextCoverageExceptionMetrics:
    def test_metrics_are_exposed_through_public_behavior(self) -> None:
        e.clear_metrics()
        for exception_type in (
            e.ValidationError,
            e.ValidationError,
            e.TimeoutError,
            e.TimeoutError,
            e.TimeoutError,
        ):
            e.record_exception(exception_type)

        metrics = e.resolve_metrics_snapshot()
        assert isinstance(metrics, m.ExceptionMetricsSnapshot)
        assert metrics.total_exceptions == 5
        assert metrics.unique_exception_types == 2
        assert metrics.exception_counts[e.ValidationError.__qualname__] == 2
        assert metrics.exception_counts[e.TimeoutError.__qualname__] == 3
        summary = metrics.exception_counts_summary
        assert "ValidationError:2" in summary
        assert "TimeoutError:3" in summary

        e.clear_metrics()
        cleared = e.resolve_metrics_snapshot()
        assert cleared.total_exceptions == 0
        assert cleared.unique_exception_types == 0
        assert cleared.exception_counts_summary == ""
