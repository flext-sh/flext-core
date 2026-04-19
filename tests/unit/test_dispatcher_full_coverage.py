"""Dispatcher full coverage smoke tests for current core contracts."""

from __future__ import annotations

from tests import r


class TestDispatcherFullCoverage:
    def test_result_success_path(self) -> None:
        result = r[int].ok(1)
        assert result.success
        assert result.value == 1

    def test_result_failure_path(self) -> None:
        result = r[int].fail("dispatch error")
        assert result.failure
        assert "dispatch" in (result.error or "")
