"""Result smoke tests for stable success/failure behavior."""

from __future__ import annotations

from tests import r


class TestResultCoverage100:
    def test_ok_result(self) -> None:
        result = r[int].ok(1)
        assert result.success
        assert result.value == 1

    def test_fail_result(self) -> None:
        result = r[int].fail("bad")
        assert result.failure
        assert result.error == "bad"
