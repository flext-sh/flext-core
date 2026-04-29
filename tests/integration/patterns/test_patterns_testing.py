"""Testing pattern smoke tests."""

from __future__ import annotations

from tests import r


class TestsFlextPatternsTesting:
    def test_pattern_success(self) -> None:
        result = r[str].ok("tested")
        assert result.success

    def test_pattern_failure(self) -> None:
        result = r[str].fail("failed")
        assert result.failure
