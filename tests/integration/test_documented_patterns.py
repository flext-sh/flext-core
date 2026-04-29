"""Documented pattern integration smoke tests."""

from __future__ import annotations

from tests import p, r


class TestsFlextDocumentedPatterns:
    def test_result_map_pattern(self) -> None:
        result = r[int].ok(1).map(lambda value: value + 1)
        assert result.success
        assert result.value == 2

    def test_result_flow_through_pattern(self) -> None:
        def step(value: int) -> p.Result[int]:
            return r[int].ok(value + 1)

        result = r[int].ok(1).flow_through(step)
        assert result.success
        assert result.value == 2
