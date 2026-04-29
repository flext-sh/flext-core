"""Mapper utility smoke tests aligned to current public mapper helpers."""

from __future__ import annotations

from tests import u


class TestsFlextUtilitiesMapper:
    def test_extract_simple(self) -> None:
        result = u.extract({"a": 1}, "a")
        assert result.success
        assert result.value == 1

    def test_extract_default(self) -> None:
        result = u.extract({"a": 1}, "b", default=0)
        assert result.success
        assert result.value == 0
