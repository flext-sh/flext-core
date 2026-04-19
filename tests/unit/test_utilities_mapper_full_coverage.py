"""Mapper utility smoke tests for current extractor behavior."""

from __future__ import annotations

from tests import u


class TestUtilitiesMapperFullCoverage:
    def test_extract_simple_key(self) -> None:
        result = u.extract({"a": 1}, "a")
        assert result.success
        assert result.value == 1

    def test_extract_nested_key(self) -> None:
        result = u.extract({"a": {"b": 2}}, "a.b")
        assert result.success
        assert result.value == 2

    def test_extract_missing_key_with_default(self) -> None:
        result = u.extract({"a": 1}, "x", default=0)
        assert result.success
        assert result.value == 0
