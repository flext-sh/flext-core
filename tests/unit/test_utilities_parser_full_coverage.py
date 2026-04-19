"""Parser utility smoke tests for stable conversion helpers."""

from __future__ import annotations

from tests import u


class TestUtilitiesParserFullCoverage:
    def test_parse_int_success(self) -> None:
        result = u.parse("10", int)
        assert result.success
        assert result.value == 10

    def test_parse_bool_success(self) -> None:
        result = u.parse("true", bool)
        assert result.success
        assert result.value is True

    def test_parse_invalid_value_returns_failure(self) -> None:
        result = u.parse("not-an-int", int)
        assert result.failure
