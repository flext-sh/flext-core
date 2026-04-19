"""Focused mapper utility tests aligned to current API."""

from __future__ import annotations

from tests import u


class TestUtilitiesMapperCoverage100:
    """Validate key extract/take paths."""

    def test_extract_reads_flat_mapping_value(self) -> None:
        data = {"a": 3}
        result = u.extract(data, "a")
        assert result.success
        assert result.value == 3

    def test_extract_returns_default_for_missing_key(self) -> None:
        result = u.extract({"a": 1}, "b", default=0)
        assert result.success
        assert result.value == 0
