"""Collection utilities smoke tests."""

from __future__ import annotations

from tests import u


class TestUtilitiesCollectionFullCoverage:
    def test_mapping_guard(self) -> None:
        assert u.mapping({"a": 1})

    def test_dict_non_empty_guard(self) -> None:
        assert u.dict_non_empty({"k": "v"})
