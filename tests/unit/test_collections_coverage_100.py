"""Collection utility smoke tests."""

from __future__ import annotations

from tests import u


class TestCollectionsCoverage100:
    def test_container_mapping_guard(self) -> None:
        assert u.mapping({"a": 1})

    def test_non_empty_dict_guard(self) -> None:
        assert u.dict_non_empty({"k": "v"})
