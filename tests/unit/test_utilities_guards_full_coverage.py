"""Concise guard utility tests aligned to current contracts."""

from __future__ import annotations

from tests import u


class TestUtilitiesGuardsFullCoverage:
    """Coverage checks for stable guard behavior."""

    def test_matches_type_for_common_scalar_guards(self) -> None:
        assert u.matches_type("abc", "str")
        assert u.matches_type(10, "int")
        assert not u.matches_type("abc", "int")

    def test_container_guard_accepts_flat_container_values(self) -> None:
        assert u.container({"a": 1})
        assert u.container([1, "x", True])

    def test_dict_and_string_non_empty_helpers(self) -> None:
        assert u.dict_non_empty({"k": "v"})
        assert u.string_non_empty("ok")
        assert not u.string_non_empty("")
