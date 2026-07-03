"""Guard utility smoke tests for internal utility tier."""

from __future__ import annotations

from tests.utilities import u


class TestsFlextUtilitiesGuards:
    def test_matches_type_basic(self) -> None:
        assert u.matches_type("x", "str")
        assert u.matches_type(1, "int")

    def test_matches_type_collection_strings(self) -> None:
        assert u.matches_type({"k": "v"}, "mapping")
        assert u.matches_type({"k": "v"}, "dict_non_empty")
        assert not u.matches_type({}, "dict_non_empty")
        assert u.matches_type(["v"], "list_non_empty")
        assert not u.matches_type("", "string_non_empty")

    def test_container_guard_basic(self) -> None:
        assert u.container({"k": "v"})
        assert u.container([1, "x", True])
