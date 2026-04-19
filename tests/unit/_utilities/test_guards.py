"""Guard utility smoke tests for internal utility tier."""

from __future__ import annotations

from tests import u


class TestUtilitiesGuards:
    def test_matches_type_basic(self) -> None:
        assert u.matches_type("x", "str")
        assert u.matches_type(1, "int")

    def test_container_guard_basic(self) -> None:
        assert u.container({"k": "v"})
        assert u.container([1, "x", True])
