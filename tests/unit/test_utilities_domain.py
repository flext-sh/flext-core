"""Utilities domain smoke tests."""

from __future__ import annotations

from tests import u


class TestsFlextUtilitiesDomain:
    def test_normalize_keeps_plain_text(self) -> None:
        assert u.normalize("abc") == "abc"

    def test_join_works_for_sequence(self) -> None:
        assert u.join(["a", "b"], case="lower") == "a b"
