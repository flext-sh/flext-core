"""Utilities smoke tests aligned to stable public helpers."""

from __future__ import annotations

from tests import u


class Testu:
    def test_matches_type_string(self) -> None:
        assert u.matches_type("abc", "str")

    def test_generate_returns_non_empty_string(self) -> None:
        generated = u.generate("ulid")
        assert isinstance(generated, str)
        assert len(generated) > 0
