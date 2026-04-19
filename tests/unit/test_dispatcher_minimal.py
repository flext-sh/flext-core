"""Minimal dispatcher-aligned tests using stable result contract."""

from __future__ import annotations

from tests import r


class TestDispatcherMinimal:
    def test_minimal_success(self) -> None:
        result = r[str].ok("dispatch")
        assert result.success

    def test_minimal_failure(self) -> None:
        result = r[str].fail("failed")
        assert result.failure
