"""Registry smoke tests aligned to current stable contracts."""

from __future__ import annotations

from tests import r


class TestRegistry:
    def test_registry_like_success_contract(self) -> None:
        result = r[str].ok("registered")
        assert result.success

    def test_registry_like_failure_contract(self) -> None:
        result = r[str].fail("missing")
        assert result.failure
