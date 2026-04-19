"""Context coverage smoke tests."""

from __future__ import annotations

from flext_core import FlextContext


class TestCoverageContext:
    def test_set_and_get_context_value(self) -> None:
        ctx = FlextContext()
        assert ctx.set("k", "v").success
        got = ctx.get("k")
        assert got.success
        assert got.value == "v"

    def test_get_missing_context_value_fails(self) -> None:
        ctx = FlextContext()
        assert ctx.get("missing").failure
