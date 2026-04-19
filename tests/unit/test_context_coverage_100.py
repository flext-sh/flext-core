"""Context 100 coverage smoke tests aligned to current context API."""

from __future__ import annotations

from flext_core import FlextContext


class TestContext100Coverage:
    def test_set_get_remove_cycle(self) -> None:
        context = FlextContext()
        assert context.set("k", "v").success
        assert context.get("k").success
        context.remove("k")
        assert context.get("k").failure

    def test_clear_removes_data(self) -> None:
        context = FlextContext()
        context.set("a", "1")
        context.set("b", "2")
        context.clear()
        assert context.get("a").failure
        assert context.get("b").failure
