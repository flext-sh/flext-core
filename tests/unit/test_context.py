"""Behavior contract for flext_core.FlextContext — public API only."""

from __future__ import annotations

from flext_core import FlextContext


class TestsFlextCoreContext:
    """Behavior contract for set/get/remove/clear/merge/clone/export."""

    def test_set_then_get_returns_success_with_original_value(self) -> None:
        ctx = FlextContext()
        assert ctx.set("key", "value").success
        got = ctx.get("key")
        assert got.success
        assert got.value == "value"

    def test_get_on_missing_key_returns_failure_result(self) -> None:
        assert FlextContext().get("missing").failure

    def test_remove_key_makes_subsequent_get_fail(self) -> None:
        ctx = FlextContext()
        ctx.set("k", "v")
        ctx.remove("k")
        assert ctx.get("k").failure

    def test_clear_removes_every_stored_key(self) -> None:
        ctx = FlextContext()
        ctx.set("a", "1")
        ctx.set("b", "2")
        ctx.clear()
        assert ctx.get("a").failure
        assert ctx.get("b").failure

    def test_merge_returns_same_context_and_adds_new_keys(self) -> None:
        ctx = FlextContext()
        ctx.set("existing", "x")
        merged = ctx.merge({"added": "y"})
        assert merged is ctx
        assert ctx.get("added").value == "y"
        assert ctx.get("existing").value == "x"

    def test_clone_is_independent_snapshot_of_original(self) -> None:
        ctx = FlextContext()
        ctx.set("k", "v")
        cloned = ctx.clone()
        ctx.clear()
        assert ctx.get("k").failure
        cloned_value = cloned.get("k")
        assert cloned_value.success
        assert cloned_value.value == "v"

    def test_initial_data_constructor_populates_store(self) -> None:
        ctx = FlextContext(initial_data={"a": 1})
        exported = ctx.export(as_dict=True)
        assert isinstance(exported, dict)
        assert "data" in exported or "a" in exported
