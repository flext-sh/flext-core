"""Behavior contract for FlextContext — pure Pydantic v2 model + contextvar facade."""

from __future__ import annotations

from flext_core import FlextContext


class TestsFlextContext:
    """Contract: scope store (set/get/remove/clear/merge/clone/export) + contextvar ops."""

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

    def test_merge_accepts_another_context_through_protocol_surface(self) -> None:
        ctx = FlextContext()
        ctx.set("existing", "x")
        other = FlextContext()
        other.set("added", "y")

        merged = ctx.merge(other)

        assert merged is ctx
        assert ctx.get("existing").value == "x"
        assert ctx.get("added").value == "y"

    def test_merge_overwrites_existing_key_with_new_value(self) -> None:
        ctx = FlextContext()
        ctx.set("existing", "x")

        ctx.merge({"existing": "y"})

        assert ctx.get("existing").success
        assert ctx.get("existing").value == "y"

    def test_clone_is_independent_snapshot_of_original(self) -> None:
        ctx = FlextContext()
        ctx.set("k", "v")
        cloned = ctx.clone()
        ctx.clear()
        assert ctx.get("k").failure
        assert cloned.get("k").success
        assert cloned.get("k").value == "v"

    def test_export_returns_dict_with_stored_keys(self) -> None:
        ctx = FlextContext()
        ctx.set("a", 1)
        ctx.set("b", "two")
        exported = ctx.export(as_dict=True)
        assert isinstance(exported, dict)
        assert exported["a"] == 1
        assert exported["b"] == "two"

    def test_create_with_initial_values_seeds_the_context_scope(self) -> None:
        ctx = FlextContext.create(operation_id="op-1", user_id="user-1")

        assert ctx.get("operation_id").unwrap_or("") == "op-1"
        assert ctx.get("user_id").unwrap_or("") == "user-1"

    def test_apply_metadata_then_resolve_metadata_returns_stored_value(self) -> None:
        ctx = FlextContext()

        ctx.apply_metadata("source", "api")

        resolved = ctx.resolve_metadata("source")
        assert resolved.success
        assert resolved.value == "api"

    def test_contextvar_apply_and_resolve_correlation_id(self) -> None:
        FlextContext.apply_correlation_id("test-corr-123")
        assert FlextContext.resolve_correlation_id() == "test-corr-123"
        FlextContext.clear_context()
        assert FlextContext.resolve_correlation_id() is None

    def test_ensure_correlation_id_generates_if_absent(self) -> None:
        FlextContext.clear_context()
        cid = FlextContext.ensure_correlation_id()
        assert isinstance(cid, str) and len(cid) > 0
        assert FlextContext.resolve_correlation_id() == cid
        FlextContext.clear_context()
