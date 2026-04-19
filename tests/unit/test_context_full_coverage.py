"""Context behavior smoke tests for stable public APIs."""

from __future__ import annotations

from flext_core import FlextContext


class TestContextFullCoverage:
    def test_create_and_get_context_value(self) -> None:
        ctx = FlextContext(user_id="u1")
        result = ctx.get("user_id")
        assert result.success
        assert result.value == "u1"

    def test_set_merge_clone_clear_flow(self) -> None:
        ctx = FlextContext()
        assert ctx.set("k1", "v1").success
        merged = ctx.merge({"k2": "v2"})
        assert merged is ctx
        cloned = ctx.clone()
        ctx.clear()
        assert ctx.get("k1").failure
        cloned_value = cloned.get("k1")
        assert cloned_value.success
        assert cloned_value.value == "v1"
