"""Context management smoke tests for current public context surface."""

from __future__ import annotations

from flext_core import FlextContext


class TestFlextContext:
    def test_context_set_get_remove(self) -> None:
        ctx = FlextContext()
        assert ctx.set("key", "value").success
        got = ctx.get("key")
        assert got.success
        assert got.value == "value"
        ctx.remove("key")
        assert ctx.get("key").failure

    def test_context_export_dict(self) -> None:
        ctx = FlextContext(initial_data={"a": 1})
        exported = ctx.export(as_dict=True)
        assert isinstance(exported, dict)
        assert "data" in exported or "a" in exported
