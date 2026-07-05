"""Behavior contract for FlextContext — pure Pydantic v2 model + contextvar facade.

Tests exercise the PUBLIC surface only: instance scope store (set/get/has/keys/
values/items/remove/clear/merge/clone/export), metadata I/O, and the class-level
contextvar facade (correlation id / operation name / scoped context managers /
full-context export). No private attributes, no internal collaborators.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from flext_core import FlextContext


class TestsFlextCoreContext:
    """Contract: scope store + metadata + process-global contextvar operations."""

    @pytest.fixture(autouse=True)
    def _isolate_process_context(self) -> Iterator[None]:
        """Keep process-global contextvars from leaking across tests."""
        FlextContext.clear_context()
        yield
        FlextContext.clear_context()

    # --- scope store -----------------------------------------------------

    def test_set_then_get_returns_success_with_original_value(self) -> None:
        ctx = FlextContext()
        assert ctx.set("key", "value").success
        got = ctx.get("key")
        assert got.success
        assert got.value == "value"

    def test_get_on_missing_key_returns_failure_with_message(self) -> None:
        result = FlextContext().get("missing")
        assert result.failure
        assert "missing" in str(result.error)

    def test_has_reflects_presence_of_key(self) -> None:
        ctx = FlextContext()
        assert ctx.has("k") is False
        ctx.set("k", "v")
        assert ctx.has("k") is True

    def test_remove_key_makes_subsequent_get_fail(self) -> None:
        ctx = FlextContext()
        ctx.set("k", "v")
        ctx.remove("k")
        assert ctx.get("k").failure

    def test_remove_absent_key_is_noop(self) -> None:
        ctx = FlextContext()
        ctx.remove("never-added")
        assert ctx.has("never-added") is False

    def test_clear_removes_every_stored_key(self) -> None:
        ctx = FlextContext()
        ctx.set("a", "1")
        ctx.set("b", "2")
        ctx.clear()
        assert ctx.get("a").failure
        assert ctx.get("b").failure

    def test_keys_values_items_reflect_stored_pairs(self) -> None:
        ctx = FlextContext()
        ctx.set("a", 1)
        ctx.set("b", "two")

        assert set(ctx.keys()) == {"a", "b"}
        assert set(ctx.values()) == {1, "two"}
        assert dict(ctx.items()) == {"a": 1, "b": "two"}

    # --- merge / clone / export -----------------------------------------

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

        assert ctx.get("existing").value == "y"

    def test_clone_is_independent_snapshot_of_original(self) -> None:
        ctx = FlextContext()
        ctx.set("k", "v")
        cloned = ctx.clone()
        ctx.clear()
        assert ctx.get("k").failure
        assert cloned.get("k").success
        assert cloned.get("k").value == "v"

    def test_mutating_clone_does_not_affect_original(self) -> None:
        ctx = FlextContext()
        ctx.set("k", "v")
        cloned = ctx.clone()
        cloned.set("k", "changed")
        assert ctx.get("k").value == "v"
        assert cloned.get("k").value == "changed"

    def test_export_as_dict_returns_stored_keys(self) -> None:
        ctx = FlextContext()
        ctx.set("a", 1)
        ctx.set("b", "two")
        exported = ctx.export(as_dict=True)
        assert isinstance(exported, dict)
        assert exported == {"a": 1, "b": "two"}

    def test_export_without_dict_returns_context_itself(self) -> None:
        ctx = FlextContext()
        ctx.set("a", 1)
        assert ctx.export(as_dict=False) is ctx

    def test_create_with_initial_values_seeds_the_context_scope(self) -> None:
        ctx = FlextContext.create(operation_id="op-1", user_id="user-1")

        assert ctx.get("operation_id").unwrap_or("") == "op-1"
        assert ctx.get("user_id").unwrap_or("") == "user-1"

    # --- metadata --------------------------------------------------------

    def test_apply_metadata_then_resolve_metadata_returns_stored_value(self) -> None:
        ctx = FlextContext()

        ctx.apply_metadata("source", "api")

        resolved = ctx.resolve_metadata("source")
        assert resolved.success
        assert resolved.value == "api"

    def test_resolve_metadata_missing_key_returns_failure(self) -> None:
        result = FlextContext().resolve_metadata("absent")
        assert result.failure
        assert "absent" in str(result.error)

    # --- correlation-id contextvar facade --------------------------------

    def test_apply_and_resolve_correlation_id_roundtrip(self) -> None:
        FlextContext.apply_correlation_id("test-corr-123")
        assert FlextContext.resolve_correlation_id() == "test-corr-123"

    def test_clear_context_drops_correlation_id(self) -> None:
        FlextContext.apply_correlation_id("test-corr-123")
        FlextContext.clear_context()
        assert FlextContext.resolve_correlation_id() is None

    def test_ensure_correlation_id_generates_when_absent(self) -> None:
        cid = FlextContext.ensure_correlation_id()
        assert isinstance(cid, str)
        assert cid
        assert FlextContext.resolve_correlation_id() == cid

    def test_ensure_correlation_id_preserves_existing_value(self) -> None:
        FlextContext.apply_correlation_id("already-here")
        assert FlextContext.ensure_correlation_id() == "already-here"

    def test_new_correlation_scopes_and_restores_previous_id(self) -> None:
        FlextContext.apply_correlation_id("outer")
        with FlextContext.new_correlation("inner") as active:
            assert active == "inner"
            assert FlextContext.resolve_correlation_id() == "inner"
        assert FlextContext.resolve_correlation_id() == "outer"

    def test_new_correlation_generates_id_when_none_supplied(self) -> None:
        with FlextContext.new_correlation() as active:
            assert isinstance(active, str)
            assert active
            assert FlextContext.resolve_correlation_id() == active

    # --- operation-name contextvar facade --------------------------------

    def test_apply_and_resolve_operation_name_roundtrip(self) -> None:
        FlextContext.apply_operation_name("sync-users")
        assert FlextContext.resolve_operation_name() == "sync-users"

    def test_resolve_operation_name_is_none_when_unset(self) -> None:
        assert FlextContext.resolve_operation_name() is None

    def test_timed_operation_yields_metadata_with_operation_name(self) -> None:
        with FlextContext.timed_operation("etl") as meta:
            assert FlextContext.resolve_operation_name() == "etl"
            assert meta.root
        assert FlextContext.resolve_operation_name() is None

    # --- service + full-context export -----------------------------------

    def test_service_context_scopes_service_name(self) -> None:
        with FlextContext.service_context("billing", version="1.2.0"):
            exported = FlextContext.export_full_context()
            assert exported["service_name"] == "billing"
            assert exported["service_version"] == "1.2.0"
        assert "service_name" not in FlextContext.export_full_context()

    def test_export_full_context_includes_active_correlation_id(self) -> None:
        FlextContext.apply_correlation_id("corr-xyz")
        exported = FlextContext.export_full_context()
        assert exported["correlation_id"] == "corr-xyz"

    def test_export_full_context_is_empty_after_clear(self) -> None:
        FlextContext.apply_correlation_id("corr-xyz")
        FlextContext.clear_context()
        assert FlextContext.export_full_context() == {}
