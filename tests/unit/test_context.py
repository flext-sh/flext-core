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
from flext_tests import tm


class TestsFlextCoreContext:
    """Contract: scope store + metadata + process-global contextvar operations."""

    pytestmark = pytest.mark.usefixtures("_isolate_process_context")

    @pytest.fixture
    def _isolate_process_context(self) -> Iterator[None]:
        """Keep process-global contextvars from leaking across tests."""
        FlextContext.clear_context()
        yield
        FlextContext.clear_context()

    # --- scope store -----------------------------------------------------

    def test_set_then_get_returns_success_with_original_value(self) -> None:
        """Return the original value after storing it in the scope."""
        ctx = FlextContext()
        tm.ok(ctx.set("key", "value"))
        got = ctx.get("key")
        tm.ok(got, eq="value")

    def test_get_on_missing_key_returns_failure_with_message(self) -> None:
        """Report the missing key when scope lookup fails."""
        result = FlextContext().get("missing")
        tm.fail(result, has="missing")

    def test_has_reflects_presence_of_key(self) -> None:
        """Reflect key absence and presence through the public query."""
        ctx = FlextContext()
        tm.that(ctx.has("k"), eq=False)
        ctx.set("k", "v")
        tm.that(ctx.has("k"), eq=True)

    def test_remove_key_makes_subsequent_get_fail(self) -> None:
        """Make subsequent lookup fail after removing a stored key."""
        ctx = FlextContext()
        ctx.set("k", "v")
        ctx.remove("k")
        tm.fail(ctx.get("k"))

    def test_remove_absent_key_is_noop(self) -> None:
        """Leave the scope unchanged when removing an absent key."""
        ctx = FlextContext()
        ctx.remove("never-added")
        tm.that(ctx.has("never-added"), eq=False)

    def test_clear_removes_every_stored_key(self) -> None:
        """Remove every stored key when clearing the scope."""
        ctx = FlextContext()
        ctx.set("a", "1")
        ctx.set("b", "2")
        ctx.clear()
        tm.fail(ctx.get("a"))
        tm.fail(ctx.get("b"))

    def test_keys_values_items_reflect_stored_pairs(self) -> None:
        """Expose keys, values, and items for the stored pairs."""
        ctx = FlextContext()
        ctx.set("a", 1)
        ctx.set("b", "two")

        tm.that(set(ctx.keys()), eq={"a", "b"})
        tm.that(set(ctx.values()), eq={1, "two"})
        tm.that(dict(ctx.items()), eq={"a": 1, "b": "two"})

    # --- merge / clone / export -----------------------------------------

    def test_merge_returns_same_context_and_adds_new_keys(self) -> None:
        """Mutate and return the receiving context when merging a mapping."""
        ctx = FlextContext()
        ctx.set("existing", "x")
        merged = ctx.merge({"added": "y"})
        tm.that(merged is ctx, eq=True)
        tm.ok(ctx.get("added"), eq="y")
        tm.ok(ctx.get("existing"), eq="x")

    def test_merge_accepts_another_context_through_protocol_surface(self) -> None:
        """Merge another context through the public protocol surface."""
        ctx = FlextContext()
        ctx.set("existing", "x")
        other = FlextContext()
        other.set("added", "y")

        merged = ctx.merge(other)

        tm.that(merged is ctx, eq=True)
        tm.ok(ctx.get("existing"), eq="x")
        tm.ok(ctx.get("added"), eq="y")

    def test_merge_overwrites_existing_key_with_new_value(self) -> None:
        """Replace a stored value when a merge contains the same key."""
        ctx = FlextContext()
        ctx.set("existing", "x")

        ctx.merge({"existing": "y"})

        tm.ok(ctx.get("existing"), eq="y")

    def test_clone_is_independent_snapshot_of_original(self) -> None:
        """Preserve cloned values when the original scope is cleared."""
        ctx = FlextContext()
        ctx.set("k", "v")
        cloned = ctx.clone()
        ctx.clear()
        tm.fail(ctx.get("k"))
        tm.ok(cloned.get("k"), eq="v")

    def test_mutating_clone_does_not_affect_original(self) -> None:
        """Keep original values independent from clone mutations."""
        ctx = FlextContext()
        ctx.set("k", "v")
        cloned = ctx.clone()
        cloned.set("k", "changed")
        tm.ok(ctx.get("k"), eq="v")
        tm.ok(cloned.get("k"), eq="changed")

    def test_export_as_dict_returns_stored_keys(self) -> None:
        """Export stored keys as a dictionary when requested."""
        ctx = FlextContext()
        ctx.set("a", 1)
        ctx.set("b", "two")
        exported = ctx.export(as_dict=True)
        tm.that(exported, is_=dict, eq={"a": 1, "b": "two"})

    def test_export_without_dict_returns_context_itself(self) -> None:
        """Return the same context when dictionary export is disabled."""
        ctx = FlextContext()
        ctx.set("a", 1)
        tm.that(ctx.export(as_dict=False) is ctx, eq=True)

    def test_create_with_initial_values_seeds_the_context_scope(self) -> None:
        """Seed the scope from initial values passed to the factory."""
        ctx = FlextContext.create(operation_id="op-1", user_id="user-1")

        tm.ok(ctx.get("operation_id"), eq="op-1")
        tm.ok(ctx.get("user_id"), eq="user-1")

    # --- metadata --------------------------------------------------------

    def test_apply_metadata_then_resolve_metadata_returns_stored_value(self) -> None:
        """Resolve the value previously applied to context metadata."""
        ctx = FlextContext()

        ctx.apply_metadata("source", "api")

        resolved = ctx.resolve_metadata("source")
        tm.ok(resolved, eq="api")

    def test_resolve_metadata_missing_key_returns_failure(self) -> None:
        """Report the missing key when metadata resolution fails."""
        result = FlextContext().resolve_metadata("absent")
        tm.fail(result, has="absent")

    # --- correlation-id contextvar facade --------------------------------

    def test_apply_and_resolve_correlation_id_roundtrip(self) -> None:
        """Round-trip an explicitly applied correlation identifier."""
        FlextContext.apply_correlation_id("test-corr-123")
        tm.that(FlextContext.resolve_correlation_id(), eq="test-corr-123")

    def test_clear_context_drops_correlation_id(self) -> None:
        """Drop the active correlation identifier when clearing context."""
        FlextContext.apply_correlation_id("test-corr-123")
        FlextContext.clear_context()
        tm.that(FlextContext.resolve_correlation_id(), none=True)

    def test_ensure_correlation_id_generates_when_absent(self) -> None:
        """Generate and apply a correlation identifier when none exists."""
        cid = FlextContext.ensure_correlation_id()
        tm.that(cid, is_=str, empty=False)
        tm.that(FlextContext.resolve_correlation_id(), eq=cid)

    def test_ensure_correlation_id_preserves_existing_value(self) -> None:
        """Preserve the active correlation identifier when ensuring it."""
        FlextContext.apply_correlation_id("already-here")
        tm.that(FlextContext.ensure_correlation_id(), eq="already-here")

    def test_new_correlation_scopes_and_restores_previous_id(self) -> None:
        """Scope a correlation identifier and restore the previous value."""
        FlextContext.apply_correlation_id("outer")
        with FlextContext.new_correlation("inner") as active:
            tm.that(active, eq="inner")
            tm.that(FlextContext.resolve_correlation_id(), eq="inner")
        tm.that(FlextContext.resolve_correlation_id(), eq="outer")

    def test_new_correlation_generates_id_when_none_supplied(self) -> None:
        """Generate a scoped identifier when none is supplied."""
        with FlextContext.new_correlation() as active:
            tm.that(active, is_=str, empty=False)
            tm.that(FlextContext.resolve_correlation_id(), eq=active)

    # --- operation-name contextvar facade --------------------------------

    def test_apply_and_resolve_operation_name_roundtrip(self) -> None:
        """Round-trip an explicitly applied operation name."""
        FlextContext.apply_operation_name("sync-users")
        tm.that(FlextContext.resolve_operation_name(), eq="sync-users")

    def test_resolve_operation_name_is_none_when_unset(self) -> None:
        """Return no operation name when the context is unset."""
        tm.that(FlextContext.resolve_operation_name(), none=True)

    def test_timed_operation_yields_metadata_with_operation_name(self) -> None:
        """Scope the operation name while yielding timing metadata."""
        with FlextContext.timed_operation("etl") as meta:
            tm.that(FlextContext.resolve_operation_name(), eq="etl")
            tm.that(meta.root, empty=False)
        tm.that(FlextContext.resolve_operation_name(), none=True)

    # --- service + full-context export -----------------------------------

    def test_service_context_scopes_service_name(self) -> None:
        """Scope service identity and remove it after context exit."""
        with FlextContext.service_context("billing", version="1.2.0"):
            exported = FlextContext.export_full_context()
            tm.that(exported["service_name"], eq="billing")
            tm.that(exported["service_version"], eq="1.2.0")
        tm.that("service_name" in FlextContext.export_full_context(), eq=False)

    def test_export_full_context_includes_active_correlation_id(self) -> None:
        """Include the active correlation identifier in full export."""
        FlextContext.apply_correlation_id("corr-xyz")
        exported = FlextContext.export_full_context()
        tm.that(exported["correlation_id"], eq="corr-xyz")

    def test_export_full_context_is_empty_after_clear(self) -> None:
        """Export an empty context after clearing all active state."""
        FlextContext.apply_correlation_id("corr-xyz")
        FlextContext.clear_context()
        tm.that(FlextContext.export_full_context(), empty=True)
