"""Behavior contract for the public operational context facade.

Tests exercise the PUBLIC surface only: instance scope store (set/get/has/keys/
values/items/remove/clear/merge/clone/export), metadata I/O, and the class-level
contextvar facade (correlation id / operation name / scoped context managers /
full-context export). No private attributes, no internal collaborators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from contextvars import copy_context
from typing import override

import pytest
import structlog.contextvars

from flext_core import FlextContext, c, h, m, p, r, t, u
from tests import tm


class TestsFlextCoreContext:
    """Contract: scope store + metadata + process-global contextvar operations."""

    pytestmark = pytest.mark.usefixtures("_isolate_process_context")

    class CorrelationHandler(h[str, str]):
        """Handler returning the identifier selected by public call scoping."""

        @override
        def handle(self, message: str) -> p.Result[str]:
            return r[str].ok(FlextContext.ensure_correlation_id())

    @staticmethod
    @pytest.fixture
    def _isolate_process_context() -> Iterator[None]:
        """Keep process-global contextvars from leaking across tests."""
        FlextContext.clear_context()
        yield
        FlextContext.clear_context()

    # --- scope store -----------------------------------------------------

    @staticmethod
    def test_set_then_get_returns_success_with_original_value() -> None:
        """Return the original value after storing it in the scope."""
        ctx = FlextContext()
        tm.ok(ctx.set("key", "value"))
        got = ctx.get("key")
        tm.ok(got, eq="value")

    @staticmethod
    def test_get_on_missing_key_returns_failure_with_message() -> None:
        """Report the missing key when scope lookup fails."""
        result = FlextContext().get("missing")
        tm.fail(result, has="missing")

    @staticmethod
    def test_has_reflects_presence_of_key() -> None:
        """Reflect key absence and presence through the public query."""
        ctx = FlextContext()
        tm.that(ctx.has("k"), eq=False)
        ctx.set("k", "v")
        tm.that(ctx.has("k"), eq=True)

    @staticmethod
    def test_remove_key_makes_subsequent_get_fail() -> None:
        """Make subsequent lookup fail after removing a stored key."""
        ctx = FlextContext()
        ctx.set("k", "v")
        ctx.remove("k")
        tm.fail(ctx.get("k"))

    @staticmethod
    def test_remove_absent_key_is_noop() -> None:
        """Leave the scope unchanged when removing an absent key."""
        ctx = FlextContext()
        ctx.remove("never-added")
        tm.that(ctx.has("never-added"), eq=False)

    @staticmethod
    def test_clear_removes_every_stored_key() -> None:
        """Remove every stored key when clearing the scope."""
        ctx = FlextContext()
        ctx.set("a", "1")
        ctx.set("b", "2")
        ctx.clear()
        tm.fail(ctx.get("a"))
        tm.fail(ctx.get("b"))

    @staticmethod
    def test_keys_values_items_reflect_stored_pairs() -> None:
        """Expose keys, values, and items for the stored pairs."""
        ctx = FlextContext()
        ctx.set("a", 1)
        ctx.set("b", "two")

        tm.that(set(ctx.keys()), eq={"a", "b"})
        tm.that(set(ctx.values()), eq={1, "two"})
        tm.that(dict(ctx.items()), eq={"a": 1, "b": "two"})

    # --- merge / clone / export -----------------------------------------

    @staticmethod
    def test_merge_returns_same_context_and_adds_new_keys() -> None:
        """Mutate and return the receiving context when merging a mapping."""
        ctx = FlextContext()
        ctx.set("existing", "x")
        merged = ctx.merge({"added": "y"})
        tm.that(merged is ctx, eq=True)
        tm.ok(ctx.get("added"), eq="y")
        tm.ok(ctx.get("existing"), eq="x")

    @staticmethod
    def test_merge_accepts_another_context_through_protocol_surface() -> None:
        """Merge another context through the public protocol surface."""
        ctx = FlextContext()
        ctx.set("existing", "x")
        other = FlextContext()
        other.set("added", "y")

        merged = ctx.merge(other)

        tm.that(merged is ctx, eq=True)
        tm.ok(ctx.get("existing"), eq="x")
        tm.ok(ctx.get("added"), eq="y")

    @staticmethod
    def test_merge_overwrites_existing_key_with_new_value() -> None:
        """Replace a stored value when a merge contains the same key."""
        ctx = FlextContext()
        ctx.set("existing", "x")

        ctx.merge({"existing": "y"})

        tm.ok(ctx.get("existing"), eq="y")

    @staticmethod
    def test_clone_is_independent_snapshot_of_original() -> None:
        """Preserve cloned values when the original scope is cleared."""
        ctx = FlextContext()
        ctx.set("k", "v")
        cloned = ctx.clone()
        ctx.clear()
        tm.fail(ctx.get("k"))
        tm.ok(cloned.get("k"), eq="v")

    @staticmethod
    def test_mutating_clone_does_not_affect_original() -> None:
        """Keep original values independent from clone mutations."""
        ctx = FlextContext()
        ctx.set("k", "v")
        cloned = ctx.clone()
        cloned.set("k", "changed")
        tm.ok(ctx.get("k"), eq="v")
        tm.ok(cloned.get("k"), eq="changed")

    @staticmethod
    def test_export_as_dict_returns_stored_keys() -> None:
        """Export stored keys as a dictionary when requested."""
        ctx = FlextContext()
        ctx.set("a", 1)
        ctx.set("b", "two")
        exported = ctx.export(as_dict=True)
        tm.that(exported, is_=dict, eq={"a": 1, "b": "two"})

    @staticmethod
    def test_export_without_dict_returns_context_itself() -> None:
        """Return the same context when dictionary export is disabled."""
        ctx = FlextContext()
        ctx.set("a", 1)
        tm.that(ctx.export(as_dict=False) is ctx, eq=True)

    @staticmethod
    def test_create_with_initial_values_seeds_the_context_scope() -> None:
        """Seed the scope from initial values passed to the factory."""
        ctx = FlextContext.create(
            operation_id="op-1",
            user_id="user-1",
        )

        tm.ok(ctx.get("operation_id"), eq="op-1")
        tm.ok(ctx.get("user_id"), eq="user-1")

    # --- metadata --------------------------------------------------------

    @staticmethod
    def test_apply_metadata_then_resolve_metadata_returns_stored_value() -> None:
        """Resolve the value previously applied to context metadata."""
        ctx = FlextContext()

        ctx.apply_metadata("source", "api")

        resolved = ctx.resolve_metadata("source")
        tm.ok(resolved, eq="api")

    @staticmethod
    def test_resolve_metadata_missing_key_returns_failure() -> None:
        """Report the missing key when metadata resolution fails."""
        result = FlextContext().resolve_metadata(
            "absent",
        )
        tm.fail(result, has="absent")

    # --- correlation-id contextvar facade --------------------------------

    @staticmethod
    def test_apply_and_resolve_correlation_id_roundtrip() -> None:
        """Round-trip an explicitly applied correlation identifier."""
        FlextContext.apply_correlation_id(
            "test-corr-123",
        )
        tm.that(
            FlextContext.resolve_correlation_id(),
            eq="test-corr-123",
        )

    @staticmethod
    def test_clear_context_drops_correlation_id() -> None:
        """Drop the active correlation identifier when clearing context."""
        FlextContext.apply_correlation_id(
            "test-corr-123",
        )
        FlextContext.clear_context()
        tm.that(
            FlextContext.resolve_correlation_id(),
            none=True,
        )

    @staticmethod
    def test_ensure_correlation_id_generates_when_absent() -> None:
        """Generate and apply a correlation identifier when none exists."""
        cid = FlextContext.ensure_correlation_id()
        tm.that(cid, is_=str, empty=False)
        tm.that(
            FlextContext.resolve_correlation_id(),
            eq=cid,
        )

    @staticmethod
    def test_ensure_correlation_id_preserves_existing_value() -> None:
        """Preserve the active correlation identifier when ensuring it."""
        FlextContext.apply_correlation_id(
            "already-here",
        )
        tm.that(
            FlextContext.ensure_correlation_id(),
            eq="already-here",
        )

    @staticmethod
    def test_new_correlation_scopes_and_restores_previous_id() -> None:
        """Scope a correlation identifier and restore the previous value."""
        FlextContext.apply_correlation_id("outer")
        with FlextContext.new_correlation(
            "inner",
        ) as active:
            tm.that(active, eq="inner")
            tm.that(
                FlextContext.resolve_correlation_id(),
                eq="inner",
            )
        tm.that(
            FlextContext.resolve_correlation_id(),
            eq="outer",
        )

    @staticmethod
    def test_new_correlation_generates_id_when_none_supplied() -> None:
        """Generate a scoped identifier when none is supplied."""
        with FlextContext.new_correlation() as active:
            tm.that(active, is_=str, empty=False)
            tm.that(
                FlextContext.resolve_correlation_id(),
                eq=active,
            )

    # --- operation-name contextvar facade --------------------------------

    @staticmethod
    def test_apply_and_resolve_operation_name_roundtrip() -> None:
        """Round-trip an explicitly applied operation name."""
        FlextContext.apply_operation_name("sync-users")
        tm.that(
            FlextContext.resolve_operation_name(),
            eq="sync-users",
        )

    @staticmethod
    def test_resolve_operation_name_is_none_when_unset() -> None:
        """Return no operation name when the context is unset."""
        tm.that(
            FlextContext.resolve_operation_name(),
            none=True,
        )

    @staticmethod
    def test_timed_operation_yields_metadata_with_operation_name() -> None:
        """Scope the operation name while yielding timing metadata."""
        with FlextContext.timed_operation(
            "etl",
        ) as meta:
            tm.that(
                FlextContext.resolve_operation_name(),
                eq="etl",
            )
            tm.that(meta.root, empty=False)
        tm.that(
            FlextContext.resolve_operation_name(),
            none=True,
        )

    # --- service + full-context export -----------------------------------

    @staticmethod
    def test_service_context_scopes_service_name() -> None:
        """Scope service identity and remove it after context exit."""
        with FlextContext.service_context(
            "billing",
            version="1.2.0",
        ):
            exported = FlextContext.export_full_context()
            tm.that(exported["service_name"], eq="billing")
            tm.that(exported["service_version"], eq="1.2.0")
        tm.that(
            "service_name" in FlextContext.export_full_context(),
            eq=False,
        )

    @staticmethod
    def test_export_full_context_includes_active_correlation_id() -> None:
        """Include the active correlation identifier in full export."""
        FlextContext.apply_correlation_id("corr-xyz")
        exported = FlextContext.export_full_context()
        tm.that(exported["correlation_id"], eq="corr-xyz")

    @staticmethod
    def test_export_full_context_is_empty_after_clear() -> None:
        """Export an empty context after clearing all active state."""
        FlextContext.apply_correlation_id("corr-xyz")
        FlextContext.clear_context()
        tm.that(
            FlextContext.export_full_context(),
            empty=True,
        )

    @staticmethod
    def test_public_factory_retains_context_identity_and_metadata_isolation() -> None:
        """Clones remain public contexts with independent metadata updates."""
        context = FlextContext.create(seed="original")
        tm.that(isinstance(context, FlextContext), eq=True)
        context.apply_metadata("source", "original")
        cloned = context.clone()
        cloned.apply_metadata("source", "cloned")
        tm.that(isinstance(cloned, FlextContext), eq=True)
        tm.ok(context.resolve_metadata("source"), eq="original")
        tm.ok(cloned.resolve_metadata("source"), eq="cloned")

    @staticmethod
    def test_clone_deep_copies_nested_scope_values() -> None:
        """Mutation of the original nested payload does not affect its clone."""
        payload: t.MutableMappingKV[str, t.JsonPayload] = {"state": "original"}
        context = FlextContext()
        context.set("nested", payload)
        cloned = context.clone()
        payload["state"] = "changed"
        tm.ok(context.get("nested"), eq={"state": "changed"})
        tm.ok(cloned.get("nested"), eq={"state": "original"})

    @staticmethod
    def test_metadata_snapshots_detach_nested_caller_values() -> None:
        """Caller mutations cannot change metadata in either public context."""
        roles: list[t.JsonValue] = ["reader"]
        payload: t.JsonDict = {"roles": roles}
        context = FlextContext()
        context.apply_metadata("actor", payload)
        cloned = context.clone()

        roles.append("admin")
        payload["source"] = "changed"

        tm.ok(context.resolve_metadata("actor"), eq={"roles": ["reader"]})
        tm.ok(cloned.resolve_metadata("actor"), eq={"roles": ["reader"]})

    @staticmethod
    def test_context_and_utility_proxies_share_structlog_storage() -> None:
        """Public context and utility writes are visible to structlog and each other."""
        FlextContext.apply_correlation_id("public-context")
        tm.that(
            structlog.contextvars.get_contextvars()[c.ContextKey.CORRELATION_ID],
            eq="public-context",
        )
        token = u.CORRELATION_ID.set("utility-context")
        tm.that(FlextContext.resolve_correlation_id(), eq="utility-context")
        u.CORRELATION_ID.reset(token)
        tm.that(FlextContext.resolve_correlation_id(), eq="public-context")
        _ = structlog.contextvars.bind_contextvars(**{
            c.ContextKey.CORRELATION_ID: "structlog-context"
        })
        tm.that(FlextContext.resolve_correlation_id(), eq="structlog-context")

    @staticmethod
    def test_proxy_factories_restore_shared_values_and_defaults() -> None:
        """All proxy factories retain structlog-backed reset and default semantics."""
        string = u.create_str_proxy("context_string_probe", default="default")
        timestamp = u.create_datetime_proxy("context_timestamp_probe")
        mapping = u.create_dict_proxy("context_mapping_probe")
        now = u.generate_datetime_utc()
        try:
            tm.that(string.get(), eq="default")
            string_token = string.set("changed")
            timestamp_token = timestamp.set(now)
            mapping_token = mapping.set({"source": "context"})
            tm.that(u.create_str_proxy("context_string_probe").get(), eq="changed")
            tm.that(timestamp.get(), eq=now)
            tm.that(mapping.get(), eq={"source": "context"})
            string.reset(string_token)
            timestamp.reset(timestamp_token)
            mapping.reset(mapping_token)
            tm.that(string.get(), eq="default")
            tm.that(timestamp.get(), none=True)
            tm.that(mapping.get(), none=True)
        finally:
            structlog.contextvars.unbind_contextvars(
                "context_string_probe",
                "context_timestamp_probe",
                "context_mapping_probe",
            )

    @staticmethod
    def test_nested_correlation_restores_parent_after_exception() -> None:
        """A failing nested scope restores both shared correlation identifiers."""
        FlextContext.apply_correlation_id("outer")
        parent_token = u.PARENT_CORRELATION_ID.set("ancestor")
        try:
            with (
                pytest.raises(ValueError, match="nested failure"),
                FlextContext.new_correlation("inner"),
            ):
                tm.that(u.PARENT_CORRELATION_ID.get(), eq="outer")
                raise ValueError("nested failure")
            tm.that(FlextContext.resolve_correlation_id(), eq="outer")
            tm.that(u.PARENT_CORRELATION_ID.get(), eq="ancestor")
        finally:
            u.PARENT_CORRELATION_ID.reset(parent_token)

    @staticmethod
    def test_contextvars_are_isolated_in_a_copied_execution_context() -> None:
        """A copied execution context cannot overwrite the calling context's state."""
        FlextContext.apply_correlation_id("outer")
        isolated = copy_context()
        isolated.run(FlextContext.apply_correlation_id, "isolated")
        tm.that(isolated.run(FlextContext.resolve_correlation_id), eq="isolated")
        tm.that(FlextContext.resolve_correlation_id(), eq="outer")

    @staticmethod
    def test_timing_and_service_scopes_restore_values_on_failure() -> None:
        """Timing still finalizes metrics and restores outer shared values on errors."""
        FlextContext.apply_operation_name("outer-operation")
        metrics: p.MutableRootDict[t.JsonPayload] = m.ConfigMap(root={})
        with FlextContext.service_context("outer-service", "outer-version"):
            with (
                pytest.raises(ValueError, match="timed failure"),
                FlextContext.service_context("inner-service", "inner-version"),
                FlextContext.timed_operation("inner-operation") as metrics,
            ):
                tm.that(FlextContext.resolve_operation_name(), eq="inner-operation")
                raise ValueError("timed failure")
            tm.that(FlextContext.resolve_operation_name(), eq="outer-operation")
            tm.that(
                FlextContext.export_full_context()[c.ContextKey.SERVICE_NAME],
                eq="outer-service",
            )
            tm.that(
                FlextContext.export_full_context()[c.ContextKey.SERVICE_VERSION],
                eq="outer-version",
            )
            tm.that(c.MetadataKey.END_TIME in metrics.root, eq=True)
            tm.that(c.MetadataKey.DURATION_SECONDS in metrics.root, eq=True)

    @staticmethod
    def test_handler_call_scopes_correlation_through_the_public_context() -> None:
        """Handler invocation uses the elected context without model class bindings."""
        FlextContext.apply_correlation_id("outer-handler")
        result = TestsFlextCoreContext.CorrelationHandler()("message")
        active = result.unwrap()
        tm.that(bool(active), eq=True)
        tm.that(active != "outer-handler", eq=True)
        tm.that(FlextContext.resolve_correlation_id(), eq="outer-handler")

    @staticmethod
    def test_runtime_scope_resolution_copies_state_and_preserves_hooks() -> None:
        """Extending scopes preserves metadata, hooks and independent variable storage."""

        def on_event(value: t.Scalar) -> t.JsonPayload:
            return value

        metadata = m.Metadata(attributes={"source": "runtime"})
        state = u.create_context_runtime(metadata)
        state = m.ContextRuntimeState.model_validate({
            "metadata": state.metadata,
            "scope_vars": state.scope_vars,
            "hooks": {"event": (on_event,)},
        })
        updated, variable = u.resolve_context_scope_var(state, "custom")
        tm.that("custom" in state.scope_vars, eq=False)
        tm.that(updated.scope_vars["custom"] is variable, eq=True)
        tm.that(updated.metadata is state.metadata, eq=True)
        tm.that(updated.hooks["event"][0]("payload"), eq="payload")
        repeated, existing = u.resolve_context_scope_var(updated, "custom")
        tm.that(repeated is updated, eq=True)
        tm.that(existing is variable, eq=True)
        other = u.create_context_runtime(metadata)
        tm.that(
            other.scope_vars[c.ContextScope.GLOBAL]
            is state.scope_vars[c.ContextScope.GLOBAL],
            eq=False,
        )
        values = m.ConfigMap(root={"source": "first"})
        token = variable.set(values)
        try:
            tm.that(updated.scope_vars["custom"].get() is values, eq=True)
        finally:
            variable.reset(token)

    @staticmethod
    def test_runtime_operation_updates_preserve_original_counters() -> None:
        """Counter updates are immutable and derive initial values from their owner."""
        state = u.create_context_runtime()
        updated = u.update_context_operation(state, "set")
        tm.that(updated.statistics.sets, eq=state.statistics.sets + 1)
        tm.that(updated.statistics.gets, eq=state.statistics.gets)
        tm.that(updated.metadata is state.metadata, eq=True)
        tm.that(u.update_context_operation(state, "unknown") is state, eq=True)
        tracked: p.ContextRuntimeState = m.ContextRuntimeState(
            statistics=m.ContextStatistics(operations={"custom": 4})
        )
        incremented = u.update_context_operation(tracked, "custom")
        tm.that(incremented.statistics.operations["custom"], eq=5)
        tm.that(tracked.statistics.operations["custom"], eq=4)
