"""Behavior-focused coverage tests for FlextContext public APIs."""

from __future__ import annotations

from flext_tests import tm
from tests import FlextContainer, FlextContext, c, m, p, t


def test_create_overloads_and_auto_correlation() -> None:
    ctx = FlextContext.create(user_id="u1", metadata=t.ConfigMap(root={"x": 1}))
    tm.that(ctx, is_=p.Context)
    tm.that(ctx.get(c.ContextKey.USER_ID).value, eq="u1")

    generated = FlextContext.create(initial_data=t.ConfigMap(root={}))
    tm.ok(generated.get(c.ContextKey.OPERATION_ID))

    explicit = FlextContext.create(operation_id="op-explicit")
    tm.that(explicit.get(c.ContextKey.OPERATION_ID).value, eq="op-explicit")


def test_public_set_get_merge_clone_and_clear_flow() -> None:
    ctx = FlextContext()

    tm.ok(ctx.set("key1", "value1"))
    tm.ok(ctx.set(t.ConfigMap(root={"key2": "value2", "key3": 3})))
    tm.that(ctx.get("key1").value, eq="value1")
    tm.that(ctx.get("key2").value, eq="value2")

    merged = ctx.merge({"key4": "value4"})
    tm.that(merged is ctx, eq=True)
    tm.that(ctx.get("key4").value, eq="value4")

    cloned = ctx.clone()
    tm.that(cloned, is_=p.Context)
    ctx.clear()
    tm.fail(ctx.get("key1"))
    tm.that(cloned.get("key1").value, eq="value1")


def test_public_export_includes_metadata_and_statistics() -> None:
    ctx = FlextContext()

    tm.ok(ctx.set("key", "value"))
    _ = ctx.get("key")
    ctx.apply_metadata("meta_key", "meta_value")

    exported_dict = ctx.export(
        include_statistics=True,
        include_metadata=True,
        as_dict=True,
    )
    tm.that(exported_dict, is_=dict)
    if isinstance(exported_dict, dict):
        tm.that(exported_dict, has="statistics")
        tm.that(exported_dict, has="metadata")

    exported_model = ctx.export(
        include_statistics=True,
        include_metadata=True,
        as_dict=False,
    )
    tm.that(exported_model, is_=m.ContextExport)
    if isinstance(exported_model, m.ContextExport):
        tm.that(exported_model.has_statistics, eq=True)
        tm.that(exported_model.total_data_items, gte=1)


def test_public_validation_and_missing_value_paths() -> None:
    ctx = FlextContext()

    tm.fail(ctx.get("missing"))
    tm.fail(ctx.resolve_metadata("missing"))
    tm.fail(ctx.set("", "value"))
    tm.ok(ctx.validate_context())


def test_service_namespace_register_fetch_and_scope() -> None:
    container = FlextContainer.shared(context=FlextContext())
    container.clear()
    FlextContext.configure_container(container)

    tm.ok(FlextContext.Service.register_service("svc", "value"))
    fetched = FlextContext.Service.fetch_service("svc")
    tm.ok(fetched)
    assert isinstance(fetched.value, str)
    tm.that(fetched.value, eq="value")
    tm.fail(FlextContext.Service.fetch_service("missing"))

    with FlextContext.Service.service_context("svc-name", version="1.0.0"):
        exported = FlextContext.Serialization.export_full_context()
        tm.that(exported[c.ContextKey.SERVICE_NAME], eq="svc-name")
        tm.that(exported[c.ContextKey.SERVICE_VERSION], eq="1.0.0")


def test_correlation_and_utility_public_apis() -> None:
    FlextContext.Utilities.clear_context()
    correlation_id = FlextContext.Utilities.ensure_correlation_id()
    tm.that(FlextContext.Correlation.resolve_correlation_id(), eq=correlation_id)

    FlextContext.Correlation.apply_correlation_id(None)
    tm.that(FlextContext.Correlation.resolve_correlation_id(), none=True)


def test_performance_and_serialization_public_apis() -> None:
    FlextContext.Utilities.clear_context()

    with FlextContext.Performance.timed_operation("demo-operation") as metadata:
        tm.that(metadata[c.MetadataKey.START_TIME], none=False)
        tm.that(metadata[c.ContextKey.OPERATION_NAME], eq="demo-operation")

    tm.that(metadata[c.MetadataKey.END_TIME], none=False)
    tm.that(metadata[c.MetadataKey.DURATION_SECONDS], gte=0.0)
