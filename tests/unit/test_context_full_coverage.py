"""Expanded coverage tests for active FlextContext behaviors."""

from __future__ import annotations

import pytest
from flext_tests import t, tm
from pydantic import BaseModel

from flext_core import FlextContainer, FlextContext, c, m, r


class _ContainerStub:
    def __init__(self) -> None:
        self.services: dict[str, t.Container | BaseModel] = {}

    def get(self, name: str) -> r[t.Container | BaseModel]:
        if name in self.services:
            return r[t.Container | BaseModel].ok(self.services[name])
        return r[t.Container | BaseModel].fail("missing")

    def with_service(
        self, name: str, service: t.Container | BaseModel
    ) -> _ContainerStub:
        if name == "bad":
            msg = "bad service"
            raise ValueError(msg)
        self.services[name] = service
        return self

    def register(self, name: str, service: t.Container | BaseModel) -> _ContainerStub:
        return self.with_service(name, service)


def test_narrow_contextvar_invalid_inputs() -> None:
    tm.that(FlextContext._narrow_contextvar_to_configuration_dict(None), eq={})
    data = FlextContext._narrow_contextvar_to_configuration_dict(
        m.ConfigMap(root={"a": "v"})
    )
    tm.that("a" in data, eq=True)


def test_narrow_contextvar_exception_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FlextContext()

    def _raise_type_error(_value: t.Tests.object) -> None:
        msg = "bad"
        raise TypeError(msg)

    monkeypatch.setattr(
        "flext_core.context.FlextRuntime.normalize_to_container",
        _raise_type_error,
    )
    tm.that(FlextContext._narrow_contextvar_to_configuration_dict({"x": 1}), eq={})


def test_create_overloads_and_auto_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _generate_id(_key: str) -> str:
        return "corr-1"

    monkeypatch.setattr("flext_core.context.u.generate", _generate_id)
    ctx = FlextContext.create(user_id="u1", metadata=m.ConfigMap(root={"x": 1}))
    tm.that(isinstance(ctx, FlextContext), eq=True)
    tm.that(ctx.get(c.Context.KEY_USER_ID).value, eq="u1")
    ctx2 = FlextContext.create(initial_data=m.ConfigMap(root={}))
    tm.ok(ctx2.get(c.Context.KEY_OPERATION_ID))
    ctx3 = FlextContext.create(operation_id="op-explicit")
    tm.that(ctx3.get(c.Context.KEY_OPERATION_ID).value, eq="op-explicit")


def test_set_set_all_get_validation_and_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FlextContext()
    _ = ctx.set("k", "v")
    tm.that(ctx.get("k").value, eq="v")
    tm.ok(ctx.set(m.ConfigMap(root={})))

    class _BadVar:
        def get(self) -> dict[str, t.Tests.object]:
            return {}

        def set(self, _v: t.Tests.object) -> None:
            msg = "boom"
            raise TypeError(msg)

    def _make_bad_var(_scope: str) -> _BadVar:
        return _BadVar()

    monkeypatch.setattr(ctx, "_get_or_create_scope_var", _make_bad_var)
    tm.fail(ctx.set("x", "y"))
    tm.fail(ctx.set(m.ConfigMap(root={"x": "y"})))
    tm.ok(FlextContext._validate_set_inputs("k", "bad"))


def test_inactive_and_none_value_paths() -> None:
    ctx = FlextContext()
    ctx._active = False
    tm.fail(ctx.set("k", "v"))
    tm.fail(ctx.set(m.ConfigMap(root={"k": "v"})))
    tm.fail(ctx.get("k"))
    tm.that(ctx.has("k") is False, eq=True)
    ctx.remove("k")
    ctx.clear()
    merge_data: dict[str, t.NormalizedValue] = {"k": "v"}
    tm.that(ctx.merge(merge_data) is ctx, eq=True)
    tm.fail(ctx.validate_context())
    tm.that(ctx.keys(), eq=[])
    tm.that(ctx.values(), eq=[])
    tm.that(ctx.items(), eq=[])
    tm.that(ctx._get_all_scopes(), eq={})
    ctx2 = FlextContext()
    ctx2._set_in_contextvar(c.Context.SCOPE_GLOBAL, m.ConfigMap(root={"k": None}))
    tm.fail(ctx2.get("k"))


def test_clear_keys_values_items_and_validate_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FlextContext()
    ctx._statistics.operations = {c.Context.OPERATION_CLEAR: 1}
    ctx.clear()
    ctx._active = False
    tm.that(ctx.keys(), eq=[])
    tm.that(ctx.values(), eq=[])
    tm.that(ctx.items(), eq=[])
    ctx2 = FlextContext()

    class _BadVar:
        def get(self) -> None:
            msg = "bad"
            raise TypeError(msg)

    monkeypatch.setattr(ctx2, "_scope_vars", {"bad": _BadVar()})
    tm.fail(ctx2.validate_input())
    ctx3 = FlextContext()
    ctx3._set_in_contextvar("global", m.ConfigMap(root={"": "x"}))
    tm.fail(ctx3.validate_input())


def test_update_statistics_remove_hook_and_clone_false_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FlextContext()
    ctx._statistics.operations = {c.Context.OPERATION_GET: 1}
    ctx._update_statistics(c.Context.OPERATION_GET)
    tm.that(ctx._statistics.operations[c.Context.OPERATION_GET], eq=2)
    clone_source = FlextContext()
    _ = clone_source.set("a", "b")
    original_set = FlextContext.set

    def _fail_set(
        self: FlextContext,
        key_or_data: str | m.ConfigMap,
        value: t.Container | None = None,
        scope: str = "current",
    ) -> r[bool]:
        _ = self, key_or_data, value, scope
        return r[bool].fail("x")

    monkeypatch.setattr(FlextContext, "set", _fail_set)
    cloned = clone_source.clone()
    monkeypatch.setattr(FlextContext, "set", original_set)
    tm.that(isinstance(cloned, FlextContext), eq=True)


def test_export_paths_with_metadata_and_statistics() -> None:
    ctx = FlextContext()
    _ = ctx.set("k", "v")
    ctx.set_metadata("mk", "mv")
    exported_dict = ctx.export(
        include_statistics=True,
        include_metadata=True,
        as_dict=True,
    )
    tm.that(isinstance(exported_dict, dict), eq=True)
    if isinstance(exported_dict, dict):
        tm.that("statistics" in exported_dict, eq=True)
        tm.that("metadata" in exported_dict, eq=True)
    exported_model = ctx.export(
        include_statistics=True,
        include_metadata=True,
        as_dict=False,
    )
    tm.that(isinstance(exported_model, m.ContextExport), eq=True)


def test_container_and_service_domain_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    FlextContext._container = None
    with pytest.raises(RuntimeError):
        FlextContext.get_container()
    container = FlextContainer()
    container.clear_all()
    FlextContext.set_container(container)
    tm.that(FlextContext.get_container() is container, eq=True)
    container.register("obj", "x")
    result = FlextContext.Service.get_service("obj")
    tm.ok(result)
    tm.that(result.value, eq="x")
    tm.ok(FlextContext.Service.register_service("ok", "value"))
    tm.ok(FlextContext.Service.register_service("bad", "value"))
    tm.fail(FlextContext.Service.get_service("missing"))


def test_create_merges_metadata_dict_branch() -> None:
    ctx = FlextContext.create(metadata=m.ConfigMap(root={"meta_key": "meta_value"}))
    tm.that(ctx.get("meta_key").value, eq="meta_value")
