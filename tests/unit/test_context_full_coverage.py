"""Expanded coverage tests for active FlextContext behaviors."""

from __future__ import annotations

import pytest

from flext_core import FlextContainer, FlextContext, c, m, r


class _ContainerStub:
    def __init__(self) -> None:
        self.services: dict[str, object] = {}

    def get(self, name: str) -> r[object]:
        if name in self.services:
            return r[object].ok(self.services[name])
        return r[object].fail("missing")

    def with_service(self, name: str, service: object) -> _ContainerStub:
        if name == "bad":
            msg = "bad service"
            raise ValueError(msg)
        self.services[name] = service
        return self

    def register(self, name: str, service: object) -> _ContainerStub:
        return self.with_service(name, service)


def test_narrow_contextvar_invalid_inputs() -> None:
    assert FlextContext._narrow_contextvar_to_configuration_dict(None) == {}
    data = FlextContext._narrow_contextvar_to_configuration_dict(
        m.ConfigMap(root={"a": "v"})
    )
    assert data["a"]


def test_protocol_name_and_narrow_contextvar_exception_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FlextContext()
    assert ctx._protocol_name() == "FlextContext"

    def _raise_type_error(_value: object) -> object:
        msg = "bad"
        raise TypeError(msg)

    monkeypatch.setattr(
        "flext_core.context.FlextRuntime.normalize_to_general_value",
        _raise_type_error,
    )
    assert FlextContext._narrow_contextvar_to_configuration_dict({"x": 1}) == {}


def test_create_overloads_and_auto_correlation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _generate_id(_key: str) -> str:
        return "corr-1"

    monkeypatch.setattr("flext_core.context.u.generate", _generate_id)
    ctx = FlextContext.create(user_id="u1", metadata=m.ConfigMap(root={"x": 1}))
    assert isinstance(ctx, FlextContext)
    assert ctx.get(c.Context.KEY_USER_ID).value == "u1"
    ctx2 = FlextContext.create(initial_data=m.ConfigMap(root={}))
    assert ctx2.get(c.Context.KEY_OPERATION_ID).is_success
    ctx3 = FlextContext.create(operation_id="op-explicit")
    assert ctx3.get(c.Context.KEY_OPERATION_ID).value == "op-explicit"


def test_set_set_all_get_validation_and_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FlextContext()
    _ = ctx.set("k", "v")
    assert ctx.get("k").value == "v"
    assert ctx.set(m.ConfigMap(root={})).is_success

    class _BadVar:
        def get(self) -> dict[str, object]:
            return {}

        def set(self, _v: object) -> None:
            msg = "boom"
            raise TypeError(msg)

    def _make_bad_var(_scope: str) -> _BadVar:
        return _BadVar()

    monkeypatch.setattr(ctx, "_get_or_create_scope_var", _make_bad_var)
    assert ctx.set("x", "y").is_failure
    assert ctx.set(m.ConfigMap(root={"x": "y"})).is_failure
    assert FlextContext._validate_set_inputs("k", "bad").is_success


def test_inactive_and_none_value_paths() -> None:
    ctx = FlextContext()
    ctx._active = False
    assert ctx.set("k", "v").is_failure
    assert ctx.set(m.ConfigMap(root={"k": "v"})).is_failure
    assert ctx.get("k").is_failure
    assert ctx.has("k") is False
    ctx.remove("k")
    ctx.clear()
    assert ctx.merge(m.ConfigMap(root={"k": "v"}).root) is ctx
    assert ctx.validate().is_failure
    assert ctx.keys() == []
    assert ctx.values() == []
    assert ctx.items() == []
    assert ctx._get_all_scopes() == {}
    ctx2 = FlextContext()
    ctx2._set_in_contextvar(c.Context.SCOPE_GLOBAL, m.ConfigMap(root={"k": None}))
    assert ctx2.get("k").is_failure


def test_clear_keys_values_items_and_validate_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FlextContext()
    ctx._statistics.operations = {c.Context.OPERATION_CLEAR: 1}
    ctx.clear()
    ctx._active = False
    assert ctx.keys() == []
    assert ctx.values() == []
    assert ctx.items() == []
    ctx2 = FlextContext()

    class _BadVar:
        def get(self) -> None:
            msg = "bad"
            raise TypeError(msg)

    monkeypatch.setattr(ctx2, "_scope_vars", {"bad": _BadVar()})
    assert ctx2.validate().is_failure
    ctx3 = FlextContext()
    ctx3._set_in_contextvar("global", m.ConfigMap(root={"": "x"}))
    assert ctx3.validate().is_failure


def test_update_statistics_remove_hook_and_clone_false_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = FlextContext()
    ctx._statistics.operations = {c.Context.OPERATION_GET: 1}
    ctx._update_statistics(c.Context.OPERATION_GET)
    assert ctx._statistics.operations[c.Context.OPERATION_GET] == 2
    clone_source = FlextContext()
    _ = clone_source.set("a", "b")
    original_set = FlextContext.set

    def _fail_set(
        self: FlextContext,
        key_or_data: str | m.ConfigMap,
        value: object | None = None,
        scope: str = "current",
    ) -> r[bool]:
        _ = self, key_or_data, value, scope
        return r[bool].fail("x")

    monkeypatch.setattr(FlextContext, "set", _fail_set)
    cloned = clone_source.clone()
    monkeypatch.setattr(FlextContext, "set", original_set)
    assert isinstance(cloned, FlextContext)


def test_export_paths_with_metadata_and_statistics() -> None:
    ctx = FlextContext()
    _ = ctx.set("k", "v")
    ctx.set_metadata("mk", "mv")
    exported_dict = ctx.export(
        include_statistics=True,
        include_metadata=True,
        as_dict=True,
    )
    assert isinstance(exported_dict, dict)
    assert "statistics" in exported_dict
    assert "metadata" in exported_dict
    exported_model = ctx.export(
        include_statistics=True,
        include_metadata=True,
        as_dict=False,
    )
    assert isinstance(exported_model, m.ContextExport)


def test_container_and_service_domain_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    FlextContext._container = None
    with pytest.raises(RuntimeError):
        FlextContext.get_container()
    container = FlextContainer()
    container.clear_all()
    FlextContext.set_container(container)
    assert FlextContext.get_container() is container
    container.register("obj", "x")
    result = FlextContext.Service.get_service("obj")
    assert result.is_success
    assert result.value == "x"
    assert FlextContext.Service.register_service("ok", "value").is_success
    assert FlextContext.Service.register_service("bad", "value").is_success
    assert FlextContext.Service.get_service("missing").is_failure


def test_create_merges_metadata_dict_branch() -> None:
    ctx = FlextContext.create(metadata=m.ConfigMap(root={"meta_key": "meta_value"}))
    assert ctx.get("meta_key").value == "meta_value"
