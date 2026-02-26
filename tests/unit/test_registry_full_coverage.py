"""Coverage tests for FlextRegistry compatibility paths."""

from __future__ import annotations

from typing import cast

import pytest
from flext_core import FlextHandlers, FlextRegistry, FlextResult, c, h, m, p, r, t
from flext_core.typings import JsonValue


class _Handler(FlextHandlers[JsonValue, JsonValue]):
    def handle(self, message: JsonValue) -> FlextResult[JsonValue]:
        return r[JsonValue].ok(message)

    def __call__(self, message: JsonValue) -> FlextResult[JsonValue]:
        return self.handle(message)


def _success_details(reg_id: str) -> m.Handler.RegistrationDetails:
    return m.Handler.RegistrationDetails(
        registration_id=reg_id,
        handler_mode=c.Cqrs.HandlerType.COMMAND,
        timestamp="",
        status=c.Cqrs.CommonStatus.RUNNING,
    )


def _as_registry_handler(handler: _Handler) -> t.HandlerCallable:
    return cast("t.HandlerCallable", handler)


def test_summary_properties_and_subclass_storage_reset() -> None:
    detail = _success_details("a")
    summary = FlextRegistry.Summary(registered=[detail], errors=["x"])
    assert len(summary.registered) == 1
    assert len(summary.errors) == 1

    class _ChildRegistry(FlextRegistry):
        pass

    assert _ChildRegistry._class_plugin_storage == {}
    assert _ChildRegistry._class_registered_keys == set()


def test_execute_and_register_handler_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FlextRegistry()
    setattr(registry, "_dispatcher", None)
    execute_result = registry.execute()
    assert execute_result.is_failure

    class _FailDispatcher:
        def register_handler(self, *_args: object):
            return r[m.Handler.RegistrationResult].fail("dispatcher-fail")

    setattr(
        registry,
        "_dispatcher",
        cast("p.CommandBus", cast("object", _FailDispatcher())),
    )
    reg_result = registry.register_handler(_as_registry_handler(_Handler()))
    assert reg_result.is_failure
    assert reg_result.error == "dispatcher-fail"

    class _OkDispatcher:
        def register_handler(self, *_args: object):
            return r[m.Handler.RegistrationResult].ok(
                m.Handler.RegistrationResult(
                    handler_name="h",
                    status="active",
                    mode="command",
                )
            )

    setattr(
        registry,
        "_dispatcher",
        cast("p.CommandBus", cast("object", _OkDispatcher())),
    )
    monkeypatch.setattr(
        FlextRegistry, "_create_registration_details", lambda *_args: None
    )
    fallback = registry.register_handler(_as_registry_handler(_Handler()))
    assert fallback.is_success
    assert fallback.value.registration_id != ""


def test_create_auto_discover_and_mode_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered_handler = _Handler()

    def fake_scan(_module: object):
        cfg = m.Handler.DecoratorConfig(command=str, middleware=[])
        return [("x", discovered_handler.handle, cfg)]

    monkeypatch.setattr(h.Discovery, "scan_module", staticmethod(fake_scan))
    monkeypatch.setattr(
        FlextRegistry,
        "register_handler",
        lambda self, handler: r[m.Handler.RegistrationDetails].ok(
            _success_details(getattr(handler, "__name__", "h"))
        ),
    )
    created = FlextRegistry.create(auto_discover_handlers=True)
    assert isinstance(created, FlextRegistry)

    registry = FlextRegistry()
    query_details = registry._create_registration_details(
        m.Handler.RegistrationResult(handler_name="q", status="active", mode="query"),
        "k1",
    )
    event_details = registry._create_registration_details(
        m.Handler.RegistrationResult(handler_name="e", status="active", mode="event"),
        "k2",
    )
    assert query_details.handler_mode == c.Cqrs.HandlerType.QUERY
    assert event_details.handler_mode == c.Cqrs.HandlerType.EVENT


def test_summary_error_paths_and_bindings_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = FlextRegistry.Summary()
    msg = FlextRegistry._add_registration_error("k", "", summary)
    assert msg == ""
    assert summary.errors[0].startswith("Failed to register handler")

    registry = FlextRegistry()
    finalize = registry._finalize_summary(summary)
    assert finalize.is_failure

    monkeypatch.setattr(
        FlextRegistry,
        "register_handler",
        lambda self, _handler: r[m.Handler.RegistrationDetails].fail("x"),
    )
    batch = registry.register_handlers([_as_registry_handler(_Handler())])
    assert batch.is_failure

    class _FailBindingDispatcher:
        def register_handler(self, *_args: object):
            return r[m.Handler.RegistrationResult].fail("bind-fail")

    class _RaiseBindingDispatcher:
        def register_handler(self, *_args: object):
            msg = "bind-ex"
            raise RuntimeError(msg)

    setattr(
        registry,
        "_dispatcher",
        cast("p.CommandBus", cast("object", _FailBindingDispatcher())),
    )
    failed = registry.register_bindings({str: _as_registry_handler(_Handler())})
    assert failed.is_failure

    setattr(
        registry,
        "_dispatcher",
        cast("p.CommandBus", cast("object", _RaiseBindingDispatcher())),
    )
    raised = registry.register_bindings({str: _as_registry_handler(_Handler())})
    assert raised.is_failure


def test_get_plugin_and_register_metadata_and_list_items_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FlextRegistry()
    registry._registered_keys.add("cat::name")
    monkeypatch.setattr(
        registry.container,
        "get",
        lambda _key: r[JsonValue].fail("missing"),
    )
    missing = registry.get_plugin("cat", "name")
    assert missing.is_failure

    metadata_result = registry.register(
        "svc",
        "service",
        metadata=m.Metadata(attributes={"k": "v"}),
    )
    assert metadata_result.is_success

    monkeypatch.setattr(
        registry.container,
        "with_service",
        lambda _name, _service: (_ for _ in ()).throw(ValueError("nope")),
    )
    reg_fail = registry.register("svc2", "service")
    assert reg_fail.is_failure
