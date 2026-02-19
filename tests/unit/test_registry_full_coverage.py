from __future__ import annotations

import importlib
from types import SimpleNamespace

core = importlib.import_module("flext_core")
FlextRegistry = core.FlextRegistry
c = core.c
h = core.h
m = core.m
r = core.r
t = core.t


class _Handler(h[t.GeneralValueType, t.GeneralValueType]):
    def handle(self, message: t.GeneralValueType) -> r[t.GeneralValueType]:
        return r[t.GeneralValueType].ok(message)


class _BrokenListRegistry(FlextRegistry):
    def __getattribute__(self, name: str):
        if name == "_registered_keys":
            raise TypeError("broken")
        return super().__getattribute__(name)


def _success_details(reg_id: str) -> m.HandlerRegistrationDetails:
    return m.HandlerRegistrationDetails(
        registration_id=reg_id,
        handler_mode=c.Cqrs.HandlerType.COMMAND,
        timestamp="",
        status=c.Cqrs.CommonStatus.RUNNING,
    )


def test_summary_properties_and_subclass_storage_reset() -> None:
    detail = _success_details("a")
    summary = FlextRegistry.Summary(registered=[detail], errors=["x"])
    assert summary.successful_registrations == 1
    assert summary.failed_registrations == 1

    class _ChildRegistry(FlextRegistry):
        pass

    assert _ChildRegistry._class_plugin_storage == {}
    assert _ChildRegistry._class_registered_keys == set()


def test_execute_and_register_handler_failure_paths(monkeypatch) -> None:
    registry = FlextRegistry()
    registry._dispatcher = None
    execute_result = registry.execute()
    assert execute_result.is_failure

    class _FailDispatcher:
        def register_handler(self, *_args: object):
            return r[m.HandlerRegistrationResult].fail("dispatcher-fail")

    registry._dispatcher = _FailDispatcher()
    reg_result = registry.register_handler(_Handler())
    assert reg_result.is_failure
    assert reg_result.error == "dispatcher-fail"

    class _OkDispatcher:
        def register_handler(self, *_args: object):
            return r[m.HandlerRegistrationResult].ok(
                m.HandlerRegistrationResult(
                    handler_name="h",
                    status="active",
                    mode="command",
                )
            )

    registry._dispatcher = _OkDispatcher()
    monkeypatch.setattr(
        FlextRegistry, "_create_registration_details", lambda *_args: None
    )
    fallback = registry.register_handler(_Handler())
    assert fallback.is_success
    assert fallback.value.registration_id != ""


def test_create_auto_discover_and_mode_mapping(monkeypatch) -> None:
    discovered_handler = _Handler()

    def fake_scan(_module: object):
        cfg = m.Handler.DecoratorConfig(command=str)
        return [("x", discovered_handler.handle, cfg)]

    monkeypatch.setattr(h.Discovery, "scan_module", staticmethod(fake_scan))
    monkeypatch.setattr(
        FlextRegistry,
        "register_handler",
        lambda self, handler: r[m.HandlerRegistrationDetails].ok(
            _success_details(getattr(handler, "__name__", "h"))
        ),
    )
    created = FlextRegistry.create(auto_discover_handlers=True)
    assert isinstance(created, FlextRegistry)

    registry = FlextRegistry()
    query_details = registry._create_registration_details(
        m.HandlerRegistrationResult(handler_name="q", status="active", mode="query"),
        "k1",
    )
    event_details = registry._create_registration_details(
        m.HandlerRegistrationResult(handler_name="e", status="active", mode="event"),
        "k2",
    )
    assert query_details.handler_mode == c.Cqrs.HandlerType.QUERY
    assert event_details.handler_mode == c.Cqrs.HandlerType.EVENT


def test_summary_error_paths_and_bindings_failures(monkeypatch) -> None:
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
        lambda self, _handler: r[m.HandlerRegistrationDetails].fail("x"),
    )
    batch = registry.register_handlers([_Handler()])
    assert batch.is_failure

    class _FailBindingDispatcher:
        def register_handler(self, *_args: object):
            return r[m.HandlerRegistrationResult].fail("bind-fail")

    class _RaiseBindingDispatcher:
        def register_handler(self, *_args: object):
            raise RuntimeError("bind-ex")

    registry._dispatcher = _FailBindingDispatcher()
    failed = registry.register_bindings({str: _Handler()})
    assert failed.is_failure

    registry._dispatcher = _RaiseBindingDispatcher()
    raised = registry.register_bindings({str: _Handler()})
    assert raised.is_failure


def test_get_plugin_and_register_metadata_and_list_items_exception(monkeypatch) -> None:
    registry = FlextRegistry()
    registry._registered_keys.add("cat::name")
    monkeypatch.setattr(
        registry.container,
        "get",
        lambda _key: r[t.GeneralValueType].fail("missing"),
    )
    missing = registry.get_plugin("cat", "name")
    assert missing.is_failure

    metadata_result = registry.register(
        "svc",
        object(),
        metadata=m.Metadata(attributes={"k": "v"}),
    )
    assert metadata_result.is_success

    monkeypatch.setattr(
        registry.container,
        "with_service",
        lambda _name, _service: (_ for _ in ()).throw(ValueError("nope")),
    )
    reg_fail = registry.register("svc2", object())
    assert reg_fail.is_failure

    broken = _BrokenListRegistry()
    listed = broken.list_items()
    assert listed.is_failure
