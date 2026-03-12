"""Coverage tests for FlextRegistry compatibility paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast, override

import pytest

from flext_core import FlextHandlers, FlextRegistry, c, h, m, p, r, t


class _Handler(FlextHandlers[object, object]):
    """Test handler implementation."""

    @override
    def handle(self, message: object) -> r[object]:
        return r[object].ok(message)

    def __call__(self, message: object) -> r[object]:
        return self.handle(message)

    def _protocol_name(self) -> str:
        """Return protocol name for registry identification."""
        return "Handler"


def _success_details(reg_id: str) -> m.HandlerRegistrationDetails:
    return m.HandlerRegistrationDetails(
        registration_id=reg_id,
        handler_mode=c.Cqrs.HandlerType.COMMAND,
        timestamp="",
        status=c.Cqrs.CommonStatus.RUNNING,
    )


def _as_registry_handler(handler: _Handler) -> _Handler:
    return handler


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
        def register_handler(self, *_args: object) -> r[m.Handler.RegistrationResult]:
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
        def register_handler(self, *_args: object) -> r[m.Handler.RegistrationResult]:
            return r[m.Handler.RegistrationResult].ok(
                m.Handler.RegistrationResult(
                    handler_name="h",
                    status="active",
                    mode="command",
                ),
            )

    setattr(
        registry,
        "_dispatcher",
        cast("p.CommandBus", cast("object", _OkDispatcher())),
    )

    def _create_registration_details_none(
        _self: FlextRegistry,
        _reg_result: m.HandlerRegistrationResult,
        _key: str,
    ) -> None:
        return None

    monkeypatch.setattr(
        FlextRegistry,
        "_create_registration_details",
        _create_registration_details_none,
    )
    fallback = registry.register_handler(_as_registry_handler(_Handler()))
    assert fallback.is_success
    assert fallback.value.registration_id != ""


def test_create_auto_discover_and_mode_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    discovered_handler = _Handler()

    def _register_handler_ok(
        _self: FlextRegistry,
        handler: p.Handler[object, object],
    ) -> r[m.HandlerRegistrationDetails]:
        return r[m.HandlerRegistrationDetails].ok(
            _success_details(handler.__class__.__name__),
        )

    def fake_scan(
        _module: object,
    ) -> list[tuple[str, Callable[..., object], m.Handler.DecoratorConfig]]:
        cfg = m.Handler.DecoratorConfig(command=str, middleware=[])
        return [("x", discovered_handler.handle, cfg)]

    monkeypatch.setattr(h.Discovery, "scan_module", staticmethod(fake_scan))
    monkeypatch.setattr(
        FlextRegistry,
        "register_handler",
        _register_handler_ok,
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


def test_summary_error_paths_and_bindings_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _register_handler_fail(
        _self: FlextRegistry,
        _handler: p.Handler[object, object],
    ) -> r[m.HandlerRegistrationDetails]:
        return r[m.HandlerRegistrationDetails].fail("x")

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
        _register_handler_fail,
    )
    batch = registry.register_handlers([_as_registry_handler(_Handler())])
    assert batch.is_failure

    class _FailBindingDispatcher:
        def register_handler(self, *_args: object) -> r[m.Handler.RegistrationResult]:
            return r[m.Handler.RegistrationResult].fail("bind-fail")

    class _RaiseBindingDispatcher:
        def register_handler(self, *_args: object) -> r[m.Handler.RegistrationResult]:
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

    def _container_get_fail(_key: str) -> r[t.JsonValue]:
        return r[t.JsonValue].fail("missing")

    def _container_register_raise(_name: str, _service: t.RegisterableService) -> None:
        msg = "nope"
        raise ValueError(msg)

    monkeypatch.setattr(
        registry.container,
        "get",
        _container_get_fail,
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
        "register",
        _container_register_raise,
    )
    reg_fail = registry.register("svc2", "service")
    assert reg_fail.is_failure
