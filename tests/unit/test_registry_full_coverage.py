"""Coverage tests for FlextRegistry compatibility paths."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import override

import pytest

from flext_core import FlextRegistry
from tests import c, h, m, p, r, t


class _Handler(h[t.ValueOrModel, t.ValueOrModel]):
    """Test handler implementation."""

    @override
    def handle(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
        if isinstance(message, (str, int, float, bool)):
            return r[t.ValueOrModel].ok(message)
        return r[t.ValueOrModel].fail(c.Core.Tests.TestErrors.UNSUPPORTED_MESSAGE)

    @override
    def __call__(self, message: t.ValueOrModel) -> r[t.ValueOrModel]:
        return self.handle(message)


def _success_details(reg_id: str) -> m.RegistrationDetails:
    return m.RegistrationDetails(
        registration_id=reg_id,
        handler_mode=c.HandlerType.COMMAND,
        timestamp="",
        status=c.CommonStatus.RUNNING,
    )


def _as_registry_handler(handler: _Handler) -> _Handler:
    return handler


def test_summary_properties_and_subclass_storage_reset() -> None:
    detail = _success_details("a")
    summary = m.RegistrySummary(registered=[detail], errors=["x"])
    assert len(summary.registered) == 1
    assert len(summary.errors) == 1

    class _ChildRegistry(FlextRegistry):
        pass

    assert _ChildRegistry._class_plugin_storage == {}
    assert _ChildRegistry._class_registered_keys == set()


def test_execute_and_register_handler_failure_paths() -> None:
    class _FalseyDispatcher(p.Dispatcher):
        def __bool__(self) -> bool:
            return False

        @override
        def publish(
            self,
            event: p.Routable | Sequence[p.Routable],
        ) -> r[bool]:
            _ = event
            return r[bool].ok(True)

        @override
        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> r[bool]:
            _ = handler
            _ = is_event
            return r[bool].ok(True)

        @override
        def dispatch(self, message: p.Routable) -> r[t.RuntimeAtomic]:
            _ = message
            return r[t.RuntimeAtomic].fail(
                c.Core.Tests.TestErrors.DISPATCHER_UNCONFIGURED
            )

    execute_registry = FlextRegistry.create(dispatcher=_FalseyDispatcher())
    execute_result = execute_registry.execute()
    assert execute_result.failure

    class _FailDispatcher(p.Dispatcher):
        @override
        def publish(
            self,
            event: p.Routable | Sequence[p.Routable],
        ) -> r[bool]:
            _ = event
            return r[bool].ok(True)

        @override
        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> r[bool]:
            _ = handler
            _ = is_event
            return r[bool].fail(c.Core.Tests.TestErrors.DISPATCHER_FAIL)

        @override
        def dispatch(self, message: p.Routable) -> r[t.RuntimeAtomic]:
            _ = message
            return r[t.RuntimeAtomic].fail(c.Core.Tests.TestErrors.DISPATCHER_FAIL)

    failing_registry = FlextRegistry.create(dispatcher=_FailDispatcher())
    reg_result = failing_registry.register_handler(_as_registry_handler(_Handler()))
    assert reg_result.failure
    assert "dispatcher-fail" in (reg_result.error or "")

    class _OkDispatcher(p.Dispatcher):
        @override
        def publish(
            self,
            event: p.Routable | Sequence[p.Routable],
        ) -> r[bool]:
            _ = event
            return r[bool].ok(True)

        @override
        def register_handler(
            self,
            handler: t.HandlerProtocolVariant,
            *,
            is_event: bool = False,
        ) -> r[bool]:
            _ = handler
            _ = is_event
            return r[bool].ok(True)

        @override
        def dispatch(self, message: p.Routable) -> r[t.RuntimeAtomic]:
            _ = message
            return r[t.RuntimeAtomic].ok(True)

    success_registry = FlextRegistry.create(dispatcher=_OkDispatcher())
    fallback = success_registry.register_handler(_as_registry_handler(_Handler()))
    assert fallback.success
    assert fallback.value.registration_id != ""


def test_create_auto_discover_and_mode_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    def _register_handler_ok(
        _self: FlextRegistry,
        handler: t.HandlerLike,
    ) -> r[m.RegistrationDetails]:
        return r[m.RegistrationDetails].ok(
            _success_details(handler.__class__.__name__),
        )

    def fake_scan(
        _module: ModuleType,
    ) -> Sequence[tuple[str, Callable[..., t.Scalar | None], m.DecoratorConfig]]:
        cfg = m.DecoratorConfig(command=str, middleware=[])

        def fake_handler(_msg: t.Scalar) -> t.Scalar | None:
            return None

        return [
            (
                "x",
                fake_handler,
                cfg,
            ),
        ]

    monkeypatch.setattr(
        FlextRegistry,
        "register_handler",
        _register_handler_ok,
    )
    created = FlextRegistry.create(auto_discover_handlers=True)
    assert isinstance(created, FlextRegistry)
    registry = FlextRegistry()
    query_details = registry._create_registration_details(
        m.RegistrationResult(handler_name="q", status="active", mode="query"),
        "k1",
    )
    event_details = registry._create_registration_details(
        m.RegistrationResult(handler_name="e", status="active", mode="event"),
        "k2",
    )
    assert query_details.handler_mode == c.HandlerType.QUERY
    assert event_details.handler_mode == c.HandlerType.EVENT


def test_summary_error_paths_and_bindings_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _register_handler_fail(
        _self: FlextRegistry,
        _handler: t.HandlerLike,
    ) -> r[m.RegistrationDetails]:
        return r[m.RegistrationDetails].fail("x")

    registry = FlextRegistry()
    summary = m.RegistrySummary(errors=["something bad"])
    finalize = registry._finalize_summary(summary)
    assert finalize.failure
    monkeypatch.setattr(
        FlextRegistry,
        "register_handler",
        _register_handler_fail,
    )
    batch = registry.register_handlers([_as_registry_handler(_Handler())])
    assert batch.failure
    failed = registry.register_bindings({str: _as_registry_handler(_Handler())})
    assert failed.failure
    raised = registry.register_bindings({str: _as_registry_handler(_Handler())})
    assert raised.failure


def test_fetch_plugin_and_register_metadata_and_list_items_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FlextRegistry()
    registry._registered_keys.add("cat::name")

    def _container_get_fail(_key: str) -> r[str]:
        return r[str].fail("missing")

    def _container_register_raise(_name: str, _service: t.RegisterableService) -> None:
        msg = "nope"
        raise ValueError(msg)

    monkeypatch.setattr(
        registry.container,
        "get",
        _container_get_fail,
    )
    missing = registry.fetch_plugin("cat", "name")
    assert missing.failure
    metadata_result = registry.register(
        "svc",
        "service",
        metadata=m.Metadata(attributes={"k": "v"}),
    )
    assert metadata_result.success
    monkeypatch.setattr(
        registry.container,
        "register",
        _container_register_raise,
    )
    reg_fail = registry.register("svc2", "service")
    assert reg_fail.failure
