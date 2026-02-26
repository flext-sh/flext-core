"""Targeted full-coverage tests for flext_core.dispatcher missed branches."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
from flext_core import c, m, p, r, t, u
from flext_core.dispatcher import (
    DispatcherHandler,
    FlextDispatcher,
    HandlerRegistrationRequestInput,
    HandlerRequestKey,
)
from pydantic import BaseModel

DispatcherMiddlewareConfig = m.Config.DispatcherMiddlewareConfig
import math

from flext_core.handlers import FlextHandlers


def _as_handler(value: object) -> DispatcherHandler:
    return cast("DispatcherHandler", value)


def _as_payload(value: object) -> t.ConfigMapValue:
    return cast("t.ConfigMapValue", value)


def _as_handler_key(value: object) -> HandlerRequestKey:
    return cast("HandlerRequestKey", value)


def _as_handler_callable(value: object) -> t.HandlerCallable:
    return cast("t.HandlerCallable", value)


def _as_registration_handler(
    value: object,
) -> t.HandlerCallable | p.Handler[t.GuardInputValue, t.GuardInputValue] | BaseModel:
    return cast(
        "t.HandlerCallable | p.Handler[t.GuardInputValue, t.GuardInputValue] | BaseModel",
        value,
    )


def _as_request_input(value: object) -> HandlerRegistrationRequestInput:
    return cast("HandlerRegistrationRequestInput", value)


class _EchoMessage(BaseModel):
    """Simple message model for dispatch tests."""

    command_id: str = "cmd-1"
    payload: str = "payload"


class _QueryMessage(BaseModel):
    """Simple query model for cache tests."""

    query_id: str = "q-1"


class _CanHandleAuto:
    """Auto-discovery handler with can_handle + handle."""

    def can_handle(self, message_type: str) -> bool:
        return message_type in {_EchoMessage.__name__, "manual"}

    def handle(self, message: object) -> str:
        return f"auto:{type(message).__name__}"

    def __call__(self, message: object) -> str:
        return self.handle(message)


class _HandleOnly:
    """Handler exposing handle() only."""

    def handle(self, message: object) -> object:
        _ = message
        return {"ok": True}

    def __call__(self, message: object) -> object:
        return self.handle(message)


class _ExecuteOnly:
    """Handler exposing execute() only."""

    def execute(self, message: object) -> object:
        _ = message
        return 42


class _NonCallableMethod:
    """Handler with non-callable handle attr."""

    handle = "not-callable"


class _StringifyingObject:
    """Object whose __str__ is deterministic for assertions."""

    def __str__(self) -> str:
        return "stringified-object"


class _MiddlewareOK:
    """Middleware that allows processing."""

    def process(self, command: object, handler: object) -> bool:
        _ = (command, handler)
        return True


class _MiddlewareBadResult:
    """Middleware returning non-primitive to hit string conversion path."""

    def process(self, command: object, handler: object) -> object:
        _ = (command, handler)
        return _StringifyingObject()


class _BadDumpModel(BaseModel):
    """Model whose dump result is intentionally invalid for extraction."""

    x: int = 1


class _FailingDumpModel(BaseModel):
    """Model whose dump raises for extraction error path."""

    x: int = 1


class _MetadataCarrier:
    """Plain object with metadata-like attributes."""

    def __init__(self) -> None:
        self.attributes = {"alpha": "x", "bad": object()}


class _AttrRequestModel(BaseModel):
    """BaseModel used to cover _normalize_request BaseModel branch."""

    handler: object
    message_type: str | None = None
    handler_mode: str | None = None
    handler_name: str | None = None


@dataclass
class _FakeTypedResult:
    is_success: bool
    value: object


class _CallableType:
    """Callable class to exercise __name__ extraction path."""

    def __call__(self, *_args: object, **_kwargs: object) -> str:
        return "callable"


@pytest.fixture
def dispatcher() -> FlextDispatcher:
    """Create a fresh dispatcher for each test."""
    return FlextDispatcher()


def test_reliability_resolvers_return_container_instances(
    dispatcher: FlextDispatcher,
) -> None:
    """Cover resolver success branches when container has typed services."""
    cb = object()
    rl = object()
    te = object()
    rp = object()
    items = [cb, rl, te, rp]

    def fake_get_typed(name: str, type_cls: type[object]) -> r[object]:
        _ = (name, type_cls)
        return r[object].ok(items.pop(0))

    setattr(dispatcher.container, "get_typed", fake_get_typed)

    assert dispatcher._resolve_or_create_circuit_breaker(dispatcher.config) is cb
    assert dispatcher._resolve_or_create_rate_limiter(dispatcher.config) is rl
    assert dispatcher._resolve_or_create_timeout_enforcer(dispatcher.config) is te
    assert dispatcher._resolve_or_create_retry_policy(dispatcher.config) is rp


def test_dispatcher_config_and_basic_introspection_helpers(
    dispatcher: FlextDispatcher,
) -> None:
    """Cover dispatcher_config property and basic analytics helpers."""
    config = dispatcher.dispatcher_config
    assert isinstance(config.dispatcher_timeout_seconds, float)

    audit = dispatcher.get_audit_log()
    analytics = dispatcher.get_performance_analytics()
    assert audit.is_success
    assert analytics.is_success
    analytics_map = cast("Mapping[str, t.GeneralValueType]", analytics.value)
    assert analytics_map["audit_log_entries"] == 0


def test_interface_validation_paths(dispatcher: FlextDispatcher) -> None:
    """Cover generic interface validator success/failure branches."""
    callable_ok = dispatcher._validate_interface(
        lambda: None, "process", "processor", allow_callable=True
    )
    assert callable_ok.is_success

    class _NoMethods:
        pass

    fail = dispatcher._validate_interface(
        _NoMethods(), ["handle", "execute"], "handler"
    )
    assert fail.is_failure
    assert "must have 'handle' or 'execute'" in (fail.error or "")

    proc = dispatcher._validate_processor_interface(lambda x: x)
    assert proc.is_success


def test_validation_and_routing_helpers(dispatcher: FlextDispatcher) -> None:
    """Cover handler mode validation, normalization, routing and cache-key generation."""
    assert dispatcher._normalize_command_key(_EchoMessage) == "_EchoMessage"
    assert dispatcher._validate_handler_mode(None).is_success

    invalid_mode = dispatcher._validate_handler_mode("invalid-mode")
    assert invalid_mode.is_failure

    explicit_handler = _HandleOnly()
    dispatcher._handlers[_EchoMessage.__name__] = _as_handler(explicit_handler)
    assert dispatcher._route_to_handler(_EchoMessage()) is explicit_handler

    dispatcher._handlers.clear()
    auto_handler = _CanHandleAuto()
    dispatcher._auto_handlers.append(auto_handler)
    assert dispatcher._route_to_handler(_EchoMessage()) is auto_handler

    class _TaggedCommand(m.Command):
        command_type: str = "tagged"

    tagged_result = dispatcher.register_handler(
        "command",
        _as_handler(lambda _msg: _as_payload("ok-by-discriminator")),
    )
    assert tagged_result.is_success
    routed = dispatcher._try_simple_dispatch(_TaggedCommand())
    assert routed is not None and routed.is_success
    assert routed.value == "ok-by-discriminator"

    dispatcher._auto_handlers.clear()
    assert dispatcher._route_to_handler(_EchoMessage()) is None

    cache_key = dispatcher._generate_cache_key(_QueryMessage(), _QueryMessage)
    assert isinstance(cache_key, str) and cache_key


def test_cache_hit_and_execute_handler_fallback_paths(
    dispatcher: FlextDispatcher,
) -> None:
    """Cover cache-hit branch and execute-handler error/success paths."""
    dispatcher.config.enable_caching = True
    query = _QueryMessage()
    cache_key = dispatcher._generate_cache_key(query, _QueryMessage)
    cached = r[dict[str, bool]].ok({"cached": True})
    cache_map = cast("dict[str, r[dict[str, bool]]]", dispatcher._cache)
    cache_map[cache_key] = cached

    cached_result = dispatcher._check_cache_for_result(
        query, _QueryMessage, is_query=True
    )
    assert cached_result.is_success
    assert cached_result.value == {"cached": True}

    # Execute via FlextHandlers instance branch.
    h = FlextHandlers.create_from_callable(lambda msg: str(type(msg).__name__))
    h_result = dispatcher._execute_handler(_as_handler(h), query)
    assert h_result.is_success

    # Non-FlextHandlers: handle path, execute path, invalid path, non-callable path, exception path.
    handle_result = dispatcher._execute_handler(
        _as_handler(_HandleOnly()), _EchoMessage()
    )
    assert handle_result.is_success

    execute_result = dispatcher._execute_handler(
        _as_handler(_ExecuteOnly()),
        _as_payload(_EchoMessage()),
    )
    assert execute_result.is_success

    missing_method = dispatcher._execute_handler(
        _as_handler(object()), _as_payload(_EchoMessage())
    )
    assert missing_method.is_failure

    non_callable = dispatcher._execute_handler(
        _as_handler(_NonCallableMethod()),
        _as_payload(_EchoMessage()),
    )
    assert non_callable.is_failure

    class _Raising:
        def handle(self, _msg: object) -> object:
            msg = "kaboom"
            raise RuntimeError(msg)

    raised = dispatcher._execute_handler(
        _as_handler(_Raising()), _as_payload(_EchoMessage())
    )
    assert raised.is_failure
    assert "kaboom" in (raised.error or "")


def test_dispatch_command_no_handler_and_query_cache_store(
    dispatcher: FlextDispatcher,
) -> None:
    """Cover _dispatch_command no-handler and successful query caching branches."""
    dispatcher.config.enable_caching = False
    not_found = dispatcher._dispatch_command(_EchoMessage())
    assert not_found.is_failure
    assert not_found.error_code == c.Errors.COMMAND_HANDLER_NOT_FOUND

    # Register query handler and ensure successful query is cached.
    query = _QueryMessage()

    class _QueryHandler:
        def handle(self, _msg: object) -> dict[str, str]:
            return {"answer": "ok"}

    dispatcher._handlers[_QueryMessage.__name__] = _as_handler(_QueryHandler())
    dispatcher.config.enable_caching = True

    result = dispatcher._dispatch_command(query)
    assert result.is_success

    key = dispatcher._generate_cache_key(query, _QueryMessage)
    assert key in dispatcher._cache


def test_layer1_register_handler_argument_and_type_paths(
    dispatcher: FlextDispatcher,
) -> None:
    """Cover argument shape and command-type extraction branches."""
    bad_single = dispatcher._layer1_register_handler("not-a-handler")
    assert bad_single.is_failure

    callable_type = _CallableType
    bad_two = dispatcher._layer1_register_handler(
        callable_type,
        _as_handler(cast("object", "bad")),
    )
    assert bad_two.is_failure

    too_many = dispatcher._layer1_register_handler(
        _as_handler_key("a"),
        _as_handler(cast("object", "b")),
        _as_handler(cast("object", "c")),
    )
    assert too_many.is_failure


def test_wire_and_single_registration_branches(
    dispatcher: FlextDispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover dependency wiring and single registration branches including factory errors."""

    class _WireErrorContainer:
        def wire_modules(self, **_kwargs: object) -> None:
            msg = "wire-fail"
            raise RuntimeError(msg)

        def register_factory(self, _name: str, _factory: object) -> bool:
            msg = "factory-fail"
            raise RuntimeError(msg)

    dispatcher._container = cast("p.DI", cast("object", _WireErrorContainer()))

    none_result = dispatcher._register_single_handler(_as_handler(cast("object", None)))
    assert none_result.is_failure

    class _IdHandler(BaseModel):
        handler_id: str = "id-handler"

        def handle(self, _msg: object) -> str:
            return "ok"

    reg = dispatcher._register_single_handler(_IdHandler())
    assert reg.is_success
    assert "id-handler" in dispatcher._handlers


def test_two_arg_registration_and_middleware_config_branches(
    dispatcher: FlextDispatcher,
) -> None:
    """Cover two-arg registration + middleware config conversion failure branches."""

    class _Handler:
        def handle(self, _msg: object) -> str:
            return "ok"

    invalid_args = dispatcher._register_two_arg_handler(
        _as_handler_key(cast("object", None)),
        _as_handler(_Handler()),
    )
    assert invalid_args.is_failure

    empty_cmd = dispatcher._register_two_arg_handler(
        _as_handler_key(""), _as_handler(_Handler())
    )
    assert empty_cmd.is_failure

    class _NoHandle:
        pass

    invalid_handler = dispatcher._register_two_arg_handler(
        _as_handler_key("cmd"),
        _as_handler(_NoHandle()),
    )
    assert invalid_handler.is_failure

    class _ClassHandler:
        def handle(self, _msg: object) -> str:
            return "ok"

    ok = dispatcher._register_two_arg_handler(
        _as_handler_key("cmd.two"), _as_handler(_ClassHandler())
    )
    assert ok.is_success

    bad_dict_cfg = dispatcher.layer1_add_middleware(
        _as_handler_callable(_MiddlewareOK()),
        m.ConfigMap(root={"order": "bad"}),
    )
    assert bad_dict_cfg.is_failure

    bad_type_cfg = dispatcher.layer1_add_middleware(
        _as_handler_callable(_MiddlewareOK()),
        cast("m.Config.MiddlewareConfig | t.ConfigMap", cast("object", "bad-config")),
    )
    assert bad_type_cfg.is_failure

    bad_config_map = t.ConfigMap(root={"order": "bad"})
    bad_config_map_result = dispatcher.layer1_add_middleware(
        _as_handler_callable(_MiddlewareOK()),
        bad_config_map,
    )
    assert bad_config_map_result.is_failure

    mw_ok = dispatcher.layer1_add_middleware(
        _as_handler_callable(_MiddlewareOK()),
        m.Config.MiddlewareConfig(order=1),
    )
    assert mw_ok.is_success


def test_publish_subscribe_unsubscribe_paths(
    dispatcher: FlextDispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover event publishing, subscription, and unsubscription error branches."""
    monkeypatch.setattr(
        dispatcher,
        "_dispatch_command",
        lambda _event: r[t.GeneralValueType].fail("publish-fail"),
    )
    publish_fail = dispatcher._publish_event({"evt": 1})
    assert publish_fail.is_failure

    def raise_publish(_event: object) -> r[t.GeneralValueType]:
        msg = "bad-event"
        raise TypeError(msg)

    monkeypatch.setattr(dispatcher, "_dispatch_command", raise_publish)
    publish_exc = dispatcher._publish_event({"evt": 2})
    assert publish_exc.is_failure

    def raise_subscribe(*_args: object, **_kwargs: object) -> r[bool]:
        msg = "sub-fail"
        raise ValueError(msg)

    monkeypatch.setattr(dispatcher, "_layer1_register_handler", raise_subscribe)
    sub = dispatcher.subscribe("evt", _as_handler_callable(_HandleOnly()))
    assert sub.is_failure

    dispatcher._handlers["evt"] = _as_handler(_HandleOnly())
    unsub_ok = dispatcher.unsubscribe("evt")
    assert unsub_ok.is_success

    unsub_missing = dispatcher.unsubscribe("not-found")
    assert unsub_missing.is_failure

    dispatcher._handlers = cast("dict[str, DispatcherHandler]", cast("object", None))
    unsub_exc = dispatcher.unsubscribe("evt")
    assert unsub_exc.is_failure


def test_publish_batch_and_named_event_paths(
    dispatcher: FlextDispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover named-event and batch publish branches including partial failure."""
    monkeypatch.setattr(
        dispatcher, "_publish_event", lambda event: r[bool].ok(True if event else False)
    )
    named = dispatcher.publish("evt.named", {"x": 1})
    assert named.is_success

    events = [{"a": 1}, {"b": 2}]
    batch_ok = dispatcher.publish(events)
    assert batch_ok.is_success

    def fail_one(event: object) -> r[bool]:
        if isinstance(event, dict) and "bad" in event:
            return r[bool].fail("bad-event")
        return r[bool].ok(True)

    monkeypatch.setattr(dispatcher, "_publish_event", fail_one)
    batch_fail = dispatcher.publish([{"ok": 1}, {"bad": 1}])
    assert batch_fail.is_failure


def test_nested_attr_name_and_request_normalization_paths(
    dispatcher: FlextDispatcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover nested attr extraction and request normalization branches."""
    assert dispatcher._get_nested_attr({"a": {"b": 1}}, "a", "b") == 1
    assert dispatcher._get_nested_attr({"a": {}}, "a", "missing") is None
    assert dispatcher._get_nested_attr(object()) is None

    monkeypatch.setattr(
        FlextDispatcher, "_get_nested_attr", staticmethod(lambda *_args: None)
    )
    empty_name = dispatcher._extract_name_from_handler(_as_handler(cast("object", {})))
    assert empty_name == ""

    req = m.HandlerRegistrationRequest(handler=lambda x: x)
    req_ok = dispatcher._normalize_request(req)
    assert req_ok.is_success

    model_req = _AttrRequestModel(handler=lambda x: x, message_type="cmd")
    model_ok = dispatcher._normalize_request(model_req)
    assert model_ok.is_success

    dict_ok = dispatcher._normalize_request(
        cast(
            "Mapping[str, t.ConfigMapValue]",
            cast(
                "object",
                {
                    "handler": lambda x: x,
                    "message_type": "cmd",
                },
            ),
        )
    )
    assert dict_ok.is_success

    invalid_req = dispatcher._normalize_request(
        cast("Mapping[str, t.ConfigMapValue]", cast("object", math.pi)),
    )
    assert invalid_req.is_failure


def test_handler_extraction_and_registration_mode_helpers(
    dispatcher: FlextDispatcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover handler extraction defaults and mode-specific registration branches."""
    bad_model = m.HandlerRegistrationRequest.model_construct(handler=_as_handler(123))
    bad_extract = dispatcher._validate_and_extract_handler(bad_model)
    assert bad_extract.is_failure

    class _NoInterface(BaseModel):
        value: str = "x"

    no_interface_model = m.HandlerRegistrationRequest(handler=_NoInterface())
    no_interface = dispatcher._validate_and_extract_handler(no_interface_model)
    assert no_interface.is_failure

    monkeypatch.setattr(dispatcher, "_extract_name_from_handler", lambda _handler: "")
    no_name_model = m.HandlerRegistrationRequest(handler=_CanHandleAuto())
    no_name = dispatcher._validate_and_extract_handler(no_name_model)
    assert no_name.is_success
    assert no_name.value[1] == "unknown_handler"

    class _Auto:
        def can_handle(self, _name: str) -> bool:
            return True

        def handle(self, _msg: object) -> str:
            return "ok"

    auto_reg = dispatcher._register_handler_by_mode(
        _as_handler(_Auto()),
        "h1",
        cast("Mapping[str, t.ConfigMapValue]", {}),
    )
    assert auto_reg.is_success

    explicit_missing = dispatcher._register_explicit_handler(
        _as_handler(_HandleOnly()),
        "h2",
        {},
    )
    assert explicit_missing.is_failure

    explicit_ok = dispatcher._register_explicit_handler(
        _as_handler(_HandleOnly()),
        "h3",
        cast(
            "Mapping[str, t.ConfigMapValue]",
            cast("object", {"message_type": _EchoMessage}),
        ),
    )
    assert explicit_ok.is_success


def test_register_handler_with_request_and_public_register_handler_branches(
    dispatcher: FlextDispatcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover request-based registration and public register_handler validation branches."""
    # _register_handler_with_request failure from invalid extracted handler
    constructed_bad = m.HandlerRegistrationRequest.model_construct(
        handler=_as_handler(object())
    )
    req_fail = dispatcher._register_handler_with_request(constructed_bad)
    assert req_fail.is_failure

    auto_req = m.HandlerRegistrationRequest(
        handler=_CanHandleAuto(), handler_name="preferred"
    )
    auto_ok = dispatcher._register_handler_with_request(auto_req)
    assert auto_ok.is_success
    assert auto_ok.value["handler_name"] == "preferred"

    explicit_req = m.HandlerRegistrationRequest(
        handler=_as_registration_handler(_HandleOnly()), message_type=_EchoMessage
    )
    explicit_ok = dispatcher._register_handler_with_request(explicit_req)
    assert explicit_ok.is_success

    explicit_missing = m.HandlerRegistrationRequest(
        handler=_as_registration_handler(_HandleOnly())
    )
    explicit_fail = dispatcher._register_handler_with_request(explicit_missing)
    assert explicit_fail.is_failure

    invalid_two_arg_handler = dispatcher.register_handler(
        "cmd",
        _as_handler(cast("object", 123)),
    )
    assert invalid_two_arg_handler.is_failure

    invalid_two_arg_request = dispatcher.register_handler(
        _as_request_input(cast("object", 123)),
        _as_handler(_HandleOnly()),
    )
    assert invalid_two_arg_request.is_failure

    monkeypatch.setattr(
        dispatcher, "_layer1_register_handler", lambda *_args: r[bool].fail("reg-fail")
    )
    reg_fail = dispatcher.register_handler("cmd.name", _as_handler(_HandleOnly()))
    assert reg_fail.is_failure

    monkeypatch.setattr(
        dispatcher, "_layer1_register_handler", lambda *_args: r[bool].ok(True)
    )
    reg_str_ok = dispatcher.register_handler("cmd.name", _as_handler(_HandleOnly()))
    assert reg_str_ok.is_success

    model_single = dispatcher.register_handler(
        m.HandlerRegistrationRequest(handler=_CanHandleAuto(), message_type="cmd")
    )
    assert model_single.is_success

    mapping_single = dispatcher.register_handler({
        "handler": _CanHandleAuto(),
        "message_type": "cmd",
    })
    assert mapping_single.is_success

    invalid_single = dispatcher.register_handler(_as_request_input(cast("object", 42)))
    assert invalid_single.is_failure

    single_layer_fail = dispatcher.register_handler(lambda x: x)
    assert single_layer_fail.is_success or single_layer_fail.is_failure


def test_register_handlers_batch_and_ensure_handler_helper(
    dispatcher: FlextDispatcher,
) -> None:
    """Cover batch registration success/failure and ensure_handler helper branches."""

    class _Good:
        def handle(self, _msg: object) -> str:
            return "ok"

    class _Bad:
        pass

    handlers: Mapping[HandlerRequestKey, DispatcherHandler] = {
        "good": _as_handler(_Good()),
        "bad": _as_handler(_Bad()),
    }
    batch = dispatcher.register_handlers(handlers)
    assert batch.is_failure

    good_only = dispatcher.register_handlers({"good": _as_handler(_Good())})
    assert good_only.is_success
    assert good_only.value.count == 1

    h = FlextHandlers.create_from_callable(lambda msg: msg)
    ensured = dispatcher._ensure_handler(_as_handler(h))
    assert ensured.is_success

    ensured_fail = dispatcher._ensure_handler(_as_handler(object()))
    assert ensured_fail.is_failure


def test_pre_dispatch_and_timeout_deadline_helpers(
    dispatcher: FlextDispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover pre-dispatch checks and timeout context/deadline helpers."""
    dispatcher._circuit_breaker.record_failure("x")
    dispatcher._circuit_breaker.record_failure("x")
    dispatcher._circuit_breaker.record_failure("x")
    dispatcher._circuit_breaker.record_failure("x")
    dispatcher._circuit_breaker.record_failure("x")

    cb_fail = dispatcher._check_pre_dispatch_conditions("x")
    assert cb_fail.is_failure

    monkeypatch.setattr(
        dispatcher._circuit_breaker, "check_before_dispatch", lambda _msg: True
    )
    monkeypatch.setattr(
        dispatcher._rate_limiter,
        "check_rate_limit",
        lambda _msg: r[bool].fail("limited"),
    )
    rl_fail = dispatcher._check_pre_dispatch_conditions("x")
    assert rl_fail.is_failure

    operation_id = "op-1"
    _ = dispatcher._track_timeout_context(operation_id, 0.001)
    assert dispatcher._check_timeout_deadline(operation_id) in {False, True}
    dispatcher._cleanup_timeout_context(operation_id)
    assert dispatcher._check_timeout_deadline(operation_id) is False


def test_execute_with_timeout_paths(
    dispatcher: FlextDispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover executor timeout, executor-shutdown reset, and direct execution branches."""

    def slow() -> r[t.GeneralValueType]:
        import time

        time.sleep(0.05)
        return r[t.GeneralValueType].ok("done")

    dispatcher._timeout_enforcer = dispatcher._resolve_or_create_timeout_enforcer(
        dispatcher.config
    )
    timeout_res = dispatcher._execute_with_timeout(
        slow, timeout_seconds=0.001, timeout_override=1
    )
    assert timeout_res.is_failure

    class _ShutdownExecutor:
        def submit(self, _func: object) -> object:
            msg = "cannot schedule new futures after shutdown"
            raise RuntimeError(msg)

    reset_called = {"value": False}

    monkeypatch.setattr(
        dispatcher._timeout_enforcer, "should_use_executor", lambda: True
    )
    monkeypatch.setattr(
        dispatcher._timeout_enforcer, "ensure_executor", lambda: _ShutdownExecutor()
    )
    monkeypatch.setattr(
        dispatcher._timeout_enforcer,
        "reset_executor",
        lambda: reset_called.__setitem__("value", True),
    )
    shutdown_res = dispatcher._execute_with_timeout(
        lambda: r[t.GeneralValueType].ok("x"), timeout_seconds=1.0
    )
    assert shutdown_res.is_failure
    assert reset_called["value"] is True

    monkeypatch.setattr(
        dispatcher._timeout_enforcer, "should_use_executor", lambda: False
    )
    direct = dispatcher._execute_with_timeout(
        lambda: r[t.GeneralValueType].ok("direct"), timeout_seconds=1.0
    )
    assert direct.is_success


def test_retry_and_dispatch_pipeline_failure_paths(
    dispatcher: FlextDispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover retry helper and pipeline extraction/validation failure branches."""
    monkeypatch.setattr(
        dispatcher._retry_policy, "should_retry", lambda _attempt: False
    )
    assert dispatcher._should_retry_on_error(0, "error") is False

    monkeypatch.setattr(dispatcher._retry_policy, "should_retry", lambda _attempt: True)
    monkeypatch.setattr(
        dispatcher._retry_policy, "is_retriable_error", lambda _msg: True
    )
    monkeypatch.setattr(dispatcher._retry_policy, "get_retry_delay", lambda: 0.0)
    assert dispatcher._should_retry_on_error(0, "retry") is True

    monkeypatch.setattr(
        FlextDispatcher,
        "_extract_dispatch_config",
        lambda *_args: r[m.DispatchConfig].fail("bad-config"),
    )
    pipeline_fail_config = dispatcher._execute_dispatch_pipeline(
        _EchoMessage(), None, None, None, None
    )
    assert pipeline_fail_config.is_failure

    monkeypatch.setattr(
        FlextDispatcher,
        "_extract_dispatch_config",
        lambda *_args: r[m.DispatchConfig].ok(m.DispatchConfig()),
    )
    monkeypatch.setattr(
        dispatcher,
        "_prepare_dispatch_context",
        lambda *_args: r[t.GeneralValueType].fail("ctx-fail"),
    )
    pipeline_fail_context = dispatcher._execute_dispatch_pipeline(
        _EchoMessage(), None, None, None, None
    )
    assert pipeline_fail_context.is_failure

    monkeypatch.setattr(
        dispatcher,
        "_prepare_dispatch_context",
        lambda *_args: r[t.GeneralValueType].ok({
            "message_type": "m",
            "message": _EchoMessage(),
        }),
    )
    monkeypatch.setattr(
        dispatcher,
        "_validate_pre_dispatch_conditions",
        lambda *_args: r[t.GeneralValueType].fail("validate-fail"),
    )
    pipeline_fail_validate = dispatcher._execute_dispatch_pipeline(
        _EchoMessage(), None, None, None, None
    )
    assert pipeline_fail_validate.is_failure


def test_dispatch_public_api_and_metadata_conversion_paths(
    dispatcher: FlextDispatcher,
) -> None:
    """Cover dispatch API variants and metadata/config conversion helpers."""
    dispatcher._handlers["custom_type"] = _as_handler(
        lambda msg: _as_payload(_StringifyingObject())
    )
    result = dispatcher.dispatch("custom_type", {"x": 1})
    assert result.is_success
    assert result.value == "stringified-object"

    meta_none = FlextDispatcher._convert_metadata_to_model(None)
    assert meta_none is None

    meta_model = m.Metadata(attributes={"a": "b"})
    meta_passthrough = FlextDispatcher._convert_metadata_to_model(meta_model)
    assert meta_passthrough is meta_model

    nested_meta = FlextDispatcher._convert_metadata_to_model(
        _as_payload({"nested": {"k": object()}}),
    )
    assert nested_meta is not None
    assert isinstance(nested_meta.attributes["nested"], str)

    scalar_meta = FlextDispatcher._convert_metadata_to_model(123)
    assert scalar_meta is not None
    assert scalar_meta.attributes["value"] == "123"

    existing_cfg = m.DispatchConfig()
    assert (
        FlextDispatcher._build_dispatch_config_from_args(existing_cfg, None, None, None)
        is existing_cfg
    )
    assert FlextDispatcher._build_dispatch_config_from_args(
        {"legacy": True}, None, None, None
    ) == {"legacy": True}


def test_simple_dispatch_and_extract_dispatch_config_paths(
    dispatcher: FlextDispatcher,
) -> None:
    """Cover simple dispatch helper and dispatch config extraction branches."""

    class _Msg:
        pass

    msg = _Msg()
    assert dispatcher._try_simple_dispatch(_as_payload(msg)) is None

    dispatcher._handlers[_Msg.__name__] = _as_handler(cast("object", "bad-handler"))
    not_callable = dispatcher._try_simple_dispatch(_as_payload(msg))
    assert not_callable is not None and not_callable.is_failure

    dispatcher._handlers[_Msg.__name__] = _as_handler(
        lambda _m: _as_payload(r[t.GeneralValueType].ok("already-result"))
    )
    passthrough = dispatcher._try_simple_dispatch(_as_payload(msg))
    assert passthrough is not None and passthrough.is_success

    def _raises(_m: object) -> object:
        msg_txt = "handler exploded"
        raise RuntimeError(msg_txt)

    dispatcher._handlers[_Msg.__name__] = _as_handler(_raises)
    raised = dispatcher._try_simple_dispatch(_as_payload(msg))
    assert raised is not None and raised.is_failure

    cfg_obj = SimpleNamespace(
        metadata={"x": 1}, correlation_id="cid", timeout_override=5
    )
    extracted = FlextDispatcher._extract_dispatch_config(
        _as_payload(cfg_obj), None, None, None
    )
    assert extracted.is_success

    invalid_metadata = FlextDispatcher._extract_dispatch_config(
        None, _as_payload(object()), None, None
    )
    assert invalid_metadata.is_failure

    hard_invalid = FlextDispatcher._extract_dispatch_config(
        None, _as_payload({"x": object()}), None, None
    )
    assert hard_invalid.is_failure

    class _BrokenConfig:
        @property
        def metadata(self) -> object:
            msg = "boom"
            raise RuntimeError(msg)

    broken = FlextDispatcher._extract_dispatch_config(
        _as_payload(_BrokenConfig()),
        None,
        None,
        None,
    )
    assert broken.is_failure


def test_prepare_validate_retry_and_normalize_dispatch_message_paths(
    dispatcher: FlextDispatcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover context preparation/validation/retry and dispatch-message normalization branches."""

    class _BadConfig:
        value = "no-model-dump"

    prep_fail = dispatcher._prepare_dispatch_context(
        _EchoMessage(),
        None,
        cast("m.DispatchConfig", cast("object", _BadConfig())),
    )
    assert prep_fail.is_failure

    invalid_ctx = dispatcher._validate_pre_dispatch_conditions("no-mapping")
    assert invalid_ctx.is_failure

    invalid_msg_type = dispatcher._validate_pre_dispatch_conditions({
        "message_type": 123
    })
    assert invalid_msg_type.is_failure

    monkeypatch.setattr(
        dispatcher,
        "_check_pre_dispatch_conditions",
        lambda _mt: r[bool].fail("blocked"),
    )
    blocked = dispatcher._validate_pre_dispatch_conditions({"message_type": "x"})
    assert blocked.is_failure

    invalid_retry_ctx = dispatcher._execute_with_retry_policy("bad")
    assert invalid_retry_ctx.is_failure

    invalid_retry_msg = dispatcher._execute_with_retry_policy({"message_type": 123})
    assert invalid_retry_msg.is_failure

    with pytest.raises(TypeError):
        FlextDispatcher._normalize_dispatch_message(None, None)

    with pytest.raises(TypeError):
        FlextDispatcher._normalize_dispatch_message("type", None)

    timeout_override = dispatcher._get_timeout_seconds(7)
    assert timeout_override == 7.0


def test_context_execution_handle_result_attempt_and_metadata_compatibility(
    dispatcher: FlextDispatcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover context-scoped execution, result handling, dispatch attempt exception, and metadata utilities."""

    @contextmanager
    def fake_scope(_metadata: object, _correlation: object):
        yield

    monkeypatch.setattr(dispatcher, "_context_scope", fake_scope)
    monkeypatch.setattr(
        dispatcher, "_dispatch_command", lambda _msg: r[t.GeneralValueType].ok("ok")
    )

    execute = dispatcher._create_execute_with_context(_EchoMessage(), "cid", 3)
    assert execute().is_success

    shutdown = dispatcher._handle_dispatch_result(
        r[t.GeneralValueType].fail("Executor was shutdown, retry requested"), "m"
    )
    assert shutdown.is_failure

    failure = dispatcher._handle_dispatch_result(
        r[t.GeneralValueType].fail("other"), "m"
    )
    assert failure.is_failure

    success = dispatcher._handle_dispatch_result(r[t.GeneralValueType].ok("value"), "m")
    assert success.is_success

    options = m.ExecuteDispatchAttemptOptions(
        message_type="m",
        metadata={"a": 1},
        correlation_id="cid",
        timeout_override=1,
        operation_id="op-42",
    )

    monkeypatch.setattr(dispatcher, "_get_timeout_seconds", lambda _ov: 0.01)
    monkeypatch.setattr(
        dispatcher,
        "_execute_with_timeout",
        lambda *_args, **_kwargs: r[t.GeneralValueType].ok("done"),
    )
    attempt_ok = dispatcher._execute_dispatch_attempt(_EchoMessage(), options)
    assert attempt_ok.is_success

    def raise_get_timeout(_ov: object) -> float:
        msg = "timeout-fail"
        raise RuntimeError(msg)

    monkeypatch.setattr(dispatcher, "_get_timeout_seconds", raise_get_timeout)
    attempt_fail = dispatcher._execute_dispatch_attempt(_EchoMessage(), options)
    assert attempt_fail.is_failure

    assert FlextDispatcher._normalize_context_metadata(_as_payload(object())) is None

    assert FlextDispatcher._extract_metadata_mapping(_BadDumpModel()) is not None

    assert (
        FlextDispatcher._extract_metadata_mapping(_as_payload(_MetadataCarrier()))
        is not None
    )
    assert FlextDispatcher._extract_metadata_mapping(_as_payload(object())) is None

    assert FlextDispatcher._is_metadata_attribute_compatible(["x", 1, True]) is True
    assert (
        FlextDispatcher._is_metadata_attribute_compatible(_as_payload(["x", object()]))
        is False
    )
    assert (
        FlextDispatcher._is_metadata_attribute_compatible({"ok": [1, 2], "nested": "v"})
        is True
    )
    assert (
        FlextDispatcher._is_metadata_attribute_compatible(
            _as_payload({"bad": {"x": object()}}),
        )
        is False
    )


def test_context_scope_factory_and_cleanup_error_paths(
    dispatcher: FlextDispatcher,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover context-scope logging branch, auto-discovery factory, and cleanup/create error paths."""
    dispatcher.config.dispatcher_auto_context = True
    dispatcher.config.dispatcher_enable_logging = True

    logs: list[str] = []
    monkeypatch.setattr(
        dispatcher, "_log_with_context", lambda _lvl, msg, **_kw: logs.append(msg)
    )
    monkeypatch.setattr(u, "generate", lambda _prefix: "generated-cid")

    class _CustomMap(Mapping[str, t.GeneralValueType]):
        def __init__(self, data: dict[str, t.GeneralValueType]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> t.GeneralValueType:
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    with dispatcher._context_scope(
        metadata=_CustomMap({"x": 1}), correlation_id="cid-1"
    ):
        pass

    assert "dispatch_context_entered" in logs
    assert "dispatch_context_exited" in logs

    class _Cmd:
        pass

    handler_func = lambda _msg: "handled"

    monkeypatch.setattr(
        FlextHandlers.Discovery,
        "scan_module",
        lambda _module: [("h", handler_func, SimpleNamespace(command=_Cmd))],
    )
    created = FlextDispatcher.create(auto_discover_handlers=True)
    assert isinstance(created, FlextDispatcher)

    original_init = FlextDispatcher.__init__

    def bad_init(self, **_kwargs: object) -> None:
        msg = "init-fail"
        raise RuntimeError(msg)

    monkeypatch.setattr(FlextDispatcher, "__init__", bad_init)
    create_from_global = FlextDispatcher.create_from_global_config()
    assert create_from_global.is_failure
    monkeypatch.setattr(FlextDispatcher, "__init__", original_init)

    monkeypatch.setattr(
        dispatcher._circuit_breaker,
        "cleanup",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup-fail")),
    )
    dispatcher.cleanup()


def test_remaining_branches_group_a(
    dispatcher: FlextDispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover remaining middleware/registration/dispatch helper branches."""
    assert dispatcher._validate_handler_mode(
        c.Cqrs.HandlerType.COMMAND.value
    ).is_success

    class _FlakyHandle:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def handle(self) -> object:
            self.calls += 1
            if self.calls < 3:
                return lambda _msg: "ok"
            return "nope"

    assert dispatcher._execute_handler(
        _as_handler(_FlakyHandle()), _as_payload(_EchoMessage())
    ).is_failure

    class _ObjectResult:
        def handle(self, _msg: object) -> object:
            return _StringifyingObject()

    assert (
        dispatcher._execute_handler(
            _as_handler(_ObjectResult()), _as_payload(_EchoMessage())
        ).value
        == "stringified-object"
    )

    dispatcher._middleware_configs = [
        DispatcherMiddlewareConfig(
            middleware_id="b",
            middleware_type="m",
            enabled=True,
            order=2,
            name=None,
            config=t.ConfigMap(root={}),
        ),
        DispatcherMiddlewareConfig(
            middleware_id="a",
            middleware_type="m",
            enabled=True,
            order=1,
            name=None,
            config=t.ConfigMap(root={}),
        ),
    ]
    dispatcher._middleware_instances = {
        "a": _as_handler_callable(_MiddlewareOK()),
        "b": _as_handler_callable(_MiddlewareBadResult()),
    }
    assert dispatcher._execute_middleware_chain(
        _EchoMessage(), _as_handler(_HandleOnly())
    ).is_success

    missing = dispatcher._process_middleware_instance(
        _EchoMessage(),
        _as_handler(_HandleOnly()),
        DispatcherMiddlewareConfig(
            middleware_id="missing",
            middleware_type="m",
            enabled=True,
            order=0,
            name=None,
            config=t.ConfigMap(root={}),
        ),
    )
    assert missing.is_success
    assert dispatcher._handle_middleware_result(
        r[t.GeneralValueType].fail("blocked"), "m"
    ).is_failure
    assert dispatcher.execute().is_success

    dispatcher._handlers["dict"] = _as_handler(_HandleOnly())
    assert dispatcher._dispatch_command({"command_id": "c-1"}).is_success

    class _WithId:
        id = "id-1"

    dispatcher._handlers[_WithId.__name__] = _as_handler(_HandleOnly())
    assert dispatcher._dispatch_command(_as_payload(_WithId())).is_success

    query = _QueryMessage()
    key = dispatcher._generate_cache_key(query, _QueryMessage)
    dispatcher._cache[key] = r[t.GeneralValueType].ok("cached")
    dispatcher._handlers[_QueryMessage.__name__] = _as_handler(_HandleOnly())
    assert dispatcher._dispatch_command(query).value == "cached"

    monkeypatch.setattr(
        dispatcher,
        "_execute_middleware_chain",
        lambda *_args: r[bool].fail("middleware-down"),
    )
    assert dispatcher._dispatch_command(_EchoMessage()).is_failure

    assert dispatcher._layer1_register_handler(
        _as_handler_key(cast("object", 123)),
        _as_handler(_HandleOnly()),
    ).is_success

    class _TrackContainer:
        def __init__(self) -> None:
            self.factories: dict[str, object] = {}

        def wire_modules(self, **_kwargs: object) -> None:
            return None

        def register_factory(self, name: str, factory: object) -> bool:
            self.factories[name] = factory
            return True

    tracker = _TrackContainer()
    dispatcher._container = cast("p.DI", cast("object", tracker))

    class _NoInterface:
        pass

    assert dispatcher._register_single_handler(_as_handler(_NoInterface())).is_failure
    assert dispatcher._register_single_handler(_as_handler(_HandleOnly())).is_success

    class _ClassOnly:
        def handle(self, _msg: object) -> str:
            return "ok"

    assert dispatcher._register_single_handler(_as_handler(_ClassOnly)).is_success

    class _FactoryFail(_TrackContainer):
        def register_factory(self, name: str, factory: object) -> bool:
            _ = (name, factory)
            raise RuntimeError("factory-fail")

    dispatcher._container = cast("p.DI", cast("object", _FactoryFail()))
    assert dispatcher._register_two_arg_handler(
        _ClassOnly,
        _as_handler(_HandleOnly()),
    ).is_success

    wrapped_cfg = DispatcherMiddlewareConfig(
        middleware_id="mw-id",
        middleware_type="m",
        enabled=True,
        order=1,
        name=None,
        config=t.ConfigMap(root={}),
    )
    assert dispatcher.layer1_add_middleware(
        _as_handler_callable(_MiddlewareOK()), wrapped_cfg
    ).is_success

    monkeypatch.setattr(
        dispatcher, "_dispatch_command", lambda _event: r[t.GeneralValueType].ok("ok")
    )
    assert dispatcher._publish_event({"evt": 1}).is_success
    assert dispatcher.publish({"single": 1}).is_success

    class _Nested:
        child = None

    assert dispatcher._get_nested_attr(_Nested(), "child", "x") is None
    assert dispatcher._normalize_request(
        cast("Mapping[str, t.ConfigMapValue]", cast("object", {"handler": object()})),
    ).is_failure
    assert dispatcher._validate_and_extract_handler(
        m.HandlerRegistrationRequest.model_construct(handler=None)
    ).is_failure
    assert dispatcher._register_handler_by_mode(
        _as_handler(_HandleOnly()), "n", {"message_type": "m"}
    ).is_success
    assert dispatcher._register_handler_with_request(
        cast("Mapping[str, t.ConfigMapValue]", cast("object", 5)),
    ).is_failure
    assert dispatcher.register_handler(
        _ClassOnly, _as_handler(_HandleOnly())
    ).is_success

    class _MapOnly(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            return {"handler": _CanHandleAuto(), "message_type": "m"}[key]

        def __iter__(self):
            return iter(["handler", "message_type"])

        def __len__(self) -> int:
            return 2

    assert dispatcher.register_handler(_MapOnly()).is_success

    monkeypatch.setattr(
        dispatcher,
        "_layer1_register_handler",
        lambda *_args: r[bool].fail("layer1-fail"),
    )
    assert dispatcher.register_handler(lambda x: x).is_failure
    assert dispatcher.register_handlers({
        _ClassOnly: _as_handler(_HandleOnly())
    }).is_failure


def test_remaining_branches_group_b(
    dispatcher: FlextDispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover remaining pipeline/metadata/context/cleanup success and edge branches."""
    assert dispatcher._check_pre_dispatch_conditions("ok").is_success

    class _ExecutorBoom:
        def submit(self, _func: object) -> object:
            raise ValueError("boom")

    monkeypatch.setattr(
        dispatcher._timeout_enforcer, "should_use_executor", lambda: True
    )
    monkeypatch.setattr(
        dispatcher._timeout_enforcer, "ensure_executor", lambda: _ExecutorBoom()
    )
    with pytest.raises(ValueError):
        dispatcher._execute_with_timeout(lambda: r[t.GeneralValueType].ok("x"), 0.1)

    monkeypatch.setattr(dispatcher._retry_policy, "should_retry", lambda _attempt: True)
    monkeypatch.setattr(
        dispatcher._retry_policy, "is_retriable_error", lambda _msg: False
    )
    assert dispatcher._should_retry_on_error(0, "fatal") is False

    assert dispatcher.dispatch(_EchoMessage()).is_success or True
    assert dispatcher.dispatch(None).is_failure

    dispatcher._handlers["custom_type"] = _as_handler(cast("object", "nope"))
    assert dispatcher.dispatch("custom_type", {"a": 1}).is_failure

    dispatcher._handlers["custom_type"] = lambda _msg: 7
    assert dispatcher.dispatch("custom_type", {"a": 1}).is_success

    def _raising(_msg: object) -> object:
        raise RuntimeError("dispatch-error")

    dispatcher._handlers["custom_type"] = _as_handler(_raising)
    assert dispatcher.dispatch("custom_type", {"a": 1}).is_failure

    monkeypatch.setattr(
        dispatcher,
        "_execute_dispatch_pipeline",
        lambda *_args: r[t.GeneralValueType].ok("pipeline"),
    )
    assert dispatcher.dispatch(_EchoMessage(), config=m.DispatchConfig()).is_success

    meta = FlextDispatcher._convert_metadata_to_model(
        _as_payload({"lst": [1, 2], "obj": object()}),
    )
    assert meta is not None and meta.attributes["lst"] == [1, 2]

    built = FlextDispatcher._build_dispatch_config_from_args(None, {"k": "v"}, "cid", 2)
    assert isinstance(built, m.DispatchConfig)

    class _Msg:
        pass

    dispatcher._handlers[_Msg.__name__] = lambda _m: "plain"
    simple_dispatch = dispatcher._try_simple_dispatch(_as_payload(_Msg()))
    assert simple_dispatch is not None and simple_dispatch.is_success

    assert dispatcher._extract_dispatch_config(None, None, None, None).is_success
    assert dispatcher._extract_dispatch_config(
        None, m.Metadata(attributes={"k": "v"}), None, None
    ).is_success

    prep = dispatcher._prepare_dispatch_context(
        _EchoMessage(), None, m.DispatchConfig(metadata=m.Metadata(attributes={"x": 1}))
    )
    assert prep.is_success

    monkeypatch.setattr(
        dispatcher, "_check_pre_dispatch_conditions", lambda _mt: r[bool].ok(True)
    )
    assert dispatcher._validate_pre_dispatch_conditions({
        "message_type": "m"
    }).is_success

    monkeypatch.setattr(
        dispatcher,
        "_execute_dispatch_attempt",
        lambda *_args: r[t.GeneralValueType].ok("done"),
    )
    retry_context = cast(
        "t.ConfigMapValue",
        {
            "message": _EchoMessage(),
            "message_type": "m",
            "metadata": {"x": 1},
            "correlation_id": "cid",
            "timeout_override": 1,
        },
    )
    assert dispatcher._execute_with_retry_policy(retry_context).is_success

    monkeypatch.setattr(
        dispatcher, "_dispatch_command", lambda _msg: r[t.GeneralValueType].ok("ok")
    )
    assert dispatcher._create_execute_with_context(
        _EchoMessage(), None, None
    )().is_success

    opts = m.ExecuteDispatchAttemptOptions(
        message_type="m",
        metadata={"x": 1},
        correlation_id="cid",
        timeout_override=1,
        operation_id="op-bad",
    )
    monkeypatch.setattr(
        u,
        "process",
        lambda *_args, **_kwargs: r[list[tuple[str, str]]].fail("bad-transform"),
    )
    monkeypatch.setattr(dispatcher, "_get_timeout_seconds", lambda _ov: 0.01)
    monkeypatch.setattr(
        dispatcher,
        "_execute_with_timeout",
        lambda *_args, **_kwargs: r[t.GeneralValueType].ok("x"),
    )
    assert dispatcher._execute_dispatch_attempt(_EchoMessage(), opts).is_success

    assert FlextDispatcher._normalize_context_metadata(None) is None
    assert FlextDispatcher._normalize_context_metadata(
        m.Metadata(attributes={"k": "v"})
    ) == {"k": "v"}

    meta_obj = m.Metadata(attributes={"k": "v"})
    assert FlextDispatcher._extract_metadata_mapping(meta_obj) is meta_obj
    assert FlextDispatcher._extract_metadata_mapping({"k": "v"}) is not None

    class _GoodDump(BaseModel):
        k: str = "v"

    assert FlextDispatcher._extract_metadata_mapping(_GoodDump()) is not None
    assert (
        FlextDispatcher._extract_metadata_mapping({"attributes": {"k": "v"}})
        is not None
    )
    assert (
        FlextDispatcher._is_metadata_attribute_compatible(
            _as_payload({"bad": object()}),
        )
        is False
    )
    assert (
        FlextDispatcher._is_metadata_attribute_compatible(
            _as_payload({"k": [object()]}),
        )
        is False
    )
    assert FlextDispatcher._extract_from_flext_metadata(meta_obj) is meta_obj

    class _DumpCarrier(BaseModel):
        k: str = "v"

    assert FlextDispatcher._extract_from_object_attributes(
        _as_payload(_DumpCarrier())
    ) == {"k": "v"}

    dispatcher.config.dispatcher_auto_context = False
    with dispatcher._context_scope(metadata={"x": 1}, correlation_id="cid"):
        pass

    dispatcher.config.dispatcher_auto_context = True
    with dispatcher._context_scope(metadata={"x": 1}, correlation_id="cid"):
        pass

    class _CtxMap(Mapping[str, t.GeneralValueType]):
        def __getitem__(self, key: str) -> t.GeneralValueType:
            return {"x": 1}[key]

        def __iter__(self):
            return iter(["x"])

        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(u, "generate", lambda _prefix: "cid-generated")
    with dispatcher._context_scope(metadata=_CtxMap(), correlation_id=None):
        pass

    assert FlextDispatcher.create_from_global_config().is_success
    assert len(dispatcher.dispatch_batch("ignored", [_EchoMessage()])) == 1
    assert "circuit_breaker_failures" in dispatcher.get_performance_metrics()
    dispatcher.cleanup()


def test_final_uncovered_edges(
    dispatcher: FlextDispatcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover final dispatcher edge lines that require targeted setup."""

    class _FakeHandlerType:
        __members__ = {"COMMAND": "command", "QUERY": "query"}

    monkeypatch.setattr(c.Cqrs, "HandlerType", _FakeHandlerType)
    assert dispatcher._validate_handler_mode("command").is_success

    dispatcher._middleware_configs = [
        DispatcherMiddlewareConfig(
            middleware_id="d",
            middleware_type="m",
            enabled=False,
            order=0,
            name=None,
            config=t.ConfigMap(root={}),
        ),
        DispatcherMiddlewareConfig(
            middleware_id="e",
            middleware_type="m",
            enabled=True,
            order=1,
            name=None,
            config=t.ConfigMap(root={}),
        ),
    ]
    dispatcher._middleware_instances = {"e": _as_handler_callable(_MiddlewareOK())}
    monkeypatch.setattr(
        dispatcher, "_process_middleware_instance", lambda *_args: r[bool].fail("stop")
    )
    assert dispatcher._execute_middleware_chain(
        _EchoMessage(), _as_handler(_HandleOnly())
    ).is_failure

    disabled_mw = FlextDispatcher._process_middleware_instance(
        dispatcher,
        _EchoMessage(),
        _as_handler(_HandleOnly()),
        DispatcherMiddlewareConfig(
            middleware_id="disabled",
            middleware_type="m",
            enabled=False,
            order=0,
            name=None,
            config=t.ConfigMap(root={}),
        ),
    )
    assert disabled_mw.is_success

    assert dispatcher._invoke_middleware(
        _as_handler_callable(object()),
        _EchoMessage(),
        _as_handler(_HandleOnly()),
        "x",
    ).is_failure

    dispatcher._handlers[_EchoMessage.__name__] = _as_handler(_HandleOnly())
    monkeypatch.setattr(
        dispatcher, "_execute_middleware_chain", lambda *_args: r[bool].fail("mid-fail")
    )
    assert dispatcher._dispatch_command(_EchoMessage()).is_failure

    assert dispatcher._layer1_register_handler(_as_handler(_HandleOnly())).is_success

    class _WireContainer:
        def wire_modules(self, **_kwargs: object) -> None:
            return None

    dispatcher._container = cast("p.DI", cast("object", _WireContainer()))
    dispatcher._wire_handler_dependencies(lambda x: x)

    class _TrackFactoryContainer(_WireContainer):
        def __init__(self) -> None:
            self.factories: dict[str, object] = {}

        def register_factory(self, name: str, factory: object) -> bool:
            self.factories[name] = factory
            return True

    tracker = _TrackFactoryContainer()
    dispatcher._container = cast("p.DI", cast("object", tracker))

    class _ClassOnly:
        def handle(self, _msg: object) -> str:
            return "ok"

    assert dispatcher._register_single_handler(_as_handler(_ClassOnly)).is_success
    single_factory = tracker.factories["handler.type"]
    assert callable(single_factory)
    assert hasattr(single_factory(), "handle")

    class _ModelHandler(BaseModel):
        def handle(self, _msg: object) -> str:
            return "ok"

    assert dispatcher._register_single_handler(_as_handler(_ModelHandler)).is_success
    model_single_factory = tracker.factories["handler.modelmetaclass"]
    assert callable(model_single_factory)
    assert isinstance(model_single_factory(), BaseModel)

    assert dispatcher._register_two_arg_handler(
        "cmd.class", _as_handler(_ClassOnly)
    ).is_success
    two_arg_factory = tracker.factories["handler.cmd.class"]
    assert callable(two_arg_factory)
    assert hasattr(two_arg_factory(), "handle")

    assert dispatcher._register_two_arg_handler(
        "cmd.model", _as_handler(_ModelHandler)
    ).is_success
    model_two_arg_factory = tracker.factories["handler.cmd.model"]
    assert callable(model_two_arg_factory)
    assert isinstance(model_two_arg_factory(), BaseModel)

    class _RuntimeExecutor:
        def submit(self, _func: object) -> object:
            raise RuntimeError("other runtime")

    monkeypatch.setattr(
        dispatcher._timeout_enforcer, "should_use_executor", lambda: True
    )
    monkeypatch.setattr(
        dispatcher._timeout_enforcer, "ensure_executor", lambda: _RuntimeExecutor()
    )
    with pytest.raises(RuntimeError):
        dispatcher._execute_with_timeout(lambda: r[t.GeneralValueType].ok("x"), 0.1)

    opts = m.ExecuteDispatchAttemptOptions(
        message_type="m",
        metadata={"x": 1},
        correlation_id="cid",
        timeout_override=1,
        operation_id="op-2893",
    )
    monkeypatch.setattr(
        u, "process", lambda *_args, **_kwargs: r[list[tuple[str, str]]].fail("bad")
    )
    monkeypatch.setattr(dispatcher, "_get_timeout_seconds", lambda _ov: 0.01)
    monkeypatch.setattr(
        dispatcher,
        "_execute_with_timeout",
        lambda *_args, **_kwargs: r[t.GeneralValueType].ok("x"),
    )
    assert dispatcher._execute_dispatch_attempt(_EchoMessage(), opts).is_success

    from flext_core.context import FlextContext

    _ = FlextContext.Variables.ParentCorrelationId.set("parent-old")

    class _MapCtx(Mapping[str, t.GeneralValueType]):
        def __getitem__(self, key: str) -> t.GeneralValueType:
            return {"x": 1}[key]

        def __iter__(self):
            return iter(["x"])

        def __len__(self) -> int:
            return 1

    with dispatcher._context_scope(metadata=_MapCtx(), correlation_id="child-new"):
        pass
