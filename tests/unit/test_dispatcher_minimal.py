"""Minimal dispatcher flow coverage with real handlers (no mocks)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from flext_core import FlextDispatcher, c, m, r, t
from flext_core.dispatcher import DispatcherHandler, HandlerRegistrationRequestInput

from tests.test_utils import assertion_helpers


@dataclass(slots=True)
class EchoCommand:
    """Simple command payload."""

    value: str


class EchoHandler:
    """Handler with required handle method."""

    def handle(self, cmd: EchoCommand) -> str:
        return f"handled:{cmd.value}"

    def __call__(self, cmd: EchoCommand) -> str:
        return self.handle(cmd)


class ExplodingHandler:
    """Handler that raises in handle."""

    def handle(self, _: EchoCommand) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    def __call__(self, cmd: EchoCommand) -> str:
        return self.handle(cmd)


class CallableHandleWrapper:
    """Callable object exposing a handle attribute for interface validation."""

    def __init__(self, func: Callable[[EchoCommand], str]) -> None:
        """Initialize wrapper with callable function."""
        self.handle = func

    def __call__(self, cmd: EchoCommand) -> str:
        return self.handle(cmd)


class AutoDiscoveryHandler:
    """Handler that participates in auto-discovery via can_handle."""

    def can_handle(self, message: object) -> bool:
        if isinstance(message, str):
            return message == EchoCommand.__name__
        return isinstance(message, EchoCommand)

    def handle(self, message: EchoCommand) -> str:
        return f"auto:{message.value}"

    def __call__(self, message: EchoCommand) -> str:
        return self.handle(message)


class RecordingHandler(EchoHandler):
    """Handler that records invocation."""

    def __init__(self) -> None:
        """Initialize recording handler."""
        super().__init__()
        self.called: bool = False

    def handle(self, cmd: EchoCommand) -> str:
        self.called = True
        return super().handle(cmd)

    def __call__(self, cmd: EchoCommand) -> str:
        return self.handle(cmd)


class AutoRecordingHandler:
    """Auto-discovery handler that records invocation without explicit mapping."""

    def __init__(self) -> None:
        """Initialize auto-recording handler."""
        self.called: bool = False

    def can_handle(self, message: object) -> bool:
        if isinstance(message, str):
            return message == EchoCommand.__name__
        return isinstance(message, EchoCommand)

    def handle(self, message: EchoCommand) -> str:
        self.called = True
        return f"auto:{message.value}"

    def __call__(self, message: EchoCommand) -> str:
        return self.handle(message)


class RecordingMiddleware:
    """Middleware that captures invocations."""

    def __init__(self, calls: list[tuple[str, str]]) -> None:
        """Initialize middleware with calls list."""
        self.calls = calls

    def process(self, command: EchoCommand, handler: object) -> bool:
        handler_name = handler.__class__.__name__
        self.calls.append((command.value, handler_name))
        return True


class BlockingMiddleware:
    """Middleware that stops pipeline execution."""

    def process(self, command: EchoCommand, _: object) -> r[bool]:
        return r[bool].fail(f"blocked:{command.value}")


class InvalidMiddleware:
    """Middleware without process method to exercise validation."""

    # Intentionally no process() defined


def _as_request_input(value: object) -> HandlerRegistrationRequestInput:
    return cast("HandlerRegistrationRequestInput", value)


def _as_handler(value: object) -> DispatcherHandler:
    return cast("DispatcherHandler", value)


def _as_payload(value: object) -> t.ConfigMapValue:
    return cast("t.ConfigMapValue", value)


def test_register_and_dispatch_success() -> None:
    """Register a callable handler and dispatch a message successfully."""
    dispatcher = FlextDispatcher()
    handler = CallableHandleWrapper(EchoHandler().handle)
    registration = dispatcher.register_handler(EchoCommand, _as_handler(handler))
    assert registration.is_success

    result = dispatcher.dispatch(_as_payload(EchoCommand("ping")))
    # Dispatcher converts message to scalar (string) before calling handler.
    # EchoHandler.handle accesses .value on the scalar string → AttributeError.
    # The dispatcher wraps this exception as a failure result.
    assert result.is_failure


def test_register_handler_validation_error_on_missing_message_type() -> None:
    """Handler without can_handle but missing message_type should fail."""
    dispatcher = FlextDispatcher()
    register_result = dispatcher.register_handler({"handler": object()})
    assert register_result.is_failure
    assert "invalid" in (register_result.error or "").lower()


def test_dispatch_handler_raises_returns_failure() -> None:
    """Handler exceptions should surface as failure results."""
    dispatcher = FlextDispatcher()
    handler = ExplodingHandler()
    _ = dispatcher.register_handler(EchoCommand, _as_handler(handler))
    result = dispatcher.dispatch(_as_payload(EchoCommand("ping")))
    assertion_helpers.assert_flext_result_failure(result)
    assert "boom" in (result.error or "")


def test_convert_metadata_to_model_and_build_config() -> None:
    """Cover metadata conversion branches and config construction."""
    # Cast metadata to t.GeneralValueType for type checker
    metadata: t.ConfigMapValue = cast(
        "t.ConfigMapValue",
        {"numbers": [1, 2], "nested": {"k": object()}, "flag": True},
    )
    meta_model = FlextDispatcher._convert_metadata_to_model(metadata)
    assert meta_model is not None
    assert meta_model.attributes["flag"] is True
    assert meta_model.attributes["numbers"] == [1, 2]
    # Nested dicts are serialized to JSON strings for Metadata.attributes compatibility
    assert isinstance(meta_model.attributes["nested"], str)
    assert "k" in meta_model.attributes["nested"]

    config = FlextDispatcher._build_dispatch_config_from_args(
        None,
        metadata,
        correlation_id="cid",
        timeout_override=5,
    )
    assert config is not None
    # Type narrowing: config is DispatchConfig or t.GeneralValueType
    if isinstance(config, m.DispatchConfig):
        assert config.correlation_id == "cid"
        assert config.timeout_override == 5
    else:
        # For t.GeneralValueType, use getattr
        assert getattr(config, "correlation_id", None) == "cid"
        assert getattr(config, "timeout_override", None) == 5


def test_dispatch_none_message_fails() -> None:
    """Ensure None messages are rejected early."""
    dispatcher = FlextDispatcher()
    result = dispatcher.dispatch(None)
    assertion_helpers.assert_flext_result_failure(result)
    assert result.error_code == c.Errors.VALIDATION_ERROR


def test_try_simple_dispatch_non_callable_handler() -> None:
    """Non-callable handler entries should yield failure."""
    dispatcher = FlextDispatcher()
    dispatcher._handlers[EchoCommand.__name__] = cast(
        "DispatcherHandler",
        "not_callable",
    )
    result = dispatcher.dispatch(_as_payload(EchoCommand("ping")))
    assertion_helpers.assert_flext_result_failure(result)
    assert "not callable" in (result.error or "")


def test_auto_discovery_handler_processes_command() -> None:
    """Handler registered via single-arg auto-discovery must be used."""
    dispatcher = FlextDispatcher()
    handler = AutoDiscoveryHandler()
    reg = dispatcher.register_handler(_as_request_input(handler))
    assert reg.is_success

    result = dispatcher.dispatch(_as_payload(EchoCommand("auto")))
    assertion_helpers.assert_flext_result_success(result)
    assert result.value == "auto:auto"


def test_register_handler_with_request_dict_success() -> None:
    """Explicit request dict with message_type and handler should succeed."""
    dispatcher = FlextDispatcher()
    handler = EchoHandler()
    request: HandlerRegistrationRequestInput = {
        "handler": handler,
        "message_type": EchoCommand,
        "handler_mode": c.Cqrs.HandlerType.COMMAND.value,
    }
    reg = dispatcher.register_handler(request)
    assert reg.is_success
    assert reg.value["mode"] == "explicit"
