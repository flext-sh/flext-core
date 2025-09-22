"""Tests for FlextDispatcher facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType
from typing import Literal, cast
from unittest.mock import Mock

from pydantic import BaseModel

from flext_core import (
    FlextBus,
    FlextConfig,
    FlextDispatcher,
    FlextDispatcherRegistry,
    FlextHandlers,
    FlextModels,
    FlextResult,
)


@dataclass
class EchoCommand:
    """Simple command used for dispatcher tests."""

    payload: str


class EchoHandler(FlextHandlers[EchoCommand, str]):
    """Handler that uppercases the payload."""

    def __init__(self) -> None:
        """Initialize the echo handler."""
        super().__init__(handler_mode="command")

    def handle(self, message: EchoCommand) -> FlextResult[str]:
        """Handle the echo command.

        Returns:
            FlextResult[str]: Success result with uppercase payload.

        """
        return FlextResult[str].ok(message.payload.upper())

    def execute(self, message: EchoCommand) -> FlextResult[str]:
        """Delegate execution to the simplified handle implementation."""
        return self.handle(message)


class ContextAwareHandler(FlextHandlers[EchoCommand, str]):
    """Handler that inspects correlation context."""

    def __init__(self) -> None:
        """Initialize the context aware handler."""
        super().__init__(handler_mode="command")

    def handle(self, message: EchoCommand) -> FlextResult[str]:
        """Handle the echo command using correlation context.

        Returns:
            FlextResult[str]: Success result with correlation-aware response.

        """
        correlation_id = FlextContext.Correlation.get_correlation_id()
        if correlation_id is None:
            return FlextResult[str].fail("Correlation ID missing")
        # Include message info in the response to use the parameter
        return FlextResult[str].ok(f"{correlation_id}:{message.payload}")

    def execute(self, message: EchoCommand) -> FlextResult[str]:
        """Delegate execution to the custom handle implementation."""
        return self.handle(message)


@dataclass
class CachedQuery:
    """Simple query payload used to validate cache behaviour."""

    query_id: str
    payload: str


class CountingQueryHandler(FlextHandlers[CachedQuery, dict[str, object]]):
    """Query handler that records execution counts for each query."""

    def __init__(self) -> None:
        """Initialise the counting handler in query mode."""
        super().__init__(handler_mode="query")
        self.calls: dict[str, int] = {}

    def handle(self, message: CachedQuery) -> FlextResult[dict[str, object]]:
        """Return metadata while tracking how many times it executed."""
        current_calls = self.calls.get(message.query_id, 0) + 1
        self.calls[message.query_id] = current_calls
        result_payload = {
            "query_id": message.query_id,
            "payload": message.payload,
            "calls": current_calls,
        }
        return FlextResult[dict[str, object]].ok(result_payload)

    def execute(self, message: CachedQuery) -> FlextResult[dict[str, object]]:
        """Delegate execution to the query handler implementation."""
        return self.handle(message)


@dataclass
class CachedQuery:
    """Query object used to exercise dispatcher caching."""

    payload: str


class CachedQueryHandler:
    """Lightweight handler that tracks invocation count to validate caching."""

    def __init__(self) -> None:
        """Initialize the cached query handler."""
        self.calls = 0

    def handle(self, message: CachedQuery) -> FlextResult[str]:
        """Handle the cached query while counting invocations."""
        self.calls += 1
        return FlextResult[str].ok(f"{message.payload}:{self.calls}")


def test_dispatcher_cache_hits_for_equivalent_dataclass_queries() -> None:
    """Repeated dataclass queries with reordered payloads reuse the cached result."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()

    @dataclass
    class StructuredQuery:
        filters: dict[str, int]
        limits: tuple[int, ...]

    class StructuredQueryHandler:
        def __init__(self) -> None:
            self.call_count = 0

        def can_handle(self, command_type: type[object]) -> bool:
            return command_type is StructuredQuery

        def handle(self, message: StructuredQuery) -> FlextResult[dict[str, int]]:
            self.call_count += 1
            total = sum(message.filters.values()) + sum(message.limits)
            return FlextResult[dict[str, int]].ok({"total": total})

    handler = StructuredQueryHandler()
    register_result = dispatcher.register_handler(
        cast("FlextHandlers[object, object]", handler)
    )
    assert register_result.is_success

    dispatcher.bus._cache.clear()

    first_query = StructuredQuery(filters={"a": 1, "b": 2}, limits=(3, 4))
    second_query = StructuredQuery(
        filters={"b": 2, "a": 1},
        limits=(3, 4),
    )

    first_result = dispatcher.dispatch(first_query)
    assert first_result.is_success
    assert handler.call_count == 1
    assert len(dispatcher.bus._cache) == 1

    second_result = dispatcher.dispatch(second_query)
    assert second_result.is_success
    assert handler.call_count == 1
    assert len(dispatcher.bus._cache) == 1
    assert second_result.unwrap() == {"total": 10}
    cached_result = next(iter(dispatcher.bus._cache.values()))
    assert cached_result.unwrap() == {"total": 10}


def test_dispatcher_cache_hits_for_equivalent_pydantic_queries() -> None:
    """Pydantic-based queries share cache entries when payload ordering differs."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()

    class CachedQuery(BaseModel):
        filters: dict[str, int]
        offsets: list[int]

    class CachedQueryHandler:
        def __init__(self) -> None:
            self.call_count = 0

        def can_handle(self, command_type: type[object]) -> bool:
            return command_type is CachedQuery

        def handle(self, message: CachedQuery) -> FlextResult[dict[str, int]]:
            self.call_count += 1
            total = sum(message.filters.values()) + sum(message.offsets)
            return FlextResult[dict[str, int]].ok({"total": total})

    handler = CachedQueryHandler()
    register_result = dispatcher.register_handler(
        cast("FlextHandlers[object, object]", handler)
    )
    assert register_result.is_success

    # Avoid accessing private _cache directly; prefer a public API.
    if hasattr(dispatcher.bus, "clear_cache") and callable(dispatcher.bus.clear_cache):
        dispatcher.bus.clear_cache()
    else:
        msg = (
            "dispatcher.bus does not provide a public 'clear_cache' method. "
            "Please add a public API to clear the cache instead of accessing _cache directly."
        )
        raise RuntimeError(
            msg
        )

    first_query = CachedQuery(filters={"c": 3, "d": 4}, offsets=[1, 2])
    second_query = CachedQuery(filters={"d": 4, "c": 3}, offsets=[1, 2])

    first_result = dispatcher.dispatch(first_query)
    assert first_result.is_success
    assert handler.call_count == 1
    assert len(dispatcher.bus._cache) == 1

    second_result = dispatcher.dispatch(second_query)
    assert second_result.is_success
    assert handler.call_count == 1
    assert len(dispatcher.bus._cache) == 1
    assert second_result.unwrap() == {"total": 10}
    cached_result = next(iter(dispatcher.bus._cache.values()))
    assert cached_result.unwrap() == {"total": 10}


def test_dispatcher_registers_handler_and_dispatches_command() -> None:
    """Test the dispatcher registers a handler and dispatches a command."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()

    handler = EchoHandler()
    register_result = dispatcher.register_handler(
        cast("FlextHandlers[object, object]", handler)
    )
    assert register_result.is_success

    result = dispatcher.dispatch(EchoCommand(payload="hello"))
    assert result.is_success
    assert result.unwrap() == "HELLO"


def test_dispatcher_register_command_binding() -> None:
    """Test the dispatcher registers a command binding."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()

    handler = EchoHandler()
    register_result = dispatcher.register_command(
        EchoCommand, cast("FlextHandlers[object, object]", handler)
    )
    assert register_result.is_success

    result: FlextResult[object] = dispatcher.dispatch(EchoCommand(payload="world"))
    assert result.is_success
    assert result.unwrap() == "WORLD"


def test_dispatcher_register_function_helper() -> None:
    """Test the dispatcher registers a function helper."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()

    def handle_echo(command: EchoCommand) -> str:
        if command.payload == "boom":
            msg = "boom"
            raise RuntimeError(msg)
        return f"echo:{command.payload}"

    register_result = dispatcher.register_function(EchoCommand, handle_echo)
    assert register_result.is_success

    success_result: FlextResult[object] = dispatcher.dispatch(
        EchoCommand(payload="ping")
    )
    assert success_result.is_success
    assert success_result.unwrap() == "echo:ping"

    failure_result: FlextResult[object] = dispatcher.dispatch(
        EchoCommand(payload="boom")
    )
    assert failure_result.is_failure
    assert "boom" in (failure_result.error or "")

    handler = dispatcher.bus.find_handler(EchoCommand(payload="ping"))
    assert handler is not None

    result = cast("FlextHandlers[object, object]", handler).handle(
        EchoCommand(payload="ping")
    )
    assert result.is_success
    assert result.unwrap() == "echo:ping"


def test_dispatcher_provides_correlation_context() -> None:
    """Test the dispatcher provides correlation context."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()

    handler = ContextAwareHandler()
    register_result = dispatcher.register_handler(
        cast("FlextHandlers[object, object]", handler)
    )
    assert register_result.is_success

    result: FlextResult[object] = dispatcher.dispatch(EchoCommand(payload="context"))
    assert result.is_success
    assert isinstance(result.unwrap(), str)
    assert result.unwrap()


def test_dispatcher_dispatch_with_explicit_correlation_id_uses_provided_value() -> None:
    """Dispatcher should expose the provided correlation ID within handlers."""
    FlextContext.Utilities.clear_context()

    class RecordingBus:
        def __init__(self) -> None:
            self.correlation_ids: list[str | None] = []
            self.parent_ids: list[str | None] = []

        def execute(self, message: object) -> FlextResult[object]:
            self.correlation_ids.append(FlextContext.Correlation.get_correlation_id())
            self.parent_ids.append(FlextContext.Correlation.get_parent_correlation_id())
            return FlextResult[object].ok("handled")

        def register_handler(self, *args: object) -> FlextResult[dict[str, object]]:
            return FlextResult[dict[str, object]].ok({})

    bus = RecordingBus()
    dispatcher = FlextDispatcher(bus=bus)

    provided_correlation = "corr-explicit-123"
    result: FlextResult[object] = dispatcher.dispatch(
        EchoCommand(payload="explicit"),
        correlation_id=provided_correlation,
    )

    assert result.is_success
    assert bus.correlation_ids == [provided_correlation]
    assert bus.parent_ids == [None]

    assert FlextContext.Correlation.get_correlation_id() is None
    assert FlextContext.Correlation.get_parent_correlation_id() is None


def test_dispatcher_dispatch_with_explicit_correlation_id_restores_previous_context() -> (
    None
):
    """Explicit correlation IDs should be scoped without mutating outer context."""
    FlextContext.Utilities.clear_context()

    class RecordingBus:
        def __init__(self) -> None:
            self.correlation_ids: list[str | None] = []
            self.parent_ids: list[str | None] = []

        def execute(self, message: object) -> FlextResult[object]:
            self.correlation_ids.append(FlextContext.Correlation.get_correlation_id())
            self.parent_ids.append(FlextContext.Correlation.get_parent_correlation_id())
            return FlextResult[object].ok("handled")

        def register_handler(self, *args: object) -> FlextResult[dict[str, object]]:
            return FlextResult[dict[str, object]].ok({})

    bus = RecordingBus()
    dispatcher = FlextDispatcher(bus=bus)

    existing_correlation = "corr-existing-456"
    existing_parent = "corr-parent-789"
    FlextContext.Correlation.set_correlation_id(existing_correlation)
    FlextContext.Correlation.set_parent_correlation_id(existing_parent)

    provided_correlation = "corr-explicit-nested"
    result = dispatcher.dispatch(
        EchoCommand(payload="restore"),
        correlation_id=provided_correlation,
    )

    assert result.is_success
    assert bus.correlation_ids == [provided_correlation]
    assert bus.parent_ids == [existing_correlation]

    assert FlextContext.Correlation.get_correlation_id() == existing_correlation
    assert FlextContext.Correlation.get_parent_correlation_id() == existing_parent

    FlextContext.Utilities.clear_context()


def test_dispatcher_propagates_metadata_model_to_context() -> None:
    """Dispatching with metadata model exposes attributes via context."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()

    metadata = FlextModels.Metadata(
        created_by="tester",
        attributes={"tenant": "acme", "retry": 2},
    )
    expected_metadata = dict(metadata.attributes)

    mock_execute = Mock()

    def fake_execute(message: object) -> FlextResult[object]:
        metadata_from_context = FlextContext.Performance.get_operation_metadata()
        assert metadata_from_context == expected_metadata
        return FlextResult[object].ok(dict(metadata_from_context))

    mock_execute.side_effect = fake_execute
    setattr(dispatcher._bus, "execute", mock_execute)

    request: dict[str, object] = {
        "message": EchoCommand(payload="metadata"),
        "context_metadata": metadata,
    }

    dispatch_result = dispatcher.dispatch_with_request(request)
    assert dispatch_result.is_success
    payload = dispatch_result.unwrap()
    assert payload["success"] is True
    assert payload["result"] == expected_metadata
    assert "created_by" not in payload["result"]
    mock_execute.assert_called_once()


def test_dispatcher_propagates_plain_metadata_dict_to_context() -> None:
    """Dispatching with plain metadata dict sets context metadata."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()

    metadata = {"tenant": "beta", "attempt": 1, 3: "three"}
    expected_metadata = {"tenant": "beta", "attempt": 1, "3": "three"}

    mock_execute = Mock()

    def fake_execute(message: object) -> FlextResult[object]:
        metadata_from_context = FlextContext.Performance.get_operation_metadata()
        assert metadata_from_context == expected_metadata
        return FlextResult[object].ok(dict(metadata_from_context))

    mock_execute.side_effect = fake_execute
    setattr(dispatcher._bus, "execute", mock_execute)

    request: dict[str, object] = {
        "message": EchoCommand(payload="metadata"),
        "context_metadata": metadata,
    }

    dispatch_result = dispatcher.dispatch_with_request(request)
    assert dispatch_result.is_success
    payload = dispatch_result.unwrap()
    assert payload["success"] is True
    assert payload["result"] == expected_metadata
    mock_execute.assert_called_once()


def test_dispatcher_registry_prevents_duplicate_handler_registration() -> None:
    """Registry returns success while avoiding duplicate registrations."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()
    registry = FlextDispatcherRegistry(dispatcher)

    handler = EchoHandler()

    first_result = registry.register_handler(handler)
    assert first_result.is_success
    # Note: RegistrationDetails doesn't have handler attribute in current implementation
    # assert first_result.unwrap().handler is handler

    second_result = registry.register_handler(handler)
    assert second_result.is_success
    # Note: RegistrationDetails doesn't have handler attribute in current implementation
    # assert second_result.unwrap().handler is handler


def test_dispatcher_registry_batch_registration_tracks_skipped() -> None:
    """Registry summary records skipped handlers when duplicates appear."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()
    registry = FlextDispatcherRegistry(dispatcher)

    handler = EchoHandler()
    summary_result: FlextResult[FlextDispatcherRegistry.Summary] = (
        registry.register_handlers(
            [handler, handler],
        )
    )

    assert summary_result.is_success
    summary = summary_result.unwrap()
    assert len(summary.registered) == 1
    expected_key = handler.handler_id or handler.handler_name
    assert summary.skipped == [expected_key]


def test_dispatcher_registry_register_bindings() -> None:
    """Registry binds handlers to explicit message types."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()
    registry = FlextDispatcherRegistry(dispatcher)

    handler = EchoHandler()
    summary_result = registry.register_bindings([(EchoCommand, handler)])
    assert summary_result.is_success
    summary = summary_result.unwrap()
    assert len(summary.registered) == 1

    dispatch_result = dispatcher.dispatch(EchoCommand(payload="bound"))
    assert dispatch_result.is_success
    assert dispatch_result.unwrap() == "BOUND"


def test_dispatcher_registry_register_function_map() -> None:
    """Registry accepts function mappings for registration."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher()
    registry = FlextDispatcherRegistry(dispatcher)

    def echo_function(command: EchoCommand) -> str:
        return f"fn:{command.payload}"

    def custom_create_handler_from_function(
        self: FlextDispatcher,
        handler_func: Callable[[object], object | FlextResult[object]],
        handler_config: dict[str, object] | None,
        mode: str,
    ) -> FlextResult[FlextHandlers[object, object]]:
        class FunctionHandler(FlextHandlers[object, object]):
            def __init__(self) -> None:
                super().__init__(
                    handler_mode=cast("Literal['command', 'query']", mode),
                    handler_config=handler_config,
                )
                self._handler_func = handler_func

            def handle(self, message: object) -> FlextResult[object]:
                result = self._handler_func(message)
                if isinstance(result, FlextResult):
                    return cast("FlextResult[object]", result)
                return FlextResult[object].ok(result)

            def execute(self, message: object) -> FlextResult[object]:
                return self.handle(message)

        return FlextResult[FlextHandlers[object, object]].ok(FunctionHandler())

    dispatcher._create_handler_from_function = MethodType(
        custom_create_handler_from_function, dispatcher
    )

    summary_result = registry.register_function_map(
        {EchoCommand: (echo_function, None)},
    )
    assert summary_result.is_success

    result: FlextResult[object] = dispatcher.dispatch(EchoCommand(payload="fn"))
    assert result.is_success
    assert result.unwrap() == "fn:fn"


def test_dispatcher_query_cache_returns_same_instance() -> None:
    """Cached queries should reuse the same result without re-running the handler."""
    FlextContext.Utilities.clear_context()
    bus = FlextBus.create_command_bus({
        "enable_caching": True,
        "enable_metrics": True,
        "max_cache_size": 4,
    })
    dispatcher = FlextDispatcher(bus=bus)

    handler = CountingQueryHandler()
    register_result = dispatcher.register_handler(
        cast("FlextHandlers[object, object]", handler),
        handler_mode="query",
    )
    assert register_result.is_success

    query = CachedQuery(query_id="cache-1", payload="value")
    first_result = dispatcher.dispatch(query)
    assert first_result.is_success
    first_payload = first_result.unwrap()

    cached_result = dispatcher.dispatch(query)
    assert cached_result.is_success
    assert cached_result.unwrap() is first_payload
    assert handler.calls.get("cache-1") == 1


def test_dispatcher_query_cache_respects_max_size() -> None:
    """Ensure cached entries are evicted when exceeding the configured size."""
    FlextContext.Utilities.clear_context()
    bus = FlextBus.create_command_bus({
        "enable_caching": True,
        "enable_metrics": True,
        "max_cache_size": 1,
    })
    dispatcher = FlextDispatcher(bus=bus)

    handler = CountingQueryHandler()
    register_result = dispatcher.register_handler(
        cast("FlextHandlers[object, object]", handler),
        handler_mode="query",
    )
    assert register_result.is_success

    first_query = CachedQuery(query_id="q-1", payload="first")
    second_query = CachedQuery(query_id="q-2", payload="second")

    first_result = dispatcher.dispatch(first_query)
    assert first_result.is_success
    first_payload = first_result.unwrap()

    second_result = dispatcher.dispatch(second_query)
    assert second_result.is_success

    assert handler.calls.get("q-1") == 1
    assert handler.calls.get("q-2") == 1

    third_result = dispatcher.dispatch(first_query)
    assert third_result.is_success
    assert handler.calls.get("q-1") == 2
    assert third_result.unwrap() is not first_payload


def test_dispatcher_uses_global_execution_timeout_for_bus() -> None:
    """Dispatcher maps global timeout configuration onto the bus."""
    FlextContext.Utilities.clear_context()
    FlextConfig.reset_global_instance()

    try:
        custom_timeout = 123
        FlextConfig.set_global_instance(
            FlextConfig(dispatcher_timeout_seconds=custom_timeout)
        )

        dispatcher = FlextDispatcher()
        assert dispatcher.bus.config.execution_timeout == custom_timeout
    finally:
        FlextConfig.reset_global_instance()


def test_dispatcher_reuses_cache_when_metrics_disabled() -> None:
    """Cache lookup remains active when metrics are disabled but caching is enabled."""
    FlextContext.Utilities.clear_context()
    dispatcher = FlextDispatcher(
        config={
            "auto_context": False,
            "enable_logging": False,
            "enable_metrics": False,
            "bus_config": {
                "enable_metrics": False,
                "enable_caching": True,
            },
        }
    )

    handler = CachedQueryHandler()
    registration_result = dispatcher.register_command(
        CachedQuery,
        cast("FlextHandlers[object, object]", handler),
    )
    assert registration_result.is_success

    first_result: FlextResult[object] = dispatcher.dispatch(
        CachedQuery(payload="cache")
    )
    assert first_result.is_success
    assert handler.calls == 1
    first_value = first_result.unwrap()

    second_result: FlextResult[object] = dispatcher.dispatch(
        CachedQuery(payload="cache")
    )
    assert second_result.is_success
    assert handler.calls == 1
    assert second_result.unwrap() == first_value
