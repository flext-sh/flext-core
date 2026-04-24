"""Behavior tests for dispatcher materialization through the public DSL."""

from __future__ import annotations

from tests import m, p, t, u


class TestDispatcherDI:
    """Test dispatcher materialization without touching internals."""

    def test_dispatcher_builder_returns_protocol_aligned_dispatcher(self) -> None:
        """Dispatcher DSL yields a usable p.Dispatcher instance."""
        dispatcher = u.build_dispatcher()
        assert isinstance(dispatcher, p.Dispatcher)
        handle: t.DispatchableHandler = lambda _m: "handled"  # noqa: E731
        handle.message_type = m.Command
        assert dispatcher.register_handler(handle).success
