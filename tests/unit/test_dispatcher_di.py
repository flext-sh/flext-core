"""Behavior tests for dispatcher materialization through the public DSL."""

from __future__ import annotations

from tests import p, u


class TestDispatcherDI:
    """Test dispatcher materialization without touching internals."""

    def test_dispatcher_builder_returns_protocol_aligned_dispatcher(self) -> None:
        """The dispatcher DSL returns a usable dispatcher protocol instance."""
        dispatcher = u.build_dispatcher()
        assert isinstance(dispatcher, p.Dispatcher)
        assert dispatcher.register_handler(lambda _message: "handled").success
