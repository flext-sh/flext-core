"""Behavior tests for dispatcher materialization through the public DSL."""

from __future__ import annotations

from tests import m, p, r, u


class TestDispatcherDI:
    """Test dispatcher materialization without touching internals."""

    class _Handler:
        message_type = "di_route"

        def handle(self, message: p.Routable) -> p.Result[str]:
            route = message.command_type or ""
            return r[str].ok(f"handled:{route}")

    def test_dispatcher_builder_returns_protocol_aligned_dispatcher(self) -> None:
        """The dispatcher DSL returns a usable dispatcher protocol instance."""
        dispatcher = u.build_dispatcher()
        assert isinstance(dispatcher, p.Dispatcher)
        assert dispatcher.register_handler(self._Handler()).success
        result = dispatcher.dispatch(
            m.Command(command_type="di_route", command_id="cmd-di"),
        )
        assert result.success
        assert result.value == "handled:di_route"
