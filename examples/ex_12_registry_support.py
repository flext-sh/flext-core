"""Support objects for the registry DSL example."""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples import m
from examples import t
from examples import u
from flext_core import h, r

if TYPE_CHECKING:
    from examples import p


class ProtocolHandler:
    """Protocol-like handler used by the registry example."""

    message_type: type[m.Command]

    def __init__(self, label: str, message_type: type[m.Command]) -> None:
        """Store the handler label and message type."""
        self._label = label
        self.message_type = message_type

    def can_handle(self, message_type: type[m.Command]) -> bool:
        """Return whether this handler supports the message type."""
        return message_type is self.message_type

    def handle(self, message: p.Routable) -> p.Result[t.Scalar]:
        """Handle a command and return a scalar result."""
        value = ""
        if isinstance(message, m.Examples.CommandA):
            value = message.value
        elif isinstance(message, m.Examples.CommandB):
            value = str(message.amount)
        return r[t.Scalar].ok(f"{self._label}:{value}")

    def __call__(self, message: p.Routable) -> p.Result[t.Scalar]:
        """Callable adapter for registry handler protocols expecting callables."""
        return self.handle(message)


def as_registry_handler(
    handler: ProtocolHandler,
) -> t.DispatchableHandler:
    """Adapt protocol handlers to the registry callable contract."""

    class _RegistryHandlerCallable:
        handler_id: str
        message_type: type[m.Command]

        def __init__(self, source: ProtocolHandler) -> None:
            self._source = source
            self.handler_id = source.message_type.__name__
            self.message_type = source.message_type

        def __call__(self, message: p.Routable) -> t.JsonPayload:
            return u.normalize_to_container(self._source.handle(message).unwrap_or(""))

    return _RegistryHandlerCallable(handler)


@h.handler(m.Examples.CommandA, priority=3)
def discovered_handler(message: m.Command) -> m.Command:
    return message
