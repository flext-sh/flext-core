"""Support objects for the registry DSL example."""

from __future__ import annotations

from examples.models import m
from examples.protocols import p
from examples.typings import t
from examples.utilities import u
from flext_core import h, r


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

    def call(message: p.Routable) -> t.JsonPayload:
        return u.normalize_to_container(handler.handle(message).unwrap_or(""))

    handler_name = handler.message_type.__name__
    setattr(call, "__name__", handler_name)
    setattr(call, "handler_id", handler_name)
    setattr(call, "message_type", handler.message_type)
    return call


@h.handler(m.Examples.CommandA, priority=3)
def discovered_handler(message: m.Command) -> m.Command:
    return message
