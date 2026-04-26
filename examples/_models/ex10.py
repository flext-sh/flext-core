"""Example models for ex10."""

from __future__ import annotations

from collections.abc import (
    Callable,
)

from flext_core import m, p, r, t, u


class Ex10Message(m.Command):
    text: str = u.Field(description="Message text content")


class Ex10DerivedMessage(Ex10Message):
    pass


class Ex10ContextPayload(m.Value):
    text: str = u.Field(description="Text payload for context")


class Ex10Entity(m.Value):
    unique_id: str = u.Field(description="Unique identifier for entity")


class Ex10ProcessorGood(m.Value):
    marker: str = u.Field(default="good", description="Marker indicating successful processing")

    def process(self) -> bool:
        return True


class Ex10ProcessorBad(m.Value):
    marker: str = u.Field(default="bad", description="Marker indicating failed processing")


class Ex10ProtocolHandler(m.BaseModel):
    message_type: type = u.Field(default=Ex10Message, description="Message type for protocol handler")

    def handle(self, message: Ex10Message) -> p.Result[str]:
        return r[str].ok(message.text)


class Ex10CommandBusStub(m.BaseModel):
    def dispatch(self, message: Ex10Message) -> p.Result[str]:
        return r[str].ok(message.text)


class Ex10ServiceStub(m.BaseModel):
    run: Callable[[], t.JsonValue] | None = u.Field(default=None, description="Callable returning JSON value or None")


class ExamplesFlextCoreModelsEx10(m):
    """Examples namespace wrapper for ex10 models."""
