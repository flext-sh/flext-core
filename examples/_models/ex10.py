"""Example models for ex10."""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import Annotated

from flext_core import m, p, r, t, u


class Ex10Message(m.Command):
    text: Annotated[str, u.Field(description="Message text content")]


class Ex10DerivedMessage(Ex10Message):
    pass


class Ex10ContextPayload(m.Value):
    text: Annotated[str, u.Field(description="Text payload for context")]


class Ex10Entity(m.Value):
    unique_id: Annotated[str, u.Field(description="Unique identifier for entity")]


class Ex10ProcessorGood(m.Value):
    marker: Annotated[
        str,
        u.Field(description="Marker indicating successful processing"),
    ] = "good"

    def process(self) -> bool:
        return True


class Ex10ProcessorBad(m.Value):
    marker: Annotated[
        str,
        u.Field(description="Marker indicating failed processing"),
    ] = "bad"


class Ex10ProtocolHandler(m.BaseModel):
    message_type: Annotated[
        type,
        u.Field(description="Message type for protocol handler"),
    ] = Ex10Message

    def handle(self, message: Ex10Message) -> p.Result[str]:
        return r[str].ok(message.text)


class Ex10CommandBusStub(m.BaseModel):
    def dispatch(self, message: Ex10Message) -> p.Result[str]:
        return r[str].ok(message.text)


class Ex10ServiceStub(m.BaseModel):
    run: Annotated[
        Callable[[], t.JsonValue] | None,
        u.Field(description="Callable returning JSON value or None"),
    ] = None


class ExamplesFlextCoreModelsEx10(m):
    """Examples namespace wrapper for ex10 models."""
