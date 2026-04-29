"""Example models for ex10."""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import Annotated

from flext_core import m, p, r, t, u


class ExamplesFlextModelsEx10:
    """Examples namespace wrapper for ex10 models."""

    class Message(m.Command):
        text: Annotated[str, u.Field(description="Message text content")]

    class DerivedMessage(Message):
        pass

    class ContextPayload(m.Value):
        text: Annotated[str, u.Field(description="Text payload for context")]

    class Entity(m.Value):
        unique_id: Annotated[str, u.Field(description="Unique identifier for entity")]

    class ProcessorGood(m.Value):
        marker: Annotated[
            str,
            u.Field(description="Marker indicating successful processing"),
        ] = "good"

        def process(self) -> bool:
            return True

    class ProcessorBad(m.Value):
        marker: Annotated[
            str,
            u.Field(description="Marker indicating failed processing"),
        ] = "bad"

    class ProtocolHandler(m.BaseModel):
        message_type: Annotated[
            type[m.Command],
            u.Field(description="Message type for protocol handler"),
        ] = m.Command

        def handle(
            self,
            message: ExamplesFlextModelsEx10.Message,
        ) -> p.Result[str]:
            return r[str].ok(message.text)

    class CommandBusStub(m.BaseModel):
        def dispatch(
            self,
            message: ExamplesFlextModelsEx10.Message,
        ) -> p.Result[str]:
            return r[str].ok(message.text)

    class ServiceStub(m.BaseModel):
        run: Annotated[
            Callable[[], t.JsonValue] | None,
            u.Field(description="Callable returning JSON value or None"),
        ] = None
