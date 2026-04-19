"""Example models for ex10."""

from __future__ import annotations

from collections.abc import Callable

from examples import m, p, r, t


class Ex10Message(m.Command):
    text: str


class Ex10DerivedMessage(Ex10Message):
    pass


class Ex10ContextPayload(m.Value):
    text: str


class Ex10Entity(m.Value):
    unique_id: str


class Ex10ProcessorGood(m.Value):
    marker: str = "good"

    def process(self) -> bool:
        return True


class Ex10ProcessorBad(m.Value):
    marker: str = "bad"


class Ex10ProtocolHandler(m.BaseModel):
    message_type: type = Ex10Message

    def handle(self, message: Ex10Message) -> p.Result[str]:
        return r[str].ok(message.text)


class Ex10CommandBusStub(m.BaseModel):
    def dispatch(self, message: Ex10Message) -> p.Result[str]:
        return r[str].ok(message.text)


class Ex10ServiceStub(m.BaseModel):
    run: Callable[[], t.Container] | None = None


class ExamplesFlextCoreModelsEx10(m):
    """Examples namespace wrapper for ex10 models."""
