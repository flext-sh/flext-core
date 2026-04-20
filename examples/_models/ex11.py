"""Example models for ex11."""

from __future__ import annotations

from flext_core import FlextSettings, m, p, r


class Ex11Payload(m.Value):
    text: str


class Ex11EntityStub(m.Value):
    unique_id: str


class Ex11HandlerLikeService(FlextSettings):
    enabled: bool = True


class Ex11HandlerLike(m.BaseModel):
    message_type: type = Ex11Payload

    def handle(self, message: Ex11Payload) -> p.Result[str]:
        return r[str].ok(message.text)


class Ex11ProcessorProtocolGood(m.Value):
    status: str = "ok"


class Ex11ProcessorProtocolBad(m.Value):
    status: str = "bad"


class Ex11CommandBusStub(m.BaseModel):
    pass


class ExamplesFlextCoreModelsEx11(m):
    """Examples namespace wrapper for ex11 models."""
