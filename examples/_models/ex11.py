"""Example models for ex11."""

from __future__ import annotations

from typing import Annotated

from flext_core import FlextSettings, m, p, r, u


class ExamplesFlextCoreModelsEx11(m):
    """Examples namespace wrapper for ex11 models."""

    class Payload(m.Value):
        text: Annotated[str, u.Field(description="Payload text content")]

    class EntityStub(m.Value):
        unique_id: Annotated[str, u.Field(description="Unique entity identifier")]

    class HandlerLikeService(FlextSettings):
        enabled: Annotated[
            bool, u.Field(description="Whether the service is enabled")
        ] = True

    class HandlerLike(m.BaseModel):
        message_type: Annotated[
            type,
            u.Field(description="Message type handled by this handler"),
        ] = Ex11Payload

        def handle(self, message: Ex11Payload) -> p.Result[str]:
            return r[str].ok(message.text)

    class ProcessorProtocolGood(m.Value):
        status: Annotated[str, u.Field(description="Processing outcome status")] = "ok"

    class ProcessorProtocolBad(m.Value):
        status: Annotated[str, u.Field(description="Processing failure status")] = "bad"

    class CommandBusStub(m.BaseModel):
        pass
