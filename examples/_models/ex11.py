"""Example models for ex11."""

from __future__ import annotations

from typing import Annotated

from flext_core import FlextSettings, m, p, r


class ExamplesFlextModelsEx11:
    """Examples namespace wrapper for ex11 models."""

    class Payload(m.Value):
        text: Annotated[str, m.Field(description="Payload text content")]

    class ServiceHandlerConfig(FlextSettings):
        enabled: Annotated[
            bool, m.Field(description="Whether the service is enabled")
        ] = True

    class ServiceHandlerLike(m.BaseModel):
        message_type: Annotated[
            type[m.Value], m.Field(description="Message type handled by this handler")
        ] = m.Value

        def handle(self, message: ExamplesFlextModelsEx11.Payload) -> p.Result[str]:
            return r[str].ok(message.text)

    class ProcessorProtocolGood(m.Value):
        status: Annotated[str, m.Field(description="Processing outcome status")] = "ok"

    class ProcessorProtocolBad(m.Value):
        status: Annotated[str, m.Field(description="Processing failure status")] = "bad"
