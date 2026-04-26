"""Example models for ex11."""

from __future__ import annotations

from typing import Annotated

from flext_core import FlextSettings, m, p, r, u


class ExamplesFlextCoreModelsEx11:
    """Examples namespace wrapper for ex11 models."""

    class Payload(m.Value):
        text: Annotated[str, u.Field(description="Payload text content")]

    class EntityStub(m.Value):
        unique_id: Annotated[str, u.Field(description="Unique entity identifier")]

    class ServiceHandlerConfig(FlextSettings):
        enabled: Annotated[
            bool, u.Field(description="Whether the service is enabled")
        ] = True

    class ServiceHandlerLike(m.BaseModel):
        message_type: Annotated[
            type[m.Value],
            u.Field(description="Message type handled by this handler"),
        ] = m.Value

        def handle(
            self,
            message: ExamplesFlextCoreModelsEx11.Payload,
        ) -> p.Result[str]:
            return r[str].ok(message.text)

    class ProcessorProtocolGood(m.Value):
        status: Annotated[str, u.Field(description="Processing outcome status")] = "ok"

    class ProcessorProtocolBad(m.Value):
        status: Annotated[str, u.Field(description="Processing failure status")] = "bad"

    class ServiceCommandBusStub(m.BaseModel):
        pass
