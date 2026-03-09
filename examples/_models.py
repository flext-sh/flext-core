"""Auto-generated centralized models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from flext_core import m, r, t


class _ProtocolHandler(BaseModel):
    model_config = ConfigDict(frozen=False)

    @staticmethod
    def _protocol_name() -> str:
        return "ProtocolHandler"

    def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        return r[t.ContainerValue].ok(message)

    def check_data(self, data: t.ContainerValue) -> r[bool]:
        return r[bool].ok(data is not None)


class _ServiceStub(BaseModel):
    model_config = ConfigDict(frozen=False)

    @property
    def is_valid(self) -> bool:
        return True

    @staticmethod
    def _protocol_name() -> str:
        return "ServiceStub"

    def execute(self) -> r[t.ContainerValue]:
        return r[t.ContainerValue].ok(m.ConfigMap(root={"ok": True}))

    def get_service_info(self) -> m.ConfigMap:
        return m.ConfigMap(root={"service": "stub"})

    def validate_business_rules(self) -> r[bool]:
        return r[bool].ok(True)


class _CommandBusStub(BaseModel):
    """Minimal BaseModel stub satisfying is_command_bus duck-typing."""

    model_config = ConfigDict(frozen=False)

    def dispatch(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        return r[t.ContainerValue].ok(message)

    def publish(self, event: t.ContainerValue) -> None:
        pass

    def register_handler(self, _handler: t.ContainerValue) -> r[bool]:
        return r[bool].ok(True)


class _Payload(BaseModel):
    text: str
    count: int


class _HandlerLike(BaseModel):
    """Minimal handler-like BaseModel for protocol checks."""

    model_config = ConfigDict(frozen=False)
    data: dict[str, str] = {}

    def handle(self) -> str:
        return "ok"


class _EntityStub(BaseModel):
    unique_id: str


class _ProcessorProtocolGood(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "ok"

    def process(self) -> str:
        return "ok"

    def _protocol_name(self) -> str:
        return "ProcessorProtocolGood"


class _ProcessorProtocolBad(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "bad"

    def _protocol_name(self) -> str:
        return "ProcessorProtocolBad"
