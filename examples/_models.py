"""Auto-generated centralized models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from flext_core import m, r


class _ProtocolHandler(BaseModel):
    model_config = ConfigDict(frozen=False)

    def handle(self, message: BaseModel) -> r[str]:
        return r[str].ok(str(message))

    def check_data(self, data: BaseModel | None) -> r[bool]:
        return r[bool].ok(data is not None)


class _ServiceStub(BaseModel):
    model_config = ConfigDict(frozen=False)

    @property
    def is_valid(self) -> bool:
        return True

    def execute(self) -> r[m.ConfigMap]:
        return r[m.ConfigMap].ok(m.ConfigMap(root={"ok": True}))

    def get_service_info(self) -> m.ConfigMap:
        return m.ConfigMap(root={"service": "stub"})

    def validate_business_rules(self) -> r[bool]:
        return r[bool].ok(True)


class _CommandBusStub(BaseModel):
    """Minimal BaseModel stub satisfying is_command_bus duck-typing."""

    model_config = ConfigDict(frozen=False)

    def dispatch(self, message: BaseModel) -> r[str]:
        return r[str].ok(str(message))

    def publish(self, event: BaseModel) -> None:
        pass

    def register_handler(self, _handler: BaseModel) -> r[bool]:
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


class _ProcessorProtocolBad(BaseModel):
    model_config = ConfigDict(frozen=False)
    status: str = "bad"


def get_example_instances() -> tuple[
    _ProtocolHandler,
    _ServiceStub,
    _CommandBusStub,
    _Payload,
    _HandlerLike,
    _EntityStub,
    _ProcessorProtocolGood,
    _ProcessorProtocolBad,
]:
    """Return one instance of each example model for tests/docs that need them."""
    return (
        _ProtocolHandler(),
        _ServiceStub(),
        _CommandBusStub(),
        _Payload(text="example", count=1),
        _HandlerLike(),
        _EntityStub(unique_id="example-entity"),
        _ProcessorProtocolGood(),
        _ProcessorProtocolBad(),
    )
