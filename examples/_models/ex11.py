"""Example 11 service models."""

from __future__ import annotations

from typing import override

from pydantic import ConfigDict, Field

from flext_core import FlextSettings, m, r


class Ex11HandlerLikeService(FlextSettings):
    """Service-like handler stub."""

    @classmethod
    @override
    def validate(cls, value: object) -> Ex11HandlerLikeService:
        """Validate service payload."""
        return cls(value)

    def can_handle(self, message_type: type) -> bool:
        """Check whether message type is handled."""
        return bool(message_type)

    def handle(self, message: object) -> r[str]:
        """Handle service message."""
        return r[str].ok(str(message))


class Ex11Payload(m.Value):
    text: str
    count: int


class Ex11HandlerLike(m.Value):
    model_config = ConfigDict(frozen=False)
    data: m.ConfigMap = Field(default_factory=lambda: m.ConfigMap(root={}))

    def handle(self) -> str:
        return "ok"


class Ex11EntityStub(m.Value):
    unique_id: str


class Ex11ProcessorProtocolGood(m.Value):
    model_config = ConfigDict(frozen=False)
    status: str = "ok"

    def process(self) -> str:
        return "ok"


class Ex11ProcessorProtocolBad(m.Value):
    model_config = ConfigDict(frozen=False)
    status: str = "bad"


class Ex11CommandBusStub(m.Value):
    model_config = ConfigDict(frozen=False)

    def dispatch(self, message: object) -> r[str]:
        return r[str].ok(str(message))

    def publish(self, _event: object) -> None:
        return

    def register_handler(self, _handler: object) -> r[bool]:
        return r[bool].ok(True)
