"""Example 11 service models."""

from __future__ import annotations

from typing import override

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextSettings, m, r, t


class Ex11Payload(BaseModel):
    """Service payload model."""

    text: str
    count: int


class Ex11HandlerLike(BaseModel):
    """Handler-like protocol stub."""

    model_config = ConfigDict(frozen=False)
    data: m.ConfigMap = Field(default_factory=lambda: m.ConfigMap(root={}))

    def handle(self) -> str:
        """Handle payload."""
        return "ok"


class Ex11EntityStub(BaseModel):
    """Entity stub for protocol checks."""

    unique_id: str


class Ex11ProcessorProtocolGood(BaseModel):
    """Processor protocol-compliant stub."""

    model_config = ConfigDict(frozen=False)
    status: str = "ok"

    def process(self) -> str:
        """Process successfully."""
        return "ok"


class Ex11ProcessorProtocolBad(BaseModel):
    """Processor protocol non-compliant stub."""

    model_config = ConfigDict(frozen=False)
    status: str = "bad"


class Ex11CommandBusStub(BaseModel):
    """Command bus stub for service tests."""

    model_config = ConfigDict(frozen=False)

    def dispatch(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        """Dispatch message."""
        return r[t.ContainerValue].ok(message)

    def publish(self, _event: t.ContainerValue) -> None:
        """Publish event."""
        return

    def register_handler(self, _handler: t.ContainerValue) -> r[bool]:
        """Register handler."""
        return r[bool].ok(True)


class Ex11HandlerLikeService(FlextSettings):
    """Service-like handler stub."""

    @classmethod
    @override
    def validate(cls, value: t.ContainerValue) -> Ex11HandlerLikeService:
        """Validate service payload."""
        return cls.model_validate(value)

    def can_handle(self, message_type: type) -> bool:
        """Check whether message type is handled."""
        return bool(message_type)

    def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        """Handle service message."""
        return r[t.ContainerValue].ok(message)
