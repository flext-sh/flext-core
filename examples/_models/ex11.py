"""Example 11 service models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, override

from examples import p, t
from flext_core import FlextSettings, m, r, u


class Ex11HandlerLikeService(FlextSettings):
    """Service-like handler stub."""

    @classmethod
    @override
    def validate(cls, value: t.ConfigMap) -> Ex11HandlerLikeService:
        """Validate service payload."""
        return cls.model_validate(value)

    def can_handle(self, message_type: type) -> bool:
        """Check whether message type is handled."""
        return bool(message_type)

    def handle(self, message: m.Command) -> p.Result[str]:
        """Handle service message."""
        return r[str].ok(str(message))


class Ex11Payload(m.Value):
    text: str
    count: int


class Ex11HandlerLike(m.Value):
    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)
    data: t.ConfigMap = u.Field(default_factory=lambda: t.ConfigMap(root={}))

    def handle(self) -> str:
        return "ok"


class Ex11EntityStub(m.Value):
    unique_id: str


class Ex11ProcessorProtocolGood(m.Value):
    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)
    status: str = "ok"

    def process(self) -> str:
        return "ok"


class Ex11ProcessorProtocolBad(m.Value):
    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)
    status: str = "bad"


class Ex11CommandBusStub(m.Value):
    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

    def dispatch(self, message: p.Routable) -> p.Result[t.RuntimeAtomic]:
        return r[t.RuntimeAtomic].ok(str(message))

    def publish(self, _event: p.Routable | Sequence[p.Routable]) -> p.Result[bool]:
        return r[bool].ok(True)

    def register_handler(
        self,
        _handler: t.HandlerLike,
        *,
        is_event: bool = False,
    ) -> p.Result[bool]:
        del is_event
        return r[bool].ok(True)
