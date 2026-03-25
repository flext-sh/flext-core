"""Example 10 handlers models."""

from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict

from flext_core import m, r, t
from flext_core._models.domain_event import FlextModelsDomainEvent


class Ex10Message(m.Command):
    """Command message payload."""

    text: str


class Ex10DerivedMessage(Ex10Message):
    """Derived message used for covariance checks."""


class Ex10Entity(m.Value):
    """Entity-like value payload for runtime checks."""

    unique_id: str


class Ex10ProcessorGood(m.Value):
    """Good processor for protocol checks."""

    marker: str = "good"

    def process(self) -> str:
        """Process successfully."""
        return "ok"


class Ex10ProcessorBad(m.Value):
    """Bad processor for protocol checks."""

    marker: str = "bad"

    def process(self) -> str:
        """Process successfully despite bad protocol metadata."""
        return "ok"


class Ex10ContextPayload(m.Value):
    handler_name: str
    handler_mode: str


class Ex10ProtocolHandler(m.Value):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)

    def handle(self, message: m.Command) -> r[str]:
        return r[str].ok(str(message))

    def check_data(self, data: m.Value | None) -> r[bool]:
        return r[bool].ok(data is not None)


class Ex10ServiceStub(m.Value):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)

    @property
    def is_valid(self) -> bool:
        return True

    def execute(self) -> r[t.ConfigMap]:
        return r[t.ConfigMap].ok(t.ConfigMap(root={"ok": True}))

    def get_service_info(self) -> t.ConfigMap:
        return t.ConfigMap(root={"service": "stub"})

    def validate_business_rules(self) -> r[bool]:
        return r[bool].ok(True)


class Ex10CommandBusStub(m.Value):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)

    def dispatch(self, message: m.Command) -> r[str]:
        return r[str].ok(str(message))

    def publish(self, event: FlextModelsDomainEvent.Entry) -> None:
        del event

    def register_handler(self, _handler: m.Value) -> r[bool]:
        return r[bool].ok(True)
