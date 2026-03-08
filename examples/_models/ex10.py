"""Example 10 handlers models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from flext_core import m, r, t


class Ex10ContextPayload(BaseModel):
    """Context payload used to build execution context."""

    handler_name: str
    handler_mode: str


class Ex10Message(m.Command):
    """Command message payload."""

    text: str


class Ex10DerivedMessage(Ex10Message):
    """Derived message used for covariance checks."""


class Ex10Entity(m.Value):
    """Entity-like value payload for runtime checks."""

    unique_id: str


class Ex10ProtocolHandler(BaseModel):
    """Handler stub implementing protocol operations."""

    model_config = ConfigDict(frozen=False)

    def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        """Echo handled message."""
        return r[t.ContainerValue].ok(message)

    def check_data(self, data: t.ContainerValue) -> r[bool]:
        """Check data is present."""
        return r[bool].ok(data is not None)


class Ex10ServiceStub(BaseModel):
    """Service protocol stub."""

    model_config = ConfigDict(frozen=False)

    @property
    def is_valid(self) -> bool:
        """Return service validity state."""
        return True

    def execute(self) -> r[t.ContainerValue]:
        """Execute service action."""
        return r[t.ContainerValue].ok(m.ConfigMap(root={"ok": True}))

    def get_service_info(self) -> m.ConfigMap:
        """Return service metadata."""
        return m.ConfigMap(root={"service": "stub"})

    def validate_business_rules(self) -> r[bool]:
        """Validate business rules."""
        return r[bool].ok(True)


class Ex10CommandBusStub(BaseModel):
    """Command bus protocol stub."""

    model_config = ConfigDict(frozen=False)

    def dispatch(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        """Dispatch command message."""
        return r[t.ContainerValue].ok(message)

    def publish(self, event: t.ContainerValue) -> None:
        """Publish event message."""
        del event

    def register_handler(self, _handler: t.ContainerValue) -> r[bool]:
        """Register a command handler."""
        return r[bool].ok(True)


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
