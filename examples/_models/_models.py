"""Auto-generated centralized models."""

from __future__ import annotations

from typing import override

from pydantic import BaseModel, ConfigDict, Field

from flext_core import m, r, t


class Ex01ValidPersonPayload(BaseModel):
    """Valid payload for person conversion tests."""

    name: str
    age: int


class Ex01InvalidPersonPayload(BaseModel):
    """Invalid payload for person conversion tests."""

    name: str
    age: str


class Ex04CreateUser(BaseModel):
    """Create command payload."""

    username: str
    command_type: str = "create_user"
    query_type: str = ""
    event_type: str = ""


class Ex04GetUser(BaseModel):
    """Get query payload."""

    username: str
    command_type: str = ""
    query_type: str = "get_user"
    event_type: str = ""


class Ex04DeleteUser(BaseModel):
    """Delete command payload."""

    username: str
    command_type: str = "delete_user"
    query_type: str = ""
    event_type: str = ""


class Ex04FailingDelete(BaseModel):
    """Delete command that intentionally fails."""

    username: str
    command_type: str = "failing_delete"
    query_type: str = ""
    event_type: str = ""


class Ex04AutoCommand(BaseModel):
    """Auto command payload."""

    payload: str
    command_type: str = "auto_command"
    query_type: str = ""
    event_type: str = ""


class Ex04Ping(BaseModel):
    """Ping command payload."""

    value: str
    command_type: str = "ping"
    query_type: str = ""
    event_type: str = ""


class Ex04UnknownQuery(BaseModel):
    """Unknown query payload."""

    payload: str
    command_type: str = ""
    query_type: str = "unknown_query"
    event_type: str = ""


class Ex04UserCreated(BaseModel):
    """User-created event payload."""

    username: str
    command_type: str = ""
    query_type: str = ""
    event_type: str = "user_created"


class Ex04NoSubscriberEvent(BaseModel):
    """Event with no subscribers."""

    marker: str
    command_type: str = ""
    query_type: str = ""
    event_type: str = "no_subscribers"


class Ex05HandlerBad(BaseModel):
    """Intentionally incomplete handler protocol stub."""

    model_config = ConfigDict(frozen=False)


class Ex05HandlerLike(BaseModel):
    """Protocol-compatible handler model."""

    model_config = ConfigDict(frozen=False)
    data: m.ConfigMap = Field(default_factory=lambda: m.ConfigMap(root={}))


class Ex05GoodProcessor(BaseModel):
    """Processor that satisfies protocol checks."""

    model_config = ConfigDict(frozen=False)

    def process(self) -> bool:
        """Return successful processing state."""
        return True

    @classmethod
    @override
    def validate(cls, value: t.ContainerValue) -> Ex05GoodProcessor:
        """Validate processor payload."""
        return cls.model_validate(value)

    def _protocol_name(self) -> str:
        return "HasModelDump"


class Ex05BadProcessor(BaseModel):
    """Processor that intentionally fails protocol checks."""

    model_config = ConfigDict(frozen=False)

    def _protocol_name(self) -> str:
        return "HasModelDump"


class Ex10ContextPayload(BaseModel):
    """Context payload used to build execution context."""

    handler_name: str
    handler_mode: str


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


class Ex07DemoPlugin(BaseModel):
    """Demo plugin model."""

    name: str


class Ex14UserDTO(BaseModel):
    """User DTO payload."""

    id: str
    name: str
    email: str


class SharedPerson(BaseModel):
    """Shared random person model."""

    name: str
    age: int


class SharedHandle(BaseModel):
    """Shared handle model."""

    value: int
    cleaned: bool = False
