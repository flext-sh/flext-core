"""Centralized Pydantic v2 example models for flext-core examples."""

from __future__ import annotations

from typing import override

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextSettings, c, m, r, t


class FlextCoreExampleModels:
    """Namespace for shared example models."""

    class Ex00:
        """Models used by example 00."""

        class UserProfile(m.Entity):
            """User profile transport model."""

            name: str = Field(min_length=1)
            email: str = Field(min_length=1)
            status: c.Domain.Status = c.Domain.Status.ACTIVE

            def activate(self) -> r[FlextCoreExampleModels.Ex00.UserProfile]:
                """Return an active profile instance."""
                if self.status == c.Domain.Status.ACTIVE:
                    return r[FlextCoreExampleModels.Ex00.UserProfile].ok(self)
                return r[FlextCoreExampleModels.Ex00.UserProfile].ok(
                    self.model_copy(update={"status": c.Domain.Status.ACTIVE})
                )

        class UserInput(m.Value):
            """Raw user input model."""

            name: str = Field(min_length=1)
            email: str = Field(min_length=1)

    class Ex01:
        """Models used by example 01."""

        class ValidPersonPayload(BaseModel):
            """Valid payload for person conversion tests."""

            name: str
            age: int

        class InvalidPersonPayload(BaseModel):
            """Invalid payload for person conversion tests."""

            name: str
            age: str

    class ExConfig:
        class AppConfig(FlextSettings):
            database_url: str = Field(
                default=f"postgresql://{c.Platform.DEFAULT_HOST}:5432/testdb",
                description="Database connection URL",
            )
            db_pool_size: int = Field(
                default=10,
                ge=1,
                le=c.Performance.MAX_DB_POOL_SIZE,
                description="Database connection pool size",
            )
            api_timeout: int = Field(default=30)
            api_host: str = Field(
                default=c.Platform.DEFAULT_HOST,
                min_length=1,
                max_length=c.Network.MAX_HOSTNAME_LENGTH,
                description="API server hostname",
            )
            api_port: t.Validation.PortNumber = Field(default=8080)
            debug: bool = Field(default=False)
            max_workers: int = Field(default=4)
            cache_enabled: bool = Field(default=True)
            cache_ttl: int = Field(
                default=3600,
                ge=0,
                le=c.Performance.MAX_TIMEOUT_SECONDS,
            )
            worker_timeout: int = Field(default=60)
            retry_attempts: int = Field(
                default=3,
                ge=0,
                le=c.Reliability.MAX_RETRY_ATTEMPTS,
            )

    class Ex04:
        """Models used by example 04 dispatcher checks."""

        class CreateUser(BaseModel):
            """Create command payload."""

            username: str
            command_type: str = "create_user"
            query_type: str = ""
            event_type: str = ""

        class GetUser(BaseModel):
            """Get query payload."""

            username: str
            command_type: str = ""
            query_type: str = "get_user"
            event_type: str = ""

        class DeleteUser(BaseModel):
            """Delete command payload."""

            username: str
            command_type: str = "delete_user"
            query_type: str = ""
            event_type: str = ""

        class FailingDelete(BaseModel):
            """Delete command that intentionally fails in tests."""

            username: str
            command_type: str = "failing_delete"
            query_type: str = ""
            event_type: str = ""

        class AutoCommand(BaseModel):
            """Auto command payload."""

            payload: str
            command_type: str = "auto_command"
            query_type: str = ""
            event_type: str = ""

        class Ping(BaseModel):
            """Ping command payload."""

            value: str
            command_type: str = "ping"
            query_type: str = ""
            event_type: str = ""

        class UnknownQuery(BaseModel):
            """Unknown query payload."""

            payload: str
            command_type: str = ""
            query_type: str = "unknown_query"
            event_type: str = ""

        class UserCreated(BaseModel):
            """User-created event payload."""

            username: str
            command_type: str = ""
            query_type: str = ""
            event_type: str = "user_created"

        class NoSubscriberEvent(BaseModel):
            """Event with no subscribers."""

            marker: str
            command_type: str = ""
            query_type: str = ""
            event_type: str = "no_subscribers"

    class Ex05:
        """Models used by example 05 mixin checks."""

        class HandlerBad(BaseModel):
            """Intentionally incomplete handler protocol stub."""

            model_config = ConfigDict(frozen=False)

        class GoodProcessor(BaseModel):
            """Processor that satisfies protocol checks."""

            model_config = ConfigDict(frozen=False)

            def process(self) -> bool:
                """Return successful processing state."""
                return True

            @classmethod
            @override
            def validate(
                cls, value: t.ContainerValue
            ) -> FlextCoreExampleModels.Ex05.GoodProcessor:
                """Validate processor payload."""
                return cls.model_validate(value)

            def _protocol_name(self) -> str:
                return "HasModelDump"

        class BadProcessor(BaseModel):
            """Processor that intentionally fails protocol checks."""

            model_config = ConfigDict(frozen=False)

            def _protocol_name(self) -> str:
                return "HasModelDump"

    class Ex10:
        """Models used by example 10 handlers checks."""

        class ContextPayload(BaseModel):
            """Context payload used to build execution context."""

            handler_name: str
            handler_mode: str

        class Message(m.Command):
            """Command message payload."""

            text: str

        class DerivedMessage(Message):
            """Derived message used for covariance checks."""

            pass

        class Entity(m.Value):
            """Entity-like value payload for runtime checks."""

            unique_id: str

        class ProtocolHandler(BaseModel):
            """Handler stub implementing protocol operations."""

            model_config = ConfigDict(frozen=False)

            @staticmethod
            def _protocol_name() -> str:
                return "ProtocolHandler"

            def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
                """Echo back handled message."""
                return r[t.ContainerValue].ok(message)

            def check_data(self, data: t.ContainerValue) -> r[bool]:
                """Check data is present."""
                return r[bool].ok(data is not None)

        class ServiceStub(BaseModel):
            """Service protocol stub."""

            model_config = ConfigDict(frozen=False)

            @property
            def is_valid(self) -> bool:
                """Return service validity state."""
                return True

            @staticmethod
            def _protocol_name() -> str:
                return "ServiceStub"

            def execute(self) -> r[t.ContainerValue]:
                """Execute service action."""
                return r[t.ContainerValue].ok(m.ConfigMap(root={"ok": True}))

            def get_service_info(self) -> m.ConfigMap:
                """Return service metadata."""
                return m.ConfigMap(root={"service": "stub"})

            def validate_business_rules(self) -> r[bool]:
                """Validate business rules."""
                return r[bool].ok(True)

        class CommandBusStub(BaseModel):
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

        class ProcessorGood(m.Value):
            """Good processor for protocol checks."""

            marker: str = "good"

            @staticmethod
            def _protocol_name() -> str:
                return "ProcessorGood"

            def process(self) -> str:
                """Process successfully."""
                return "ok"

        class ProcessorBad(m.Value):
            """Bad processor for protocol checks."""

            marker: str = "bad"

            @staticmethod
            def _protocol_name() -> str:
                return "ProcessorBad"

            def process(self) -> str:
                """Process successfully despite bad protocol metadata."""
                return "ok"

    class Ex11:
        """Models used by example 11 service checks."""

        class Payload(BaseModel):
            """Service payload model."""

            text: str
            count: int

        class HandlerLike(BaseModel):
            """Handler-like protocol stub."""

            model_config = ConfigDict(frozen=False)
            data: m.ConfigMap = Field(default_factory=lambda: m.ConfigMap(root={}))

            def handle(self) -> str:
                """Handle payload."""
                return "ok"

        class EntityStub(BaseModel):
            """Entity stub for protocol checks."""

            unique_id: str

        class ProcessorProtocolGood(BaseModel):
            """Processor protocol-compliant stub."""

            model_config = ConfigDict(frozen=False)
            status: str = "ok"

            def process(self) -> str:
                """Process successfully."""
                return "ok"

            def _protocol_name(self) -> str:
                return "ProcessorProtocolGood"

        class ProcessorProtocolBad(BaseModel):
            """Processor protocol non-compliant stub."""

            model_config = ConfigDict(frozen=False)
            status: str = "bad"

            def _protocol_name(self) -> str:
                return "ProcessorProtocolBad"

        class CommandBusStub(BaseModel):
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

        class HandlerLikeService(FlextSettings):
            """Service-like handler stub."""

            @classmethod
            @override
            def validate(
                cls, value: t.ContainerValue
            ) -> FlextCoreExampleModels.Ex11.HandlerLikeService:
                """Validate service payload."""
                return cls.model_validate(value)

            def can_handle(self, message_type: type) -> bool:
                """Check whether message type is handled."""
                return bool(message_type)

            def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
                """Handle service message."""
                return r[t.ContainerValue].ok(message)


em = FlextCoreExampleModels
UserProfile = FlextCoreExampleModels.Ex00.UserProfile
UserInput = FlextCoreExampleModels.Ex00.UserInput

__all__ = ["FlextCoreExampleModels", "UserInput", "UserProfile", "em"]
