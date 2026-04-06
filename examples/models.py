"""Centralized model facade for flext-core examples."""

from __future__ import annotations

from examples import (
    Ex00UserInput,
    Ex00UserProfile,
    Ex01DemonstrationResult,
    Ex01InvalidPersonPayload,
    Ex01RunDemonstrationCommand,
    Ex01User,
    Ex01ValidPersonPayload,
    Ex02CacheService,
    Ex02DatabaseService,
    Ex02EmailService,
    Ex02TestConfig,
    Ex04AutoCommand,
    Ex04CreateUser,
    Ex04DeleteUser,
    Ex04FailingDelete,
    Ex04GetUser,
    Ex04NoSubscriberEvent,
    Ex04Ping,
    Ex04UnknownQuery,
    Ex04UserCreated,
    Ex05BadProcessor,
    Ex05GoodProcessor,
    Ex05HandlerBad,
    Ex05HandlerLike,
    Ex05StatusEnum,
    Ex05UserModel,
    Ex07CreateUserCommand,
    Ex07DemoPlugin,
    Ex07GetUserQuery,
    Ex07UserCreatedEvent,
    Ex08Order,
    Ex08User,
    Ex10CommandBusStub,
    Ex10ContextPayload,
    Ex10DerivedMessage,
    Ex10Entity,
    Ex10Message,
    Ex10ProcessorBad,
    Ex10ProcessorGood,
    Ex10ProtocolHandler,
    Ex10ServiceStub,
    Ex11CommandBusStub,
    Ex11EntityStub,
    Ex11HandlerLike,
    Ex11HandlerLikeService,
    Ex11Payload,
    Ex11ProcessorProtocolBad,
    Ex11ProcessorProtocolGood,
    Ex12CommandA,
    Ex12CommandB,
    Ex14CreateUserCommand,
    Ex14GetUserQuery,
    Ex14UserDTO,
    ExConfigAppConfig,
    SharedHandle,
    SharedPerson,
)
from flext_core.models import FlextModels

from ._models import Ex03Email, Ex03Money, Ex03Order, Ex03OrderItem, Ex03User


class FlextCoreExampleModels(FlextModels):
    """Facade namespace for all shared example models."""

    class Ex00:
        """Example 00 facade."""

        class UserProfile(Ex00UserProfile):
            """User profile model export."""

        class UserInput(Ex00UserInput):
            """User input model export."""

    class Ex01:
        """Example 01 facade."""

        class User(Ex01User):
            """Result user model export."""

        class DemonstrationResult(Ex01DemonstrationResult):
            """Result summary model export."""

        class RunDemonstrationCommand(Ex01RunDemonstrationCommand):
            """Result command model export."""

        class ValidPersonPayload(Ex01ValidPersonPayload):
            """Valid person payload export."""

        class InvalidPersonPayload(Ex01InvalidPersonPayload):
            """Invalid person payload export."""

    class Ex02:
        """Example 02 facade."""

        class TestConfig(Ex02TestConfig):
            """Settings test model export."""

        class DatabaseService(Ex02DatabaseService):
            """Database service export."""

        class CacheService(Ex02CacheService):
            """Cache service export."""

        class EmailService(Ex02EmailService):
            """Email service export."""

    class Ex03:
        """Example 03 facade."""

        class Email(Ex03Email):
            """Email value t.NormalizedValue export."""

        class Money(Ex03Money):
            """Money value t.NormalizedValue export."""

        class User(Ex03User):
            """Domain user export."""

        class OrderItem(Ex03OrderItem):
            """Order item export."""

        class Order(Ex03Order):
            """Order aggregate export."""

    class ExConfig:
        """Configuration facade."""

        class AppConfig(ExConfigAppConfig):
            """App settings export."""

    class Ex04:
        """Example 04 facade."""

        class CreateUser(Ex04CreateUser):
            """Create command export."""

        class GetUser(Ex04GetUser):
            """Get query export."""

        class DeleteUser(Ex04DeleteUser):
            """Delete command export."""

        class FailingDelete(Ex04FailingDelete):
            """Failing delete command export."""

        class AutoCommand(Ex04AutoCommand):
            """Auto command export."""

        class Ping(Ex04Ping):
            """Ping command export."""

        class UnknownQuery(Ex04UnknownQuery):
            """Unknown query export."""

        class UserCreated(Ex04UserCreated):
            """User-created event export."""

        class NoSubscriberEvent(Ex04NoSubscriberEvent):
            """No-subscriber event export."""

    class Ex05:
        """Example 05 facade."""

        StatusEnum = Ex05StatusEnum
        UserModel = Ex05UserModel

        class HandlerLike(Ex05HandlerLike):
            """Handler-like model export."""

        class HandlerBad(Ex05HandlerBad):
            """Bad handler model export."""

        class GoodProcessor(Ex05GoodProcessor):
            """Good processor model export."""

        class BadProcessor(Ex05BadProcessor):
            """Bad processor model export."""

    class Ex07:
        """Example 07 facade."""

        class CreateUserCommand(Ex07CreateUserCommand):
            """Create command export."""

        class UserCreatedEvent(Ex07UserCreatedEvent):
            """User-created event export."""

        class GetUserQuery(Ex07GetUserQuery):
            """Get-user query export."""

        class DemoPlugin(Ex07DemoPlugin):
            """Demo plugin export."""

    class Ex08:
        """Example 08 facade."""

        class User(Ex08User):
            """Integration user export."""

        class Order(Ex08Order):
            """Integration order export."""

    class Ex10:
        """Example 10 facade."""

        class ContextPayload(Ex10ContextPayload):
            """Context payload export."""

        class Message(Ex10Message):
            """Message export."""

        class DerivedMessage(Ex10DerivedMessage):
            """Derived message export."""

        class Entity(Ex10Entity):
            """Entity export."""

        class ProtocolHandler(Ex10ProtocolHandler):
            """Protocol handler export."""

        class ServiceStub(Ex10ServiceStub):
            """Service stub export."""

        class CommandBusStub(Ex10CommandBusStub):
            """Command bus stub export."""

        class ProcessorGood(Ex10ProcessorGood):
            """Good processor export."""

        class ProcessorBad(Ex10ProcessorBad):
            """Bad processor export."""

    class Ex11:
        """Example 11 facade."""

        class Payload(Ex11Payload):
            """Payload export."""

        class HandlerLike(Ex11HandlerLike):
            """Handler-like export."""

        class EntityStub(Ex11EntityStub):
            """Entity stub export."""

        class ProcessorProtocolGood(Ex11ProcessorProtocolGood):
            """Good processor protocol export."""

        class ProcessorProtocolBad(Ex11ProcessorProtocolBad):
            """Bad processor protocol export."""

        class CommandBusStub(Ex11CommandBusStub):
            """Command bus stub export."""

        class HandlerLikeService(Ex11HandlerLikeService):
            """Handler-like service export."""

    class Ex12:
        """Example 12 facade."""

        class CommandA(Ex12CommandA):
            """Registry command A export."""

        class CommandB(Ex12CommandB):
            """Registry command B export."""

    class Ex14:
        """Example 14 facade."""

        class CreateUserCommand(Ex14CreateUserCommand):
            """Create-user command export."""

        class GetUserQuery(Ex14GetUserQuery):
            """Get-user query export."""

        class UserDTO(Ex14UserDTO):
            """User DTO export."""

    class Shared:
        """Shared helper facade."""

        class Person(SharedPerson):
            """Shared person export."""

        class Handle(SharedHandle):
            """Shared handle export."""


em = FlextCoreExampleModels


class UserProfile(FlextCoreExampleModels.Ex00.UserProfile):
    """Top-level user profile export."""


class UserInput(FlextCoreExampleModels.Ex00.UserInput):
    """Top-level user input export."""


__all__ = ["FlextCoreExampleModels", "UserInput", "UserProfile", "em"]
