"""Centralized model facade for flext-core examples."""

from __future__ import annotations

from ._models.ex00 import Ex00UserInput, Ex00UserProfile
from ._models.ex01 import (
    Ex01DemonstrationResult,
    Ex01InvalidPersonPayload,
    Ex01RunDemonstrationCommand,
    Ex01User,
    Ex01ValidPersonPayload,
)
from ._models.ex04 import (
    Ex04AutoCommand,
    Ex04CreateUser,
    Ex04DeleteUser,
    Ex04FailingDelete,
    Ex04GetUser,
    Ex04NoSubscriberEvent,
    Ex04Ping,
    Ex04UnknownQuery,
    Ex04UserCreated,
)
from ._models.ex05 import Ex05BadProcessor, Ex05GoodProcessor, Ex05HandlerBad
from ._models.ex10 import (
    Ex10CommandBusStub,
    Ex10ContextPayload,
    Ex10DerivedMessage,
    Ex10Entity,
    Ex10Message,
    Ex10ProcessorBad,
    Ex10ProcessorGood,
    Ex10ProtocolHandler,
    Ex10ServiceStub,
)
from ._models.ex11 import (
    Ex11CommandBusStub,
    Ex11EntityStub,
    Ex11HandlerLike,
    Ex11HandlerLikeService,
    Ex11Payload,
    Ex11ProcessorProtocolBad,
    Ex11ProcessorProtocolGood,
)
from ._models.exconfig import ExConfigAppConfig


class FlextCoreExampleModels:
    """Facade namespace for all shared example models."""

    class Ex00:
        """Example 00 facade."""

        UserProfile = Ex00UserProfile
        UserInput = Ex00UserInput

    class Ex01:
        """Example 01 facade."""

        User = Ex01User
        DemonstrationResult = Ex01DemonstrationResult
        RunDemonstrationCommand = Ex01RunDemonstrationCommand
        ValidPersonPayload = Ex01ValidPersonPayload
        InvalidPersonPayload = Ex01InvalidPersonPayload

    class ExConfig:
        """Example 04 config facade."""

        AppConfig = ExConfigAppConfig

    class Ex04:
        """Example 04 dispatcher facade."""

        CreateUser = Ex04CreateUser
        GetUser = Ex04GetUser
        DeleteUser = Ex04DeleteUser
        FailingDelete = Ex04FailingDelete
        AutoCommand = Ex04AutoCommand
        Ping = Ex04Ping
        UnknownQuery = Ex04UnknownQuery
        UserCreated = Ex04UserCreated
        NoSubscriberEvent = Ex04NoSubscriberEvent

    class Ex05:
        """Example 05 mixins facade."""

        HandlerBad = Ex05HandlerBad
        GoodProcessor = Ex05GoodProcessor
        BadProcessor = Ex05BadProcessor

    class Ex10:
        """Example 10 handlers facade."""

        ContextPayload = Ex10ContextPayload
        Message = Ex10Message
        DerivedMessage = Ex10DerivedMessage
        Entity = Ex10Entity
        ProtocolHandler = Ex10ProtocolHandler
        ServiceStub = Ex10ServiceStub
        CommandBusStub = Ex10CommandBusStub
        ProcessorGood = Ex10ProcessorGood
        ProcessorBad = Ex10ProcessorBad

    class Ex11:
        """Example 11 service facade."""

        Payload = Ex11Payload
        HandlerLike = Ex11HandlerLike
        EntityStub = Ex11EntityStub
        ProcessorProtocolGood = Ex11ProcessorProtocolGood
        ProcessorProtocolBad = Ex11ProcessorProtocolBad
        CommandBusStub = Ex11CommandBusStub
        HandlerLikeService = Ex11HandlerLikeService


em = FlextCoreExampleModels
UserProfile = FlextCoreExampleModels.Ex00.UserProfile
UserInput = FlextCoreExampleModels.Ex00.UserInput

__all__ = ["FlextCoreExampleModels", "UserInput", "UserProfile", "em"]
