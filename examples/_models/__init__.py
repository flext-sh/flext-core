# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Shared model modules for examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from examples._models import (
        ex00,
        ex01,
        ex02,
        ex03,
        ex04,
        ex05,
        ex07,
        ex08,
        ex10,
        ex11,
        ex12,
        ex14,
        exconfig,
        shared,
    )
    from examples._models.ex00 import Ex00UserInput, Ex00UserProfile
    from examples._models.ex01 import (
        Ex01DemonstrationResult,
        Ex01InvalidPersonPayload,
        Ex01RunDemonstrationCommand,
        Ex01User,
        Ex01ValidPersonPayload,
    )
    from examples._models.ex02 import (
        Ex02CacheService,
        Ex02DatabaseService,
        Ex02EmailService,
        Ex02TestConfig,
    )
    from examples._models.ex03 import (
        Ex03Email,
        Ex03Money,
        Ex03Order,
        Ex03OrderItem,
        Ex03User,
    )
    from examples._models.ex04 import (
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
    from examples._models.ex05 import (
        Ex05BadProcessor,
        Ex05GoodProcessor,
        Ex05HandlerBad,
        Ex05HandlerLike,
        Ex05StatusEnum,
        Ex05UserModel,
    )
    from examples._models.ex07 import (
        Ex07CreateUserCommand,
        Ex07DemoPlugin,
        Ex07GetUserQuery,
        Ex07UserCreatedEvent,
    )
    from examples._models.ex08 import Ex08Order, Ex08User
    from examples._models.ex10 import (
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
    from examples._models.ex11 import (
        Ex11CommandBusStub,
        Ex11EntityStub,
        Ex11HandlerLike,
        Ex11HandlerLikeService,
        Ex11Payload,
        Ex11ProcessorProtocolBad,
        Ex11ProcessorProtocolGood,
    )
    from examples._models.ex12 import Ex12CommandA, Ex12CommandB
    from examples._models.ex14 import (
        Ex14CreateUserCommand,
        Ex14GetUserQuery,
        Ex14UserDTO,
    )
    from examples._models.exconfig import ExConfigAppConfig
    from examples._models.shared import SharedHandle, SharedPerson
    from flext_core import FlextTypes

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "Ex00UserInput": "examples._models.ex00",
    "Ex00UserProfile": "examples._models.ex00",
    "Ex01DemonstrationResult": "examples._models.ex01",
    "Ex01InvalidPersonPayload": "examples._models.ex01",
    "Ex01RunDemonstrationCommand": "examples._models.ex01",
    "Ex01User": "examples._models.ex01",
    "Ex01ValidPersonPayload": "examples._models.ex01",
    "Ex02CacheService": "examples._models.ex02",
    "Ex02DatabaseService": "examples._models.ex02",
    "Ex02EmailService": "examples._models.ex02",
    "Ex02TestConfig": "examples._models.ex02",
    "Ex03Email": "examples._models.ex03",
    "Ex03Money": "examples._models.ex03",
    "Ex03Order": "examples._models.ex03",
    "Ex03OrderItem": "examples._models.ex03",
    "Ex03User": "examples._models.ex03",
    "Ex04AutoCommand": "examples._models.ex04",
    "Ex04CreateUser": "examples._models.ex04",
    "Ex04DeleteUser": "examples._models.ex04",
    "Ex04FailingDelete": "examples._models.ex04",
    "Ex04GetUser": "examples._models.ex04",
    "Ex04NoSubscriberEvent": "examples._models.ex04",
    "Ex04Ping": "examples._models.ex04",
    "Ex04UnknownQuery": "examples._models.ex04",
    "Ex04UserCreated": "examples._models.ex04",
    "Ex05BadProcessor": "examples._models.ex05",
    "Ex05GoodProcessor": "examples._models.ex05",
    "Ex05HandlerBad": "examples._models.ex05",
    "Ex05HandlerLike": "examples._models.ex05",
    "Ex05StatusEnum": "examples._models.ex05",
    "Ex05UserModel": "examples._models.ex05",
    "Ex07CreateUserCommand": "examples._models.ex07",
    "Ex07DemoPlugin": "examples._models.ex07",
    "Ex07GetUserQuery": "examples._models.ex07",
    "Ex07UserCreatedEvent": "examples._models.ex07",
    "Ex08Order": "examples._models.ex08",
    "Ex08User": "examples._models.ex08",
    "Ex10CommandBusStub": "examples._models.ex10",
    "Ex10ContextPayload": "examples._models.ex10",
    "Ex10DerivedMessage": "examples._models.ex10",
    "Ex10Entity": "examples._models.ex10",
    "Ex10Message": "examples._models.ex10",
    "Ex10ProcessorBad": "examples._models.ex10",
    "Ex10ProcessorGood": "examples._models.ex10",
    "Ex10ProtocolHandler": "examples._models.ex10",
    "Ex10ServiceStub": "examples._models.ex10",
    "Ex11CommandBusStub": "examples._models.ex11",
    "Ex11EntityStub": "examples._models.ex11",
    "Ex11HandlerLike": "examples._models.ex11",
    "Ex11HandlerLikeService": "examples._models.ex11",
    "Ex11Payload": "examples._models.ex11",
    "Ex11ProcessorProtocolBad": "examples._models.ex11",
    "Ex11ProcessorProtocolGood": "examples._models.ex11",
    "Ex12CommandA": "examples._models.ex12",
    "Ex12CommandB": "examples._models.ex12",
    "Ex14CreateUserCommand": "examples._models.ex14",
    "Ex14GetUserQuery": "examples._models.ex14",
    "Ex14UserDTO": "examples._models.ex14",
    "ExConfigAppConfig": "examples._models.exconfig",
    "SharedHandle": "examples._models.shared",
    "SharedPerson": "examples._models.shared",
    "ex00": "examples._models.ex00",
    "ex01": "examples._models.ex01",
    "ex02": "examples._models.ex02",
    "ex03": "examples._models.ex03",
    "ex04": "examples._models.ex04",
    "ex05": "examples._models.ex05",
    "ex07": "examples._models.ex07",
    "ex08": "examples._models.ex08",
    "ex10": "examples._models.ex10",
    "ex11": "examples._models.ex11",
    "ex12": "examples._models.ex12",
    "ex14": "examples._models.ex14",
    "exconfig": "examples._models.exconfig",
    "shared": "examples._models.shared",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
