# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Models package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import examples._models.ex00 as _examples__models_ex00

    ex00 = _examples__models_ex00
    import examples._models.ex01 as _examples__models_ex01
    from examples._models.ex00 import Ex00UserInput, Ex00UserProfile

    ex01 = _examples__models_ex01
    import examples._models.ex02 as _examples__models_ex02
    from examples._models.ex01 import (
        Ex01DemonstrationResult,
        Ex01InvalidPersonPayload,
        Ex01RunDemonstrationCommand,
        Ex01User,
        Ex01ValidPersonPayload,
    )

    ex02 = _examples__models_ex02
    import examples._models.ex03 as _examples__models_ex03
    from examples._models.ex02 import (
        Ex02CacheService,
        Ex02DatabaseService,
        Ex02EmailService,
        Ex02TestConfig,
    )

    ex03 = _examples__models_ex03
    import examples._models.ex04 as _examples__models_ex04
    from examples._models.ex03 import (
        Ex03Email,
        Ex03Money,
        Ex03Order,
        Ex03OrderItem,
        Ex03User,
    )

    ex04 = _examples__models_ex04
    import examples._models.ex05 as _examples__models_ex05
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

    ex05 = _examples__models_ex05
    import examples._models.ex07 as _examples__models_ex07
    from examples._models.ex05 import (
        Ex05BadProcessor,
        Ex05GoodProcessor,
        Ex05HandlerBad,
        Ex05HandlerLike,
        Ex05StatusEnum,
        Ex05UserModel,
    )

    ex07 = _examples__models_ex07
    import examples._models.ex08 as _examples__models_ex08
    from examples._models.ex07 import (
        Ex07CreateUserCommand,
        Ex07DemoPlugin,
        Ex07GetUserQuery,
        Ex07UserCreatedEvent,
    )

    ex08 = _examples__models_ex08
    import examples._models.ex10 as _examples__models_ex10
    from examples._models.ex08 import Ex08Order, Ex08User

    ex10 = _examples__models_ex10
    import examples._models.ex11 as _examples__models_ex11
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

    ex11 = _examples__models_ex11
    import examples._models.ex12 as _examples__models_ex12
    from examples._models.ex11 import (
        Ex11CommandBusStub,
        Ex11EntityStub,
        Ex11HandlerLike,
        Ex11HandlerLikeService,
        Ex11Payload,
        Ex11ProcessorProtocolBad,
        Ex11ProcessorProtocolGood,
    )

    ex12 = _examples__models_ex12
    import examples._models.ex14 as _examples__models_ex14
    from examples._models.ex12 import Ex12CommandA, Ex12CommandB

    ex14 = _examples__models_ex14
    import examples._models.exconfig as _examples__models_exconfig
    from examples._models.ex14 import (
        Ex14CreateUserCommand,
        Ex14GetUserQuery,
        Ex14UserDTO,
    )

    exconfig = _examples__models_exconfig
    import examples._models.shared as _examples__models_shared
    from examples._models.exconfig import ExConfigAppConfig

    shared = _examples__models_shared
    from examples._models.shared import SharedHandle, SharedPerson
_LAZY_IMPORTS = {
    "Ex00UserInput": ("examples._models.ex00", "Ex00UserInput"),
    "Ex00UserProfile": ("examples._models.ex00", "Ex00UserProfile"),
    "Ex01DemonstrationResult": ("examples._models.ex01", "Ex01DemonstrationResult"),
    "Ex01InvalidPersonPayload": ("examples._models.ex01", "Ex01InvalidPersonPayload"),
    "Ex01RunDemonstrationCommand": (
        "examples._models.ex01",
        "Ex01RunDemonstrationCommand",
    ),
    "Ex01User": ("examples._models.ex01", "Ex01User"),
    "Ex01ValidPersonPayload": ("examples._models.ex01", "Ex01ValidPersonPayload"),
    "Ex02CacheService": ("examples._models.ex02", "Ex02CacheService"),
    "Ex02DatabaseService": ("examples._models.ex02", "Ex02DatabaseService"),
    "Ex02EmailService": ("examples._models.ex02", "Ex02EmailService"),
    "Ex02TestConfig": ("examples._models.ex02", "Ex02TestConfig"),
    "Ex03Email": ("examples._models.ex03", "Ex03Email"),
    "Ex03Money": ("examples._models.ex03", "Ex03Money"),
    "Ex03Order": ("examples._models.ex03", "Ex03Order"),
    "Ex03OrderItem": ("examples._models.ex03", "Ex03OrderItem"),
    "Ex03User": ("examples._models.ex03", "Ex03User"),
    "Ex04AutoCommand": ("examples._models.ex04", "Ex04AutoCommand"),
    "Ex04CreateUser": ("examples._models.ex04", "Ex04CreateUser"),
    "Ex04DeleteUser": ("examples._models.ex04", "Ex04DeleteUser"),
    "Ex04FailingDelete": ("examples._models.ex04", "Ex04FailingDelete"),
    "Ex04GetUser": ("examples._models.ex04", "Ex04GetUser"),
    "Ex04NoSubscriberEvent": ("examples._models.ex04", "Ex04NoSubscriberEvent"),
    "Ex04Ping": ("examples._models.ex04", "Ex04Ping"),
    "Ex04UnknownQuery": ("examples._models.ex04", "Ex04UnknownQuery"),
    "Ex04UserCreated": ("examples._models.ex04", "Ex04UserCreated"),
    "Ex05BadProcessor": ("examples._models.ex05", "Ex05BadProcessor"),
    "Ex05GoodProcessor": ("examples._models.ex05", "Ex05GoodProcessor"),
    "Ex05HandlerBad": ("examples._models.ex05", "Ex05HandlerBad"),
    "Ex05HandlerLike": ("examples._models.ex05", "Ex05HandlerLike"),
    "Ex05StatusEnum": ("examples._models.ex05", "Ex05StatusEnum"),
    "Ex05UserModel": ("examples._models.ex05", "Ex05UserModel"),
    "Ex07CreateUserCommand": ("examples._models.ex07", "Ex07CreateUserCommand"),
    "Ex07DemoPlugin": ("examples._models.ex07", "Ex07DemoPlugin"),
    "Ex07GetUserQuery": ("examples._models.ex07", "Ex07GetUserQuery"),
    "Ex07UserCreatedEvent": ("examples._models.ex07", "Ex07UserCreatedEvent"),
    "Ex08Order": ("examples._models.ex08", "Ex08Order"),
    "Ex08User": ("examples._models.ex08", "Ex08User"),
    "Ex10CommandBusStub": ("examples._models.ex10", "Ex10CommandBusStub"),
    "Ex10ContextPayload": ("examples._models.ex10", "Ex10ContextPayload"),
    "Ex10DerivedMessage": ("examples._models.ex10", "Ex10DerivedMessage"),
    "Ex10Entity": ("examples._models.ex10", "Ex10Entity"),
    "Ex10Message": ("examples._models.ex10", "Ex10Message"),
    "Ex10ProcessorBad": ("examples._models.ex10", "Ex10ProcessorBad"),
    "Ex10ProcessorGood": ("examples._models.ex10", "Ex10ProcessorGood"),
    "Ex10ProtocolHandler": ("examples._models.ex10", "Ex10ProtocolHandler"),
    "Ex10ServiceStub": ("examples._models.ex10", "Ex10ServiceStub"),
    "Ex11CommandBusStub": ("examples._models.ex11", "Ex11CommandBusStub"),
    "Ex11EntityStub": ("examples._models.ex11", "Ex11EntityStub"),
    "Ex11HandlerLike": ("examples._models.ex11", "Ex11HandlerLike"),
    "Ex11HandlerLikeService": ("examples._models.ex11", "Ex11HandlerLikeService"),
    "Ex11Payload": ("examples._models.ex11", "Ex11Payload"),
    "Ex11ProcessorProtocolBad": ("examples._models.ex11", "Ex11ProcessorProtocolBad"),
    "Ex11ProcessorProtocolGood": ("examples._models.ex11", "Ex11ProcessorProtocolGood"),
    "Ex12CommandA": ("examples._models.ex12", "Ex12CommandA"),
    "Ex12CommandB": ("examples._models.ex12", "Ex12CommandB"),
    "Ex14CreateUserCommand": ("examples._models.ex14", "Ex14CreateUserCommand"),
    "Ex14GetUserQuery": ("examples._models.ex14", "Ex14GetUserQuery"),
    "Ex14UserDTO": ("examples._models.ex14", "Ex14UserDTO"),
    "ExConfigAppConfig": ("examples._models.exconfig", "ExConfigAppConfig"),
    "SharedHandle": ("examples._models.shared", "SharedHandle"),
    "SharedPerson": ("examples._models.shared", "SharedPerson"),
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

__all__ = [
    "Ex00UserInput",
    "Ex00UserProfile",
    "Ex01DemonstrationResult",
    "Ex01InvalidPersonPayload",
    "Ex01RunDemonstrationCommand",
    "Ex01User",
    "Ex01ValidPersonPayload",
    "Ex02CacheService",
    "Ex02DatabaseService",
    "Ex02EmailService",
    "Ex02TestConfig",
    "Ex03Email",
    "Ex03Money",
    "Ex03Order",
    "Ex03OrderItem",
    "Ex03User",
    "Ex04AutoCommand",
    "Ex04CreateUser",
    "Ex04DeleteUser",
    "Ex04FailingDelete",
    "Ex04GetUser",
    "Ex04NoSubscriberEvent",
    "Ex04Ping",
    "Ex04UnknownQuery",
    "Ex04UserCreated",
    "Ex05BadProcessor",
    "Ex05GoodProcessor",
    "Ex05HandlerBad",
    "Ex05HandlerLike",
    "Ex05StatusEnum",
    "Ex05UserModel",
    "Ex07CreateUserCommand",
    "Ex07DemoPlugin",
    "Ex07GetUserQuery",
    "Ex07UserCreatedEvent",
    "Ex08Order",
    "Ex08User",
    "Ex10CommandBusStub",
    "Ex10ContextPayload",
    "Ex10DerivedMessage",
    "Ex10Entity",
    "Ex10Message",
    "Ex10ProcessorBad",
    "Ex10ProcessorGood",
    "Ex10ProtocolHandler",
    "Ex10ServiceStub",
    "Ex11CommandBusStub",
    "Ex11EntityStub",
    "Ex11HandlerLike",
    "Ex11HandlerLikeService",
    "Ex11Payload",
    "Ex11ProcessorProtocolBad",
    "Ex11ProcessorProtocolGood",
    "Ex12CommandA",
    "Ex12CommandB",
    "Ex14CreateUserCommand",
    "Ex14GetUserQuery",
    "Ex14UserDTO",
    "ExConfigAppConfig",
    "SharedHandle",
    "SharedPerson",
    "ex00",
    "ex01",
    "ex02",
    "ex03",
    "ex04",
    "ex05",
    "ex07",
    "ex08",
    "ex10",
    "ex11",
    "ex12",
    "ex14",
    "exconfig",
    "shared",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
