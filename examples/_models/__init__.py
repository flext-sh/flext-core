# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from _models.ex00 import ExamplesFlextCoreModelsEx00
    from _models.ex01 import ExamplesFlextCoreModelsEx01
    from _models.ex02 import (
        Ex02CacheService,
        Ex02DatabaseService,
        Ex02EmailService,
        Ex02TestConfig,
    )
    from _models.ex03 import Ex03Email, Ex03Money, Ex03Order, Ex03OrderItem, Ex03User
    from _models.ex04 import (
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
    from _models.ex05 import (
        Ex05BadProcessor,
        Ex05GoodProcessor,
        Ex05HandlerBad,
        Ex05HandlerLike,
        Ex05StatusEnum,
        Ex05UserModel,
    )
    from _models.ex07 import (
        Ex07CreateUserCommand,
        Ex07DemoPlugin,
        Ex07GetUserQuery,
        Ex07UserCreatedEvent,
    )
    from _models.ex08 import Ex08Order, Ex08User
    from _models.ex10 import (
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
    from _models.ex11 import (
        Ex11CommandBusStub,
        Ex11EntityStub,
        Ex11HandlerLike,
        Ex11HandlerLikeService,
        Ex11Payload,
        Ex11ProcessorProtocolBad,
        Ex11ProcessorProtocolGood,
    )
    from _models.ex12 import Ex12CommandA, Ex12CommandB
    from _models.ex14 import Ex14CreateUserCommand, Ex14GetUserQuery, Ex14UserDTO
    from _models.exconfig import ExConfigAppConfig
    from _models.shared import SharedHandle, SharedPerson
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".ex00": ("ExamplesFlextCoreModelsEx00",),
        ".ex01": ("ExamplesFlextCoreModelsEx01",),
        ".ex02": (
            "Ex02CacheService",
            "Ex02DatabaseService",
            "Ex02EmailService",
            "Ex02TestConfig",
        ),
        ".ex03": (
            "Ex03Email",
            "Ex03Money",
            "Ex03Order",
            "Ex03OrderItem",
            "Ex03User",
        ),
        ".ex04": (
            "Ex04AutoCommand",
            "Ex04CreateUser",
            "Ex04DeleteUser",
            "Ex04FailingDelete",
            "Ex04GetUser",
            "Ex04NoSubscriberEvent",
            "Ex04Ping",
            "Ex04UnknownQuery",
            "Ex04UserCreated",
        ),
        ".ex05": (
            "Ex05BadProcessor",
            "Ex05GoodProcessor",
            "Ex05HandlerBad",
            "Ex05HandlerLike",
            "Ex05StatusEnum",
            "Ex05UserModel",
        ),
        ".ex07": (
            "Ex07CreateUserCommand",
            "Ex07DemoPlugin",
            "Ex07GetUserQuery",
            "Ex07UserCreatedEvent",
        ),
        ".ex08": (
            "Ex08Order",
            "Ex08User",
        ),
        ".ex10": (
            "Ex10CommandBusStub",
            "Ex10ContextPayload",
            "Ex10DerivedMessage",
            "Ex10Entity",
            "Ex10Message",
            "Ex10ProcessorBad",
            "Ex10ProcessorGood",
            "Ex10ProtocolHandler",
            "Ex10ServiceStub",
        ),
        ".ex11": (
            "Ex11CommandBusStub",
            "Ex11EntityStub",
            "Ex11HandlerLike",
            "Ex11HandlerLikeService",
            "Ex11Payload",
            "Ex11ProcessorProtocolBad",
            "Ex11ProcessorProtocolGood",
        ),
        ".ex12": (
            "Ex12CommandA",
            "Ex12CommandB",
        ),
        ".ex14": (
            "Ex14CreateUserCommand",
            "Ex14GetUserQuery",
            "Ex14UserDTO",
        ),
        ".exconfig": ("ExConfigAppConfig",),
        ".shared": (
            "SharedHandle",
            "SharedPerson",
        ),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__ = [
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
    "ExamplesFlextCoreModelsEx00",
    "ExamplesFlextCoreModelsEx01",
    "SharedHandle",
    "SharedPerson",
]
