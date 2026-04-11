# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if _t.TYPE_CHECKING:
    from examples._models.ex00 import ExamplesFlextCoreModelsEx00
    from examples._models.ex01 import ExamplesFlextCoreModelsEx01
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
    from examples.models import ExamplesFlextCoreModels, m
    from flext_core.constants import FlextConstants as c
    from flext_core.decorators import d
    from flext_core.exceptions import e
    from flext_core.handlers import h
    from flext_core.mixins import x
    from flext_core.protocols import p
    from flext_core.result import r
    from flext_core.service import s
    from flext_core.typings import t
    from flext_core.utilities import u
_LAZY_IMPORTS = merge_lazy_imports(
    ("._models",),
    build_lazy_import_map(
        {
            ".models": (
                "ExamplesFlextCoreModels",
                "m",
            ),
            "flext_core.decorators": ("d",),
            "flext_core.exceptions": ("e",),
            "flext_core.handlers": ("h",),
            "flext_core.mixins": ("x",),
            "flext_core.protocols": ("p",),
            "flext_core.result": ("r",),
            "flext_core.service": ("s",),
            "flext_core.typings": ("t",),
            "flext_core.utilities": ("u",),
        },
        alias_groups={
            "flext_core.constants": (("c", "FlextConstants"),),
        },
    ),
    exclude_names=(
        "cleanup_submodule_namespace",
        "install_lazy_exports",
        "lazy_getattr",
        "logger",
        "merge_lazy_imports",
        "output",
        "output_reporting",
    ),
    module_name=__name__,
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
    "ExamplesFlextCoreModels",
    "ExamplesFlextCoreModelsEx00",
    "ExamplesFlextCoreModelsEx01",
    "SharedHandle",
    "SharedPerson",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "u",
    "x",
]
