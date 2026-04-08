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
    from examples.models import ExamplesFlextCoreModels, ExamplesFlextCoreModels as m
    from flext_core.constants import FlextConstants as c
    from flext_core.decorators import FlextDecorators as d
    from flext_core.exceptions import FlextExceptions as e
    from flext_core.handlers import FlextHandlers as h
    from flext_core.mixins import FlextMixins as x
    from flext_core.protocols import FlextProtocols as p
    from flext_core.result import FlextResult as r
    from flext_core.service import FlextService as s
    from flext_core.typings import FlextTypes as t
    from flext_core.utilities import FlextUtilities as u
_LAZY_IMPORTS = merge_lazy_imports(
    ("._models",),
    build_lazy_import_map(
        {
            ".models": ("ExamplesFlextCoreModels",),
        },
        alias_groups={
            ".models": (("m", "ExamplesFlextCoreModels"),),
            "flext_core.constants": (("c", "FlextConstants"),),
            "flext_core.decorators": (("d", "FlextDecorators"),),
            "flext_core.exceptions": (("e", "FlextExceptions"),),
            "flext_core.handlers": (("h", "FlextHandlers"),),
            "flext_core.mixins": (("x", "FlextMixins"),),
            "flext_core.protocols": (("p", "FlextProtocols"),),
            "flext_core.result": (("r", "FlextResult"),),
            "flext_core.service": (("s", "FlextService"),),
            "flext_core.typings": (("t", "FlextTypes"),),
            "flext_core.utilities": (("u", "FlextUtilities"),),
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
    "ExamplesFlextCoreModels",
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
