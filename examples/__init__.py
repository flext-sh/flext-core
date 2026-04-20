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
    from examples._models.errors import ExamplesFlextCoreModelsErrors
    from examples._models.ex00 import ExamplesFlextCoreModelsEx00
    from examples._models.ex01 import ExamplesFlextCoreModelsEx01
    from examples._models.ex02 import (
        Ex02CacheService,
        Ex02DatabaseService,
        Ex02EmailService,
        Ex02TestConfig,
        ExamplesFlextCoreModelsEx02,
        ExamplesFlextCoreSettingsEx02TestConfig,
    )
    from examples._models.ex03 import (
        Ex03Email,
        Ex03Money,
        Ex03Order,
        Ex03OrderItem,
        Ex03User,
        ExamplesFlextCoreModelsEx03,
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
        ExamplesFlextCoreModelsEx04,
    )
    from examples._models.ex05 import (
        Ex05BadProcessor,
        Ex05GoodProcessor,
        Ex05HandlerBad,
        Ex05HandlerLike,
        Ex05StatusEnum,
        Ex05UserModel,
        ExamplesFlextCoreModelsEx05,
    )
    from examples._models.ex07 import (
        Ex07CreateUserCommand,
        Ex07DemoPlugin,
        Ex07GetUserQuery,
        Ex07UserCreatedEvent,
        ExamplesFlextCoreModelsEx07,
    )
    from examples._models.ex08 import Ex08Order, Ex08User, ExamplesFlextCoreModelsEx08
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
        ExamplesFlextCoreModelsEx10,
    )
    from examples._models.ex11 import (
        Ex11CommandBusStub,
        Ex11EntityStub,
        Ex11HandlerLike,
        Ex11HandlerLikeService,
        Ex11Payload,
        Ex11ProcessorProtocolBad,
        Ex11ProcessorProtocolGood,
        ExamplesFlextCoreModelsEx11,
    )
    from examples._models.ex12 import (
        Ex12CommandA,
        Ex12CommandB,
        ExamplesFlextCoreModelsEx12,
    )
    from examples._models.ex14 import ExamplesFlextCoreModelsEx14
    from examples._models.exsettings import ExSettingsAppSettings
    from examples._models.output import ExamplesFlextCoreModelsOutput
    from examples._models.shared import (
        ExamplesFlextCoreSharedHandle,
        ExamplesFlextCoreSharedPerson,
    )
    from examples.constants import c
    from examples.ex_01_flext_result import Ex01r
    from examples.ex_02_flext_settings import Ex02FlextSettings
    from examples.ex_03_flext_logger import Ex03LoggingDsl
    from examples.ex_04_flext_dispatcher import Ex04DispatchDsl
    from examples.ex_05_flext_mixins import Ex05FlextMixins
    from examples.ex_06_flext_context import Ex06FlextContext
    from examples.ex_07_flext_exceptions import Ex07FlextExceptions
    from examples.ex_08_flext_container import Ex08FlextContainer
    from examples.ex_09_flext_decorators import Ex09FlextDecorators
    from examples.ex_10_flext_handlers import Ex10FlextHandlers
    from examples.ex_11_flext_service import Ex11FlextService, ExampleService
    from examples.ex_12_flext_registry import Ex12RegistryDsl
    from examples.logging_config_once_pattern import DatabaseService, MigrationService
    from examples.models import ExamplesFlextCoreModels, m
    from examples.protocols import p
    from examples.shared import ExamplesFlextCoreShared
    from examples.typings import ExamplesFlextCoreTypes, t
    from examples.utilities import u
    from flext_core import d, e, h, r, s, x
_LAZY_IMPORTS = merge_lazy_imports(
    ("._models",),
    build_lazy_import_map(
        {
            "._models.errors": ("ExamplesFlextCoreModelsErrors",),
            "._models.ex00": ("ExamplesFlextCoreModelsEx00",),
            "._models.ex01": ("ExamplesFlextCoreModelsEx01",),
            "._models.ex02": (
                "Ex02CacheService",
                "Ex02DatabaseService",
                "Ex02EmailService",
                "Ex02TestConfig",
                "ExamplesFlextCoreModelsEx02",
                "ExamplesFlextCoreSettingsEx02TestConfig",
            ),
            "._models.ex03": (
                "Ex03Email",
                "Ex03Money",
                "Ex03Order",
                "Ex03OrderItem",
                "Ex03User",
                "ExamplesFlextCoreModelsEx03",
            ),
            "._models.ex04": (
                "Ex04AutoCommand",
                "Ex04CreateUser",
                "Ex04DeleteUser",
                "Ex04FailingDelete",
                "Ex04GetUser",
                "Ex04NoSubscriberEvent",
                "Ex04Ping",
                "Ex04UnknownQuery",
                "Ex04UserCreated",
                "ExamplesFlextCoreModelsEx04",
            ),
            "._models.ex05": (
                "Ex05BadProcessor",
                "Ex05GoodProcessor",
                "Ex05HandlerBad",
                "Ex05HandlerLike",
                "Ex05StatusEnum",
                "Ex05UserModel",
                "ExamplesFlextCoreModelsEx05",
            ),
            "._models.ex07": (
                "Ex07CreateUserCommand",
                "Ex07DemoPlugin",
                "Ex07GetUserQuery",
                "Ex07UserCreatedEvent",
                "ExamplesFlextCoreModelsEx07",
            ),
            "._models.ex08": (
                "Ex08Order",
                "Ex08User",
                "ExamplesFlextCoreModelsEx08",
            ),
            "._models.ex10": (
                "Ex10CommandBusStub",
                "Ex10ContextPayload",
                "Ex10DerivedMessage",
                "Ex10Entity",
                "Ex10Message",
                "Ex10ProcessorBad",
                "Ex10ProcessorGood",
                "Ex10ProtocolHandler",
                "Ex10ServiceStub",
                "ExamplesFlextCoreModelsEx10",
            ),
            "._models.ex11": (
                "Ex11CommandBusStub",
                "Ex11EntityStub",
                "Ex11HandlerLike",
                "Ex11HandlerLikeService",
                "Ex11Payload",
                "Ex11ProcessorProtocolBad",
                "Ex11ProcessorProtocolGood",
                "ExamplesFlextCoreModelsEx11",
            ),
            "._models.ex12": (
                "Ex12CommandA",
                "Ex12CommandB",
                "ExamplesFlextCoreModelsEx12",
            ),
            "._models.ex14": ("ExamplesFlextCoreModelsEx14",),
            "._models.exsettings": ("ExSettingsAppSettings",),
            "._models.output": ("ExamplesFlextCoreModelsOutput",),
            "._models.shared": (
                "ExamplesFlextCoreSharedHandle",
                "ExamplesFlextCoreSharedPerson",
            ),
            ".constants": ("c",),
            ".ex_01_flext_result": ("Ex01r",),
            ".ex_02_flext_settings": ("Ex02FlextSettings",),
            ".ex_03_flext_logger": ("Ex03LoggingDsl",),
            ".ex_04_flext_dispatcher": ("Ex04DispatchDsl",),
            ".ex_05_flext_mixins": ("Ex05FlextMixins",),
            ".ex_06_flext_context": ("Ex06FlextContext",),
            ".ex_07_flext_exceptions": ("Ex07FlextExceptions",),
            ".ex_08_flext_container": ("Ex08FlextContainer",),
            ".ex_09_flext_decorators": ("Ex09FlextDecorators",),
            ".ex_10_flext_handlers": ("Ex10FlextHandlers",),
            ".ex_11_flext_service": (
                "Ex11FlextService",
                "ExampleService",
            ),
            ".ex_12_flext_registry": ("Ex12RegistryDsl",),
            ".logging_config_once_pattern": (
                "DatabaseService",
                "MigrationService",
            ),
            ".models": (
                "ExamplesFlextCoreModels",
                "m",
            ),
            ".protocols": ("p",),
            ".shared": ("ExamplesFlextCoreShared",),
            ".typings": (
                "ExamplesFlextCoreTypes",
                "t",
            ),
            ".utilities": ("u",),
            "flext_core": (
                "d",
                "e",
                "h",
                "r",
                "s",
                "x",
            ),
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

__all__: list[str] = [
    "DatabaseService",
    "Ex01r",
    "Ex02CacheService",
    "Ex02DatabaseService",
    "Ex02EmailService",
    "Ex02FlextSettings",
    "Ex02TestConfig",
    "Ex03Email",
    "Ex03LoggingDsl",
    "Ex03Money",
    "Ex03Order",
    "Ex03OrderItem",
    "Ex03User",
    "Ex04AutoCommand",
    "Ex04CreateUser",
    "Ex04DeleteUser",
    "Ex04DispatchDsl",
    "Ex04FailingDelete",
    "Ex04GetUser",
    "Ex04NoSubscriberEvent",
    "Ex04Ping",
    "Ex04UnknownQuery",
    "Ex04UserCreated",
    "Ex05BadProcessor",
    "Ex05FlextMixins",
    "Ex05GoodProcessor",
    "Ex05HandlerBad",
    "Ex05HandlerLike",
    "Ex05StatusEnum",
    "Ex05UserModel",
    "Ex06FlextContext",
    "Ex07CreateUserCommand",
    "Ex07DemoPlugin",
    "Ex07FlextExceptions",
    "Ex07GetUserQuery",
    "Ex07UserCreatedEvent",
    "Ex08FlextContainer",
    "Ex08Order",
    "Ex08User",
    "Ex09FlextDecorators",
    "Ex10CommandBusStub",
    "Ex10ContextPayload",
    "Ex10DerivedMessage",
    "Ex10Entity",
    "Ex10FlextHandlers",
    "Ex10Message",
    "Ex10ProcessorBad",
    "Ex10ProcessorGood",
    "Ex10ProtocolHandler",
    "Ex10ServiceStub",
    "Ex11CommandBusStub",
    "Ex11EntityStub",
    "Ex11FlextService",
    "Ex11HandlerLike",
    "Ex11HandlerLikeService",
    "Ex11Payload",
    "Ex11ProcessorProtocolBad",
    "Ex11ProcessorProtocolGood",
    "Ex12CommandA",
    "Ex12CommandB",
    "Ex12RegistryDsl",
    "ExSettingsAppSettings",
    "ExampleService",
    "ExamplesFlextCoreModels",
    "ExamplesFlextCoreModelsErrors",
    "ExamplesFlextCoreModelsEx00",
    "ExamplesFlextCoreModelsEx01",
    "ExamplesFlextCoreModelsEx02",
    "ExamplesFlextCoreModelsEx03",
    "ExamplesFlextCoreModelsEx04",
    "ExamplesFlextCoreModelsEx05",
    "ExamplesFlextCoreModelsEx07",
    "ExamplesFlextCoreModelsEx08",
    "ExamplesFlextCoreModelsEx10",
    "ExamplesFlextCoreModelsEx11",
    "ExamplesFlextCoreModelsEx12",
    "ExamplesFlextCoreModelsEx14",
    "ExamplesFlextCoreModelsOutput",
    "ExamplesFlextCoreSettingsEx02TestConfig",
    "ExamplesFlextCoreShared",
    "ExamplesFlextCoreSharedHandle",
    "ExamplesFlextCoreSharedPerson",
    "ExamplesFlextCoreTypes",
    "MigrationService",
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
