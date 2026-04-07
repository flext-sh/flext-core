# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Examples package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _t.TYPE_CHECKING:
    import examples._models as _examples__models

    _models = _examples__models
    import examples.ex_01_flext_result as _examples_ex_01_flext_result
    from examples._models import (
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
        Ex03Email,
        Ex03Money,
        Ex03Order,
        Ex03OrderItem,
        Ex03User,
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
    )

    ex_01_flext_result = _examples_ex_01_flext_result
    import examples.ex_02_flext_settings as _examples_ex_02_flext_settings
    from examples.ex_01_flext_result import Ex01r

    ex_02_flext_settings = _examples_ex_02_flext_settings
    import examples.ex_03_flext_logger as _examples_ex_03_flext_logger
    from examples.ex_02_flext_settings import Ex02FlextSettings

    ex_03_flext_logger = _examples_ex_03_flext_logger
    import examples.ex_04_flext_dispatcher as _examples_ex_04_flext_dispatcher
    from examples.ex_03_flext_logger import Ex03FlextLogger

    ex_04_flext_dispatcher = _examples_ex_04_flext_dispatcher
    import examples.ex_05_flext_mixins as _examples_ex_05_flext_mixins
    from examples.ex_04_flext_dispatcher import Ex04FlextDispatcher

    ex_05_flext_mixins = _examples_ex_05_flext_mixins
    import examples.ex_06_flext_context as _examples_ex_06_flext_context
    from examples.ex_05_flext_mixins import Ex05FlextMixins

    ex_06_flext_context = _examples_ex_06_flext_context
    import examples.ex_07_flext_exceptions as _examples_ex_07_flext_exceptions
    from examples.ex_06_flext_context import Ex06FlextContext

    ex_07_flext_exceptions = _examples_ex_07_flext_exceptions
    import examples.ex_08_flext_container as _examples_ex_08_flext_container
    from examples.ex_07_flext_exceptions import Ex07FlextExceptions

    ex_08_flext_container = _examples_ex_08_flext_container
    import examples.ex_09_flext_decorators as _examples_ex_09_flext_decorators
    from examples.ex_08_flext_container import Ex08FlextContainer

    ex_09_flext_decorators = _examples_ex_09_flext_decorators
    import examples.ex_10_flext_handlers as _examples_ex_10_flext_handlers
    from examples.ex_09_flext_decorators import Ex09FlextDecorators

    ex_10_flext_handlers = _examples_ex_10_flext_handlers
    import examples.ex_11_flext_service as _examples_ex_11_flext_service
    from examples.ex_10_flext_handlers import Ex10FlextHandlers

    ex_11_flext_service = _examples_ex_11_flext_service
    import examples.ex_12_flext_registry as _examples_ex_12_flext_registry
    from examples.ex_11_flext_service import Ex11FlextService

    ex_12_flext_registry = _examples_ex_12_flext_registry
    import examples.logging_config_once_pattern as _examples_logging_config_once_pattern
    from examples.ex_12_flext_registry import Ex12FlextRegistry

    logging_config_once_pattern = _examples_logging_config_once_pattern
    import examples.models as _examples_models
    from examples.logging_config_once_pattern import (
        DatabaseService,
        MigrationService,
        main,
    )

    models = _examples_models
    import examples.shared as _examples_shared
    from examples.models import (
        FlextCoreExamplesModels,
        FlextCoreExamplesModels as m,
        Handle,
        Person,
    )

    shared = _examples_shared
    from examples.shared import Examples
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
    ("examples._models",),
    {
        "DatabaseService": ("examples.logging_config_once_pattern", "DatabaseService"),
        "Ex01r": ("examples.ex_01_flext_result", "Ex01r"),
        "Ex02FlextSettings": ("examples.ex_02_flext_settings", "Ex02FlextSettings"),
        "Ex03FlextLogger": ("examples.ex_03_flext_logger", "Ex03FlextLogger"),
        "Ex04FlextDispatcher": (
            "examples.ex_04_flext_dispatcher",
            "Ex04FlextDispatcher",
        ),
        "Ex05FlextMixins": ("examples.ex_05_flext_mixins", "Ex05FlextMixins"),
        "Ex06FlextContext": ("examples.ex_06_flext_context", "Ex06FlextContext"),
        "Ex07FlextExceptions": (
            "examples.ex_07_flext_exceptions",
            "Ex07FlextExceptions",
        ),
        "Ex08FlextContainer": ("examples.ex_08_flext_container", "Ex08FlextContainer"),
        "Ex09FlextDecorators": (
            "examples.ex_09_flext_decorators",
            "Ex09FlextDecorators",
        ),
        "Ex10FlextHandlers": ("examples.ex_10_flext_handlers", "Ex10FlextHandlers"),
        "Ex11FlextService": ("examples.ex_11_flext_service", "Ex11FlextService"),
        "Ex12FlextRegistry": ("examples.ex_12_flext_registry", "Ex12FlextRegistry"),
        "Examples": ("examples.shared", "Examples"),
        "FlextCoreExamplesModels": ("examples.models", "FlextCoreExamplesModels"),
        "Handle": ("examples.models", "Handle"),
        "MigrationService": (
            "examples.logging_config_once_pattern",
            "MigrationService",
        ),
        "Person": ("examples.models", "Person"),
        "_models": "examples._models",
        "c": ("flext_core.constants", "FlextConstants"),
        "d": ("flext_core.decorators", "FlextDecorators"),
        "e": ("flext_core.exceptions", "FlextExceptions"),
        "ex_01_flext_result": "examples.ex_01_flext_result",
        "ex_02_flext_settings": "examples.ex_02_flext_settings",
        "ex_03_flext_logger": "examples.ex_03_flext_logger",
        "ex_04_flext_dispatcher": "examples.ex_04_flext_dispatcher",
        "ex_05_flext_mixins": "examples.ex_05_flext_mixins",
        "ex_06_flext_context": "examples.ex_06_flext_context",
        "ex_07_flext_exceptions": "examples.ex_07_flext_exceptions",
        "ex_08_flext_container": "examples.ex_08_flext_container",
        "ex_09_flext_decorators": "examples.ex_09_flext_decorators",
        "ex_10_flext_handlers": "examples.ex_10_flext_handlers",
        "ex_11_flext_service": "examples.ex_11_flext_service",
        "ex_12_flext_registry": "examples.ex_12_flext_registry",
        "h": ("flext_core.handlers", "FlextHandlers"),
        "logging_config_once_pattern": "examples.logging_config_once_pattern",
        "m": ("examples.models", "FlextCoreExamplesModels"),
        "main": ("examples.logging_config_once_pattern", "main"),
        "models": "examples.models",
        "p": ("flext_core.protocols", "FlextProtocols"),
        "r": ("flext_core.result", "FlextResult"),
        "s": ("flext_core.service", "FlextService"),
        "shared": "examples.shared",
        "t": ("flext_core.typings", "FlextTypes"),
        "u": ("flext_core.utilities", "FlextUtilities"),
        "x": ("flext_core.mixins", "FlextMixins"),
    },
)
_ = _LAZY_IMPORTS.pop("cleanup_submodule_namespace", None)
_ = _LAZY_IMPORTS.pop("install_lazy_exports", None)
_ = _LAZY_IMPORTS.pop("lazy_getattr", None)
_ = _LAZY_IMPORTS.pop("merge_lazy_imports", None)
_ = _LAZY_IMPORTS.pop("output", None)
_ = _LAZY_IMPORTS.pop("output_reporting", None)

__all__ = [
    "DatabaseService",
    "Ex00UserInput",
    "Ex00UserProfile",
    "Ex01DemonstrationResult",
    "Ex01InvalidPersonPayload",
    "Ex01RunDemonstrationCommand",
    "Ex01User",
    "Ex01ValidPersonPayload",
    "Ex01r",
    "Ex02CacheService",
    "Ex02DatabaseService",
    "Ex02EmailService",
    "Ex02FlextSettings",
    "Ex02TestConfig",
    "Ex03Email",
    "Ex03FlextLogger",
    "Ex03Money",
    "Ex03Order",
    "Ex03OrderItem",
    "Ex03User",
    "Ex04AutoCommand",
    "Ex04CreateUser",
    "Ex04DeleteUser",
    "Ex04FailingDelete",
    "Ex04FlextDispatcher",
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
    "Ex12FlextRegistry",
    "Ex14CreateUserCommand",
    "Ex14GetUserQuery",
    "Ex14UserDTO",
    "ExConfigAppConfig",
    "Examples",
    "FlextCoreExamplesModels",
    "Handle",
    "MigrationService",
    "Person",
    "SharedHandle",
    "SharedPerson",
    "_models",
    "c",
    "d",
    "e",
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
    "ex_01_flext_result",
    "ex_02_flext_settings",
    "ex_03_flext_logger",
    "ex_04_flext_dispatcher",
    "ex_05_flext_mixins",
    "ex_06_flext_context",
    "ex_07_flext_exceptions",
    "ex_08_flext_container",
    "ex_09_flext_decorators",
    "ex_10_flext_handlers",
    "ex_11_flext_service",
    "ex_12_flext_registry",
    "exconfig",
    "h",
    "logging_config_once_pattern",
    "m",
    "main",
    "models",
    "p",
    "r",
    "s",
    "shared",
    "t",
    "u",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
