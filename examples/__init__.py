# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Examples package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _TYPE_CHECKING:
    from examples import (
        _models,
        ex_01_flext_result,
        ex_02_flext_settings,
        ex_03_flext_logger,
        ex_04_flext_dispatcher,
        ex_05_flext_mixins,
        ex_06_flext_context,
        ex_07_flext_exceptions,
        ex_08_flext_container,
        ex_09_flext_decorators,
        ex_10_flext_handlers,
        ex_11_flext_service,
        ex_12_flext_registry,
        logging_config_once_pattern,
        models,
        shared,
    )
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
    from examples.ex_01_flext_result import Ex01r
    from examples.ex_02_flext_settings import Ex02FlextSettings
    from examples.ex_03_flext_logger import Ex03FlextLogger
    from examples.ex_04_flext_dispatcher import Ex04FlextDispatcher
    from examples.ex_05_flext_mixins import Ex05FlextMixins
    from examples.ex_06_flext_context import Ex06FlextContext
    from examples.ex_07_flext_exceptions import Ex07FlextExceptions
    from examples.ex_08_flext_container import Ex08FlextContainer
    from examples.ex_09_flext_decorators import Ex09FlextDecorators
    from examples.ex_10_flext_handlers import Ex10FlextHandlers
    from examples.ex_11_flext_service import Ex11FlextService
    from examples.ex_12_flext_registry import Ex12FlextRegistry
    from examples.logging_config_once_pattern import (
        DatabaseService,
        MigrationService,
        main,
    )
    from examples.models import (
        FlextCoreExampleModels,
        FlextCoreExampleModels as m,
        UserInput,
        UserProfile,
        em,
    )
    from examples.shared import Examples
    from flext_core import FlextTypes

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = merge_lazy_imports(
    ("examples._models",),
    {
        "DatabaseService": "examples.logging_config_once_pattern",
        "Ex01r": "examples.ex_01_flext_result",
        "Ex02FlextSettings": "examples.ex_02_flext_settings",
        "Ex03FlextLogger": "examples.ex_03_flext_logger",
        "Ex04FlextDispatcher": "examples.ex_04_flext_dispatcher",
        "Ex05FlextMixins": "examples.ex_05_flext_mixins",
        "Ex06FlextContext": "examples.ex_06_flext_context",
        "Ex07FlextExceptions": "examples.ex_07_flext_exceptions",
        "Ex08FlextContainer": "examples.ex_08_flext_container",
        "Ex09FlextDecorators": "examples.ex_09_flext_decorators",
        "Ex10FlextHandlers": "examples.ex_10_flext_handlers",
        "Ex11FlextService": "examples.ex_11_flext_service",
        "Ex12FlextRegistry": "examples.ex_12_flext_registry",
        "Examples": "examples.shared",
        "FlextCoreExampleModels": "examples.models",
        "MigrationService": "examples.logging_config_once_pattern",
        "UserInput": "examples.models",
        "UserProfile": "examples.models",
        "_models": "examples._models",
        "em": "examples.models",
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
        "logging_config_once_pattern": "examples.logging_config_once_pattern",
        "m": ("examples.models", "FlextCoreExampleModels"),
        "main": "examples.logging_config_once_pattern",
        "models": "examples.models",
        "shared": "examples.shared",
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
