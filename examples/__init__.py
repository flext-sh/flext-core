# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Examples package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from examples._models.ex00 import Ex00UserInput, Ex00UserProfile
    from examples._models.ex01 import (
        Ex01DemonstrationResult,
        Ex01DemonstrationResult as r,
        Ex01InvalidPersonPayload,
        Ex01RunDemonstrationCommand,
        Ex01User,
        Ex01ValidPersonPayload,
    )
    from examples._models.ex02 import (
        Ex02CacheService,
        Ex02DatabaseService,
        Ex02DatabaseService as s,
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
    from examples.ex_01_flext_result import Ex01r
    from examples.ex_02_flext_settings import Ex02FlextSettings
    from examples.ex_03_flext_logger import Ex03FlextLogger
    from examples.ex_04_flext_dispatcher import Ex04FlextDispatcher
    from examples.ex_05_flext_mixins import Ex05FlextMixins, Ex05FlextMixins as x
    from examples.ex_06_flext_context import Ex06FlextContext
    from examples.ex_07_flext_exceptions import (
        Ex07FlextExceptions,
        Ex07FlextExceptions as e,
    )
    from examples.ex_08_flext_container import Ex08FlextContainer
    from examples.ex_09_flext_decorators import (
        Ex09FlextDecorators,
        Ex09FlextDecorators as d,
    )
    from examples.ex_10_flext_handlers import Ex10FlextHandlers, Ex10FlextHandlers as h
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
    from flext_core.typings import FlextTypes

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DatabaseService": ("examples.logging_config_once_pattern", "DatabaseService"),
    "Ex00UserInput": ("examples._models.ex00", "Ex00UserInput"),
    "Ex00UserProfile": ("examples._models.ex00", "Ex00UserProfile"),
    "Ex01DemonstrationResult": ("examples._models.ex01", "Ex01DemonstrationResult"),
    "Ex01InvalidPersonPayload": ("examples._models.ex01", "Ex01InvalidPersonPayload"),
    "Ex01RunDemonstrationCommand": ("examples._models.ex01", "Ex01RunDemonstrationCommand"),
    "Ex01User": ("examples._models.ex01", "Ex01User"),
    "Ex01ValidPersonPayload": ("examples._models.ex01", "Ex01ValidPersonPayload"),
    "Ex01r": ("examples.ex_01_flext_result", "Ex01r"),
    "Ex02CacheService": ("examples._models.ex02", "Ex02CacheService"),
    "Ex02DatabaseService": ("examples._models.ex02", "Ex02DatabaseService"),
    "Ex02EmailService": ("examples._models.ex02", "Ex02EmailService"),
    "Ex02FlextSettings": ("examples.ex_02_flext_settings", "Ex02FlextSettings"),
    "Ex02TestConfig": ("examples._models.ex02", "Ex02TestConfig"),
    "Ex03Email": ("examples._models.ex03", "Ex03Email"),
    "Ex03FlextLogger": ("examples.ex_03_flext_logger", "Ex03FlextLogger"),
    "Ex03Money": ("examples._models.ex03", "Ex03Money"),
    "Ex03Order": ("examples._models.ex03", "Ex03Order"),
    "Ex03OrderItem": ("examples._models.ex03", "Ex03OrderItem"),
    "Ex03User": ("examples._models.ex03", "Ex03User"),
    "Ex04AutoCommand": ("examples._models.ex04", "Ex04AutoCommand"),
    "Ex04CreateUser": ("examples._models.ex04", "Ex04CreateUser"),
    "Ex04DeleteUser": ("examples._models.ex04", "Ex04DeleteUser"),
    "Ex04FailingDelete": ("examples._models.ex04", "Ex04FailingDelete"),
    "Ex04FlextDispatcher": ("examples.ex_04_flext_dispatcher", "Ex04FlextDispatcher"),
    "Ex04GetUser": ("examples._models.ex04", "Ex04GetUser"),
    "Ex04NoSubscriberEvent": ("examples._models.ex04", "Ex04NoSubscriberEvent"),
    "Ex04Ping": ("examples._models.ex04", "Ex04Ping"),
    "Ex04UnknownQuery": ("examples._models.ex04", "Ex04UnknownQuery"),
    "Ex04UserCreated": ("examples._models.ex04", "Ex04UserCreated"),
    "Ex05BadProcessor": ("examples._models.ex05", "Ex05BadProcessor"),
    "Ex05FlextMixins": ("examples.ex_05_flext_mixins", "Ex05FlextMixins"),
    "Ex05GoodProcessor": ("examples._models.ex05", "Ex05GoodProcessor"),
    "Ex05HandlerBad": ("examples._models.ex05", "Ex05HandlerBad"),
    "Ex05HandlerLike": ("examples._models.ex05", "Ex05HandlerLike"),
    "Ex05StatusEnum": ("examples._models.ex05", "Ex05StatusEnum"),
    "Ex05UserModel": ("examples._models.ex05", "Ex05UserModel"),
    "Ex06FlextContext": ("examples.ex_06_flext_context", "Ex06FlextContext"),
    "Ex07CreateUserCommand": ("examples._models.ex07", "Ex07CreateUserCommand"),
    "Ex07DemoPlugin": ("examples._models.ex07", "Ex07DemoPlugin"),
    "Ex07FlextExceptions": ("examples.ex_07_flext_exceptions", "Ex07FlextExceptions"),
    "Ex07GetUserQuery": ("examples._models.ex07", "Ex07GetUserQuery"),
    "Ex07UserCreatedEvent": ("examples._models.ex07", "Ex07UserCreatedEvent"),
    "Ex08FlextContainer": ("examples.ex_08_flext_container", "Ex08FlextContainer"),
    "Ex08Order": ("examples._models.ex08", "Ex08Order"),
    "Ex08User": ("examples._models.ex08", "Ex08User"),
    "Ex09FlextDecorators": ("examples.ex_09_flext_decorators", "Ex09FlextDecorators"),
    "Ex10CommandBusStub": ("examples._models.ex10", "Ex10CommandBusStub"),
    "Ex10ContextPayload": ("examples._models.ex10", "Ex10ContextPayload"),
    "Ex10DerivedMessage": ("examples._models.ex10", "Ex10DerivedMessage"),
    "Ex10Entity": ("examples._models.ex10", "Ex10Entity"),
    "Ex10FlextHandlers": ("examples.ex_10_flext_handlers", "Ex10FlextHandlers"),
    "Ex10Message": ("examples._models.ex10", "Ex10Message"),
    "Ex10ProcessorBad": ("examples._models.ex10", "Ex10ProcessorBad"),
    "Ex10ProcessorGood": ("examples._models.ex10", "Ex10ProcessorGood"),
    "Ex10ProtocolHandler": ("examples._models.ex10", "Ex10ProtocolHandler"),
    "Ex10ServiceStub": ("examples._models.ex10", "Ex10ServiceStub"),
    "Ex11CommandBusStub": ("examples._models.ex11", "Ex11CommandBusStub"),
    "Ex11EntityStub": ("examples._models.ex11", "Ex11EntityStub"),
    "Ex11FlextService": ("examples.ex_11_flext_service", "Ex11FlextService"),
    "Ex11HandlerLike": ("examples._models.ex11", "Ex11HandlerLike"),
    "Ex11HandlerLikeService": ("examples._models.ex11", "Ex11HandlerLikeService"),
    "Ex11Payload": ("examples._models.ex11", "Ex11Payload"),
    "Ex11ProcessorProtocolBad": ("examples._models.ex11", "Ex11ProcessorProtocolBad"),
    "Ex11ProcessorProtocolGood": ("examples._models.ex11", "Ex11ProcessorProtocolGood"),
    "Ex12CommandA": ("examples._models.ex12", "Ex12CommandA"),
    "Ex12CommandB": ("examples._models.ex12", "Ex12CommandB"),
    "Ex12FlextRegistry": ("examples.ex_12_flext_registry", "Ex12FlextRegistry"),
    "Ex14CreateUserCommand": ("examples._models.ex14", "Ex14CreateUserCommand"),
    "Ex14GetUserQuery": ("examples._models.ex14", "Ex14GetUserQuery"),
    "Ex14UserDTO": ("examples._models.ex14", "Ex14UserDTO"),
    "ExConfigAppConfig": ("examples._models.exconfig", "ExConfigAppConfig"),
    "Examples": ("examples.shared", "Examples"),
    "FlextCoreExampleModels": ("examples.models", "FlextCoreExampleModels"),
    "MigrationService": ("examples.logging_config_once_pattern", "MigrationService"),
    "SharedHandle": ("examples._models.shared", "SharedHandle"),
    "SharedPerson": ("examples._models.shared", "SharedPerson"),
    "UserInput": ("examples.models", "UserInput"),
    "UserProfile": ("examples.models", "UserProfile"),
    "d": ("examples.ex_09_flext_decorators", "Ex09FlextDecorators"),
    "e": ("examples.ex_07_flext_exceptions", "Ex07FlextExceptions"),
    "em": ("examples.models", "em"),
    "h": ("examples.ex_10_flext_handlers", "Ex10FlextHandlers"),
    "m": ("examples.models", "FlextCoreExampleModels"),
    "main": ("examples.logging_config_once_pattern", "main"),
    "r": ("examples._models.ex01", "Ex01DemonstrationResult"),
    "s": ("examples._models.ex02", "Ex02DatabaseService"),
    "x": ("examples.ex_05_flext_mixins", "Ex05FlextMixins"),
}

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
    "FlextCoreExampleModels",
    "MigrationService",
    "SharedHandle",
    "SharedPerson",
    "UserInput",
    "UserProfile",
    "d",
    "e",
    "em",
    "h",
    "m",
    "main",
    "r",
    "s",
    "x",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
