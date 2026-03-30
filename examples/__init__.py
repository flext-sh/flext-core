# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Examples package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from examples import (
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
    from examples._models import *
    from examples.ex_01_flext_result import *
    from examples.ex_02_flext_settings import *
    from examples.ex_03_flext_logger import *
    from examples.ex_04_flext_dispatcher import *
    from examples.ex_05_flext_mixins import *
    from examples.ex_06_flext_context import *
    from examples.ex_07_flext_exceptions import *
    from examples.ex_08_flext_container import *
    from examples.ex_09_flext_decorators import *
    from examples.ex_10_flext_handlers import *
    from examples.ex_11_flext_service import *
    from examples.ex_12_flext_registry import *
    from examples.logging_config_once_pattern import *
    from examples.models import *
    from examples.shared import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "DatabaseService": "examples.logging_config_once_pattern",
    "Ex00UserInput": "examples._models.ex00",
    "Ex00UserProfile": "examples._models.ex00",
    "Ex01DemonstrationResult": "examples._models.ex01",
    "Ex01InvalidPersonPayload": "examples._models.ex01",
    "Ex01RunDemonstrationCommand": "examples._models.ex01",
    "Ex01User": "examples._models.ex01",
    "Ex01ValidPersonPayload": "examples._models.ex01",
    "Ex01r": "examples.ex_01_flext_result",
    "Ex02CacheService": "examples._models.ex02",
    "Ex02DatabaseService": "examples._models.ex02",
    "Ex02EmailService": "examples._models.ex02",
    "Ex02FlextSettings": "examples.ex_02_flext_settings",
    "Ex02TestConfig": "examples._models.ex02",
    "Ex03Email": "examples._models.ex03",
    "Ex03FlextLogger": "examples.ex_03_flext_logger",
    "Ex03Money": "examples._models.ex03",
    "Ex03Order": "examples._models.ex03",
    "Ex03OrderItem": "examples._models.ex03",
    "Ex03User": "examples._models.ex03",
    "Ex04AutoCommand": "examples._models.ex04",
    "Ex04CreateUser": "examples._models.ex04",
    "Ex04DeleteUser": "examples._models.ex04",
    "Ex04FailingDelete": "examples._models.ex04",
    "Ex04FlextDispatcher": "examples.ex_04_flext_dispatcher",
    "Ex04GetUser": "examples._models.ex04",
    "Ex04NoSubscriberEvent": "examples._models.ex04",
    "Ex04Ping": "examples._models.ex04",
    "Ex04UnknownQuery": "examples._models.ex04",
    "Ex04UserCreated": "examples._models.ex04",
    "Ex05BadProcessor": "examples._models.ex05",
    "Ex05FlextMixins": "examples.ex_05_flext_mixins",
    "Ex05GoodProcessor": "examples._models.ex05",
    "Ex05HandlerBad": "examples._models.ex05",
    "Ex05HandlerLike": "examples._models.ex05",
    "Ex05StatusEnum": "examples._models.ex05",
    "Ex05UserModel": "examples._models.ex05",
    "Ex06FlextContext": "examples.ex_06_flext_context",
    "Ex07CreateUserCommand": "examples._models.ex07",
    "Ex07DemoPlugin": "examples._models.ex07",
    "Ex07FlextExceptions": "examples.ex_07_flext_exceptions",
    "Ex07GetUserQuery": "examples._models.ex07",
    "Ex07UserCreatedEvent": "examples._models.ex07",
    "Ex08FlextContainer": "examples.ex_08_flext_container",
    "Ex08Order": "examples._models.ex08",
    "Ex08User": "examples._models.ex08",
    "Ex09FlextDecorators": "examples.ex_09_flext_decorators",
    "Ex10CommandBusStub": "examples._models.ex10",
    "Ex10ContextPayload": "examples._models.ex10",
    "Ex10DerivedMessage": "examples._models.ex10",
    "Ex10Entity": "examples._models.ex10",
    "Ex10FlextHandlers": "examples.ex_10_flext_handlers",
    "Ex10Message": "examples._models.ex10",
    "Ex10ProcessorBad": "examples._models.ex10",
    "Ex10ProcessorGood": "examples._models.ex10",
    "Ex10ProtocolHandler": "examples._models.ex10",
    "Ex10ServiceStub": "examples._models.ex10",
    "Ex11CommandBusStub": "examples._models.ex11",
    "Ex11EntityStub": "examples._models.ex11",
    "Ex11FlextService": "examples.ex_11_flext_service",
    "Ex11HandlerLike": "examples._models.ex11",
    "Ex11HandlerLikeService": "examples._models.ex11",
    "Ex11Payload": "examples._models.ex11",
    "Ex11ProcessorProtocolBad": "examples._models.ex11",
    "Ex11ProcessorProtocolGood": "examples._models.ex11",
    "Ex12CommandA": "examples._models.ex12",
    "Ex12CommandB": "examples._models.ex12",
    "Ex12FlextRegistry": "examples.ex_12_flext_registry",
    "Ex14CreateUserCommand": "examples._models.ex14",
    "Ex14GetUserQuery": "examples._models.ex14",
    "Ex14UserDTO": "examples._models.ex14",
    "ExConfigAppConfig": "examples._models.exconfig",
    "Examples": "examples.shared",
    "FlextCoreExampleModels": "examples.models",
    "MigrationService": "examples.logging_config_once_pattern",
    "SharedHandle": "examples._models.shared",
    "SharedPerson": "examples._models.shared",
    "UserInput": "examples.models",
    "UserProfile": "examples.models",
    "_models": "examples._models",
    "em": "examples.models",
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
    "exconfig": "examples._models.exconfig",
    "logging_config_once_pattern": "examples.logging_config_once_pattern",
    "m": ["examples.models", "FlextCoreExampleModels"],
    "main": "examples.logging_config_once_pattern",
    "models": "examples.models",
    "shared": "examples.shared",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, sorted(_LAZY_IMPORTS))
