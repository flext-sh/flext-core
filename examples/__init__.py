# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core import d, e, h, r, s, x

    from . import _models, _shared_parts
    from .constants import c
    from .ex_01_flext_result import Ex01r
    from .ex_01_flext_result_helpers import Ex01ResultAdvancedSections
    from .ex_02_flext_settings import Ex02FlextSettings
    from .ex_02_flext_settings_helpers import Ex02FlextSettingsFieldChecks
    from .ex_03_flext_logger import Ex03FlextLogger
    from .ex_04_flext_dispatcher import (
        Ex04DispatchDsl,
        _AuditSubscriber,
        _AutoFallbackHandler,
        _CreateUserHandler,
        _DeleteUserHandler,
        _EventSubscriber,
        _Ex04DispatchGolden,
        _FailingDeleteHandler,
        _GetUserHandler,
        _PingHandler,
    )
    from .ex_05_flext_mixins import Ex05FlextMixins
    from .ex_06_flext_context import Ex06FlextContext
    from .ex_07_flext_exceptions import Ex07FlextExceptions
    from .ex_07_flext_exceptions_helpers import Ex07FlextExceptionSubclasses
    from .ex_08_container_lifecycle import Ex08ContainerLifecycle
    from .ex_08_container_registration import Ex08ContainerRegistration
    from .ex_08_container_scoped import Ex08ContainerScoped, _WireProbe
    from .ex_08_flext_container import Ex08FlextContainer
    from .ex_09_flext_decorators import Ex09FlextDecorators
    from .ex_10_flext_handlers import Ex10FlextHandlers
    from .ex_11_flext_service import ExampleService, _EchoService, _ExampleServiceGolden
    from .ex_12_flext_registry import Ex12RegistryDsl
    from .ex_12_registry_flow import Ex12RegistryFlow
    from .ex_12_registry_plugins import Ex12RegistryPlugins
    from .ex_12_registry_support import ProtocolHandler
    from .logging_config_once_pattern import (
        ExamplesFlextDatabaseService,
        ExamplesFlextMigrationService,
    )
    from .models import ExamplesFlextModels, ExamplesFlextModels as m
    from .protocols import p
    from .settings import ExamplesSettings
    from .shared import ExamplesFlextShared
    from .typings import t
    from .utilities import u
__all__: tuple[str, ...] = (
    "Ex01ResultAdvancedSections",
    "Ex01r",
    "Ex02FlextSettings",
    "Ex02FlextSettingsFieldChecks",
    "Ex03FlextLogger",
    "Ex04DispatchDsl",
    "Ex05FlextMixins",
    "Ex06FlextContext",
    "Ex07FlextExceptionSubclasses",
    "Ex07FlextExceptions",
    "Ex08ContainerLifecycle",
    "Ex08ContainerRegistration",
    "Ex08ContainerScoped",
    "Ex08FlextContainer",
    "Ex09FlextDecorators",
    "Ex10FlextHandlers",
    "Ex12RegistryDsl",
    "Ex12RegistryFlow",
    "Ex12RegistryPlugins",
    "ExampleService",
    "ExamplesFlextDatabaseService",
    "ExamplesFlextMigrationService",
    "ExamplesFlextModels",
    "ExamplesFlextShared",
    "ExamplesSettings",
    "ProtocolHandler",
    "_AuditSubscriber",
    "_AutoFallbackHandler",
    "_CreateUserHandler",
    "_DeleteUserHandler",
    "_EchoService",
    "_EventSubscriber",
    "_Ex04DispatchGolden",
    "_ExampleServiceGolden",
    "_FailingDeleteHandler",
    "_GetUserHandler",
    "_PingHandler",
    "_WireProbe",
    "_models",
    "_shared_parts",
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
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._models": ("_models",),
            "._shared_parts": ("_shared_parts",),
            ".constants": ("c",),
            ".ex_01_flext_result": ("Ex01r",),
            ".ex_01_flext_result_helpers": ("Ex01ResultAdvancedSections",),
            ".ex_02_flext_settings": ("Ex02FlextSettings",),
            ".ex_02_flext_settings_helpers": ("Ex02FlextSettingsFieldChecks",),
            ".ex_03_flext_logger": ("Ex03FlextLogger",),
            ".ex_04_flext_dispatcher": (
                "Ex04DispatchDsl",
                "_AuditSubscriber",
                "_AutoFallbackHandler",
                "_CreateUserHandler",
                "_DeleteUserHandler",
                "_EventSubscriber",
                "_Ex04DispatchGolden",
                "_FailingDeleteHandler",
                "_GetUserHandler",
                "_PingHandler",
            ),
            ".ex_05_flext_mixins": ("Ex05FlextMixins",),
            ".ex_06_flext_context": ("Ex06FlextContext",),
            ".ex_07_flext_exceptions": ("Ex07FlextExceptions",),
            ".ex_07_flext_exceptions_helpers": ("Ex07FlextExceptionSubclasses",),
            ".ex_08_container_lifecycle": ("Ex08ContainerLifecycle",),
            ".ex_08_container_registration": ("Ex08ContainerRegistration",),
            ".ex_08_container_scoped": ("Ex08ContainerScoped", "_WireProbe"),
            ".ex_08_flext_container": ("Ex08FlextContainer",),
            ".ex_09_flext_decorators": ("Ex09FlextDecorators",),
            ".ex_10_flext_handlers": ("Ex10FlextHandlers",),
            ".ex_11_flext_service": (
                "ExampleService",
                "_EchoService",
                "_ExampleServiceGolden",
            ),
            ".ex_12_flext_registry": ("Ex12RegistryDsl",),
            ".ex_12_registry_flow": ("Ex12RegistryFlow",),
            ".ex_12_registry_plugins": ("Ex12RegistryPlugins",),
            ".ex_12_registry_support": ("ProtocolHandler",),
            ".logging_config_once_pattern": (
                "ExamplesFlextDatabaseService",
                "ExamplesFlextMigrationService",
            ),
            ".models": ("ExamplesFlextModels", "m"),
            ".protocols": ("p",),
            ".settings": ("ExamplesSettings",),
            ".shared": ("ExamplesFlextShared",),
            ".typings": ("t",),
            ".utilities": ("u",),
            "flext_core": ("d", "e", "h", "r", "s", "x"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
