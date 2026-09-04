# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _models as _models
    from . import _shared_parts as _shared_parts
    from flext_core import c as _c, d, e, h, r, s, x
    from typing import TYPE_CHECKING

    from ._models.errors import ExamplesFlextModelsErrors
    from ._models.ex00 import ExamplesFlextModelsEx00
    from ._models.ex01 import ExamplesFlextModelsEx01
    from ._models.ex02 import ExamplesFlextModelsEx02
    from ._models.ex03 import ExamplesFlextModelsEx03
    from ._models.ex04 import ExamplesFlextModelsEx04
    from ._models.ex05 import ExamplesFlextModelsEx05
    from ._models.ex07 import ExamplesFlextModelsEx07
    from ._models.ex08 import ExamplesFlextModelsEx08
    from ._models.ex10 import ExamplesFlextModelsEx10
    from ._models.ex11 import ExamplesFlextModelsEx11
    from ._models.ex12 import ExamplesFlextModelsEx12
    from ._models.ex14 import ExamplesFlextModelsEx14
    from ._models.output import ExamplesFlextModelsOutput
    from ._models.shared import ExamplesFlextSharedHandle, ExamplesFlextSharedPerson
    from ._shared_parts.shared_part_01 import ExamplesFlextSharedBase
    from .constants import c
    from .ex_01_flext_result import Ex01r, main
    from .ex_01_flext_result_helpers import Ex01ResultAdvancedSections
    from .ex_02_flext_settings import Ex02FlextSettings
    from .ex_02_flext_settings_helpers import Ex02FlextSettingsFieldChecks
    from .ex_03_flext_logger import Ex03FlextLogger
    from .ex_04_flext_dispatcher import Ex04DispatchDsl
    from .ex_05_flext_mixins import Ex05FlextMixins, run
    from .ex_06_flext_context import Ex06FlextContext
    from .ex_07_flext_exceptions import Ex07FlextExceptions
    from .ex_07_flext_exceptions_helpers import Ex07FlextExceptionSubclasses
    from .ex_08_container_lifecycle import Ex08ContainerLifecycle
    from .ex_08_container_registration import Ex08ContainerRegistration
    from .ex_08_container_scoped import Ex08ContainerScoped
    from .ex_08_flext_container import Ex08FlextContainer
    from .ex_09_flext_decorators import Ex09FlextDecorators, msg, result
    from .ex_10_flext_handlers import Ex10FlextHandlers
    from .ex_11_flext_service import ExampleService
    from .ex_12_flext_registry import Ex12RegistryDsl
    from .ex_12_registry_flow import Ex12RegistryFlow
    from .ex_12_registry_plugins import Ex12RegistryPlugins
    from .ex_12_registry_support import (
        ProtocolHandler,
        as_registry_handler,
        discovered_handler,
    )
    from .logging_config_once_pattern import (
        ExamplesFlextDatabaseService,
        ExamplesFlextMigrationService,
    )
    from .models import ExamplesFlextModels, ExamplesFlextModels as m
    from .protocols import p
    from .settings import ExamplesSettings
    from .shared import ExamplesFlextShared
    from .typings import ExamplesFlextTypes, ExamplesFlextTypes as t
    from .utilities import u
__all__: tuple[str, ...] = (
    "TYPE_CHECKING",
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
    "ExamplesFlextModelsErrors",
    "ExamplesFlextModelsEx00",
    "ExamplesFlextModelsEx01",
    "ExamplesFlextModelsEx02",
    "ExamplesFlextModelsEx03",
    "ExamplesFlextModelsEx04",
    "ExamplesFlextModelsEx05",
    "ExamplesFlextModelsEx07",
    "ExamplesFlextModelsEx08",
    "ExamplesFlextModelsEx10",
    "ExamplesFlextModelsEx11",
    "ExamplesFlextModelsEx12",
    "ExamplesFlextModelsEx14",
    "ExamplesFlextModelsOutput",
    "ExamplesFlextShared",
    "ExamplesFlextSharedBase",
    "ExamplesFlextSharedHandle",
    "ExamplesFlextSharedPerson",
    "ExamplesFlextTypes",
    "ExamplesSettings",
    "ProtocolHandler",
    "_c",
    "_models",
    "_shared_parts",
    "as_registry_handler",
    "c",
    "d",
    "discovered_handler",
    "e",
    "h",
    "m",
    "main",
    "msg",
    "p",
    "r",
    "result",
    "run",
    "s",
    "t",
    "u",
    "x",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._models": ("_models",),
            "._models.errors": ("ExamplesFlextModelsErrors",),
            "._models.ex00": ("ExamplesFlextModelsEx00",),
            "._models.ex01": ("ExamplesFlextModelsEx01",),
            "._models.ex02": ("ExamplesFlextModelsEx02",),
            "._models.ex03": ("ExamplesFlextModelsEx03",),
            "._models.ex04": ("ExamplesFlextModelsEx04",),
            "._models.ex05": ("ExamplesFlextModelsEx05",),
            "._models.ex07": ("ExamplesFlextModelsEx07",),
            "._models.ex08": ("ExamplesFlextModelsEx08",),
            "._models.ex10": ("ExamplesFlextModelsEx10",),
            "._models.ex11": ("ExamplesFlextModelsEx11",),
            "._models.ex12": ("ExamplesFlextModelsEx12",),
            "._models.ex14": ("ExamplesFlextModelsEx14",),
            "._models.output": ("ExamplesFlextModelsOutput",),
            "._models.shared": (
                "ExamplesFlextSharedHandle",
                "ExamplesFlextSharedPerson",
            ),
            "._shared_parts": ("_shared_parts",),
            "._shared_parts.shared_part_01": ("ExamplesFlextSharedBase",),
            ".constants": ("c",),
            ".ex_01_flext_result": ("Ex01r", "main"),
            ".ex_01_flext_result_helpers": ("Ex01ResultAdvancedSections",),
            ".ex_02_flext_settings": ("Ex02FlextSettings",),
            ".ex_02_flext_settings_helpers": ("Ex02FlextSettingsFieldChecks",),
            ".ex_03_flext_logger": ("Ex03FlextLogger",),
            ".ex_04_flext_dispatcher": ("Ex04DispatchDsl",),
            ".ex_05_flext_mixins": ("Ex05FlextMixins", "run"),
            ".ex_06_flext_context": ("Ex06FlextContext",),
            ".ex_07_flext_exceptions": ("Ex07FlextExceptions",),
            ".ex_07_flext_exceptions_helpers": ("Ex07FlextExceptionSubclasses",),
            ".ex_08_container_lifecycle": ("Ex08ContainerLifecycle",),
            ".ex_08_container_registration": ("Ex08ContainerRegistration",),
            ".ex_08_container_scoped": ("Ex08ContainerScoped",),
            ".ex_08_flext_container": ("Ex08FlextContainer",),
            ".ex_09_flext_decorators": ("Ex09FlextDecorators", "msg", "result"),
            ".ex_10_flext_handlers": ("Ex10FlextHandlers",),
            ".ex_11_flext_service": ("ExampleService",),
            ".ex_12_flext_registry": ("Ex12RegistryDsl",),
            ".ex_12_registry_flow": ("Ex12RegistryFlow",),
            ".ex_12_registry_plugins": ("Ex12RegistryPlugins",),
            ".ex_12_registry_support": (
                "ProtocolHandler",
                "as_registry_handler",
                "discovered_handler",
            ),
            ".logging_config_once_pattern": (
                "ExamplesFlextDatabaseService",
                "ExamplesFlextMigrationService",
            ),
            ".models": ("ExamplesFlextModels", "m"),
            ".protocols": ("p",),
            ".settings": ("ExamplesSettings",),
            ".shared": ("ExamplesFlextShared",),
            ".typings": ("ExamplesFlextTypes", "t"),
            ".utilities": ("u",),
            "flext_core": ("d", "e", "h", "r", "s", "x"),
            "typing": ("TYPE_CHECKING",),
        }),
        alias_groups=MappingProxyType({"flext_core": (("_c", "c"),)}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
