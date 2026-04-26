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
    from examples._models.ex02 import ExamplesFlextCoreModelsEx02
    from examples._models.ex03 import (
        Ex03Email,
        Ex03Money,
        Ex03Order,
        Ex03OrderItem,
        Ex03User,
        ExamplesFlextCoreModelsEx03,
    )
    from examples._models.ex04 import ExamplesFlextCoreModelsEx04
    from examples._models.ex05 import ExamplesFlextCoreModelsEx05
    from examples._models.ex07 import ExamplesFlextCoreModelsEx07
    from examples._models.ex08 import ExamplesFlextCoreModelsEx08
    from examples._models.ex10 import ExamplesFlextCoreModelsEx10
    from examples._models.ex11 import ExamplesFlextCoreModelsEx11
    from examples._models.ex12 import ExamplesFlextCoreModelsEx12
    from examples._models.ex14 import ExamplesFlextCoreModelsEx14
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
    from examples.settings import ExamplesSettings
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
            "._models.ex02": ("ExamplesFlextCoreModelsEx02",),
            "._models.ex03": (
                "Ex03Email",
                "Ex03Money",
                "Ex03Order",
                "Ex03OrderItem",
                "Ex03User",
                "ExamplesFlextCoreModelsEx03",
            ),
            "._models.ex04": ("ExamplesFlextCoreModelsEx04",),
            "._models.ex05": ("ExamplesFlextCoreModelsEx05",),
            "._models.ex07": ("ExamplesFlextCoreModelsEx07",),
            "._models.ex08": ("ExamplesFlextCoreModelsEx08",),
            "._models.ex10": ("ExamplesFlextCoreModelsEx10",),
            "._models.ex11": ("ExamplesFlextCoreModelsEx11",),
            "._models.ex12": ("ExamplesFlextCoreModelsEx12",),
            "._models.ex14": ("ExamplesFlextCoreModelsEx14",),
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
            ".settings": ("ExamplesSettings",),
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
        "pytest_addoption",
        "pytest_collect_file",
        "pytest_collection_modifyitems",
        "pytest_configure",
        "pytest_runtest_setup",
        "pytest_runtest_teardown",
        "pytest_sessionfinish",
        "pytest_sessionstart",
        "pytest_terminal_summary",
        "pytest_warning_recorded",
    ),
    module_name=__name__,
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__: list[str] = [
    "DatabaseService",
    "Ex01r",
    "Ex02FlextSettings",
    "Ex03Email",
    "Ex03LoggingDsl",
    "Ex03Money",
    "Ex03Order",
    "Ex03OrderItem",
    "Ex03User",
    "Ex04DispatchDsl",
    "Ex05FlextMixins",
    "Ex06FlextContext",
    "Ex07FlextExceptions",
    "Ex08FlextContainer",
    "Ex09FlextDecorators",
    "Ex10FlextHandlers",
    "Ex11FlextService",
    "Ex12RegistryDsl",
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
    "ExamplesFlextCoreShared",
    "ExamplesFlextCoreSharedHandle",
    "ExamplesFlextCoreSharedPerson",
    "ExamplesFlextCoreTypes",
    "ExamplesSettings",
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
