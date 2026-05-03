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
    from examples._models.errors import ExamplesFlextModelsErrors
    from examples._models.ex00 import ExamplesFlextModelsEx00
    from examples._models.ex01 import ExamplesFlextModelsEx01
    from examples._models.ex02 import ExamplesFlextModelsEx02
    from examples._models.ex03 import ExamplesFlextModelsEx03
    from examples._models.ex04 import ExamplesFlextModelsEx04
    from examples._models.ex05 import ExamplesFlextModelsEx05
    from examples._models.ex07 import ExamplesFlextModelsEx07
    from examples._models.ex08 import ExamplesFlextModelsEx08
    from examples._models.ex10 import ExamplesFlextModelsEx10
    from examples._models.ex11 import ExamplesFlextModelsEx11
    from examples._models.ex12 import ExamplesFlextModelsEx12
    from examples._models.ex14 import ExamplesFlextModelsEx14
    from examples._models.output import ExamplesFlextModelsOutput
    from examples._models.shared import (
        ExamplesFlextSharedHandle,
        ExamplesFlextSharedPerson,
    )
    from examples.constants import c
    from examples.ex_01_flext_result import Ex01r
    from examples.ex_02_flext_settings import Ex02FlextSettings
    from examples.ex_04_flext_dispatcher import Ex04DispatchDsl
    from examples.ex_05_flext_mixins import Ex05FlextMixins
    from examples.ex_06_flext_context import Ex06FlextContext
    from examples.ex_07_flext_exceptions import Ex07FlextExceptions
    from examples.ex_08_flext_container import Ex08FlextContainer
    from examples.ex_09_flext_decorators import Ex09FlextDecorators
    from examples.ex_10_flext_handlers import Ex10FlextHandlers
    from examples.ex_11_flext_service import ExampleService
    from examples.ex_12_flext_registry import Ex12RegistryDsl
    from examples.logging_config_once_pattern import (
        ExamplesFlextDatabaseService,
        ExamplesFlextMigrationService,
    )
    from examples.models import ExamplesFlextModels, m
    from examples.protocols import p
    from examples.settings import ExamplesSettings
    from examples.shared import ExamplesFlextShared
    from examples.typings import ExamplesFlextTypes, t
    from examples.utilities import u
    from flext_core import d, e, h, r, s, x
_LAZY_IMPORTS = merge_lazy_imports(
    ("._models",),
    build_lazy_import_map(
        {
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
            ".constants": ("c",),
            ".ex_01_flext_result": ("Ex01r",),
            ".ex_02_flext_settings": ("Ex02FlextSettings",),
            ".ex_04_flext_dispatcher": ("Ex04DispatchDsl",),
            ".ex_05_flext_mixins": ("Ex05FlextMixins",),
            ".ex_06_flext_context": ("Ex06FlextContext",),
            ".ex_07_flext_exceptions": ("Ex07FlextExceptions",),
            ".ex_08_flext_container": ("Ex08FlextContainer",),
            ".ex_09_flext_decorators": ("Ex09FlextDecorators",),
            ".ex_10_flext_handlers": ("Ex10FlextHandlers",),
            ".ex_11_flext_service": (
                "ExampleService",
            ),
            ".ex_12_flext_registry": ("Ex12RegistryDsl",),
            ".logging_config_once_pattern": (
                "ExamplesFlextDatabaseService",
                "ExamplesFlextMigrationService",
            ),
            ".models": (
                "ExamplesFlextModels",
                "m",
            ),
            ".protocols": ("p",),
            ".settings": ("ExamplesSettings",),
            ".shared": ("ExamplesFlextShared",),
            ".typings": (
                "ExamplesFlextTypes",
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
    "Ex01r",
    "Ex02FlextSettings",
    "Ex04DispatchDsl",
    "Ex05FlextMixins",
    "Ex06FlextContext",
    "Ex07FlextExceptions",
    "Ex08FlextContainer",
    "Ex09FlextDecorators",
    "Ex10FlextHandlers",
    "Ex12RegistryDsl",
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
    "ExamplesFlextSharedHandle",
    "ExamplesFlextSharedPerson",
    "ExamplesFlextTypes",
    "ExamplesSettings",
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
