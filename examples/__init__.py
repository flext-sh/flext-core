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
    from examples._models.errors import (
        ExamplesFlextModelsErrors as ExamplesFlextModelsErrors,
    )
    from examples._models.ex00 import ExamplesFlextModelsEx00 as ExamplesFlextModelsEx00
    from examples._models.ex01 import ExamplesFlextModelsEx01 as ExamplesFlextModelsEx01
    from examples._models.ex02 import ExamplesFlextModelsEx02 as ExamplesFlextModelsEx02
    from examples._models.ex03 import ExamplesFlextModelsEx03 as ExamplesFlextModelsEx03
    from examples._models.ex04 import ExamplesFlextModelsEx04 as ExamplesFlextModelsEx04
    from examples._models.ex05 import ExamplesFlextModelsEx05 as ExamplesFlextModelsEx05
    from examples._models.ex07 import ExamplesFlextModelsEx07 as ExamplesFlextModelsEx07
    from examples._models.ex08 import ExamplesFlextModelsEx08 as ExamplesFlextModelsEx08
    from examples._models.ex10 import ExamplesFlextModelsEx10 as ExamplesFlextModelsEx10
    from examples._models.ex11 import ExamplesFlextModelsEx11 as ExamplesFlextModelsEx11
    from examples._models.ex12 import ExamplesFlextModelsEx12 as ExamplesFlextModelsEx12
    from examples._models.ex14 import ExamplesFlextModelsEx14 as ExamplesFlextModelsEx14
    from examples._models.output import (
        ExamplesFlextModelsOutput as ExamplesFlextModelsOutput,
    )
    from examples._models.shared import (
        ExamplesFlextSharedHandle as ExamplesFlextSharedHandle,
        ExamplesFlextSharedPerson as ExamplesFlextSharedPerson,
    )
    from examples.constants import c as c
    from examples.ex_01_flext_result import Ex01r as Ex01r
    from examples.ex_02_flext_settings import Ex02FlextSettings as Ex02FlextSettings
    from examples.ex_03_flext_logger import Ex03FlextLogger as Ex03FlextLogger
    from examples.ex_04_flext_dispatcher import Ex04DispatchDsl as Ex04DispatchDsl
    from examples.ex_05_flext_mixins import Ex05FlextMixins as Ex05FlextMixins
    from examples.ex_06_flext_context import Ex06FlextContext as Ex06FlextContext
    from examples.ex_07_flext_exceptions import (
        Ex07FlextExceptions as Ex07FlextExceptions,
    )
    from examples.ex_08_flext_container import Ex08FlextContainer as Ex08FlextContainer
    from examples.ex_09_flext_decorators import (
        Ex09FlextDecorators as Ex09FlextDecorators,
    )
    from examples.ex_10_flext_handlers import Ex10FlextHandlers as Ex10FlextHandlers
    from examples.ex_11_flext_service import ExampleService as ExampleService
    from examples.ex_12_flext_registry import Ex12RegistryDsl as Ex12RegistryDsl
    from examples.logging_config_once_pattern import (
        ExamplesFlextDatabaseService as ExamplesFlextDatabaseService,
        ExamplesFlextMigrationService as ExamplesFlextMigrationService,
    )
    from examples.models import ExamplesFlextModels as ExamplesFlextModels, m as m
    from examples.protocols import p as p
    from examples.settings import ExamplesSettings as ExamplesSettings
    from examples.shared import ExamplesFlextShared as ExamplesFlextShared
    from examples.typings import ExamplesFlextTypes as ExamplesFlextTypes, t as t
    from examples.utilities import u as u
    from flext_core import d as d, e as e, h as h, r as r, s as s, x as x
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
            ".ex_03_flext_logger": ("Ex03FlextLogger",),
            ".ex_04_flext_dispatcher": ("Ex04DispatchDsl",),
            ".ex_05_flext_mixins": ("Ex05FlextMixins",),
            ".ex_06_flext_context": ("Ex06FlextContext",),
            ".ex_07_flext_exceptions": ("Ex07FlextExceptions",),
            ".ex_08_flext_container": ("Ex08FlextContainer",),
            ".ex_09_flext_decorators": ("Ex09FlextDecorators",),
            ".ex_10_flext_handlers": ("Ex10FlextHandlers",),
            ".ex_11_flext_service": ("ExampleService",),
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
    "Ex03FlextLogger",
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
    "ExamplesFlextShared",
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
