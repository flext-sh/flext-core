# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if TYPE_CHECKING:
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
    from examples._shared_parts.shared_part_01 import ExamplesFlextSharedBase
    from examples.constants import c
    from examples.ex_01_flext_result import Ex01r
    from examples.ex_01_flext_result_helpers import Ex01ResultAdvancedSections
    from examples.ex_02_flext_settings import Ex02FlextSettings
    from examples.ex_02_flext_settings_helpers import Ex02FlextSettingsFieldChecks
    from examples.ex_03_flext_logger import Ex03FlextLogger
    from examples.ex_04_flext_dispatcher import Ex04DispatchDsl
    from examples.ex_05_flext_mixins import Ex05FlextMixins
    from examples.ex_06_flext_context import Ex06FlextContext
    from examples.ex_07_flext_exceptions import Ex07FlextExceptions
    from examples.ex_07_flext_exceptions_helpers import Ex07FlextExceptionSubclasses
    from examples.ex_08_container_lifecycle import Ex08ContainerLifecycle
    from examples.ex_08_container_registration import Ex08ContainerRegistration
    from examples.ex_08_container_scoped import Ex08ContainerScoped
    from examples.ex_08_flext_container import Ex08FlextContainer
    from examples.ex_09_flext_decorators import Ex09FlextDecorators
    from examples.ex_10_flext_handlers import Ex10FlextHandlers
    from examples.ex_11_flext_service import ExampleService
    from examples.ex_12_flext_registry import Ex12RegistryDsl
    from examples.ex_12_registry_flow import Ex12RegistryFlow
    from examples.ex_12_registry_plugins import Ex12RegistryPlugins
    from examples.ex_12_registry_support import ProtocolHandler
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
    from flext_core._root_typing_parts.facades import d, e, h, r, s, x
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._models",
        "._shared_parts",
    ),
    build_lazy_import_map(
        {
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
            ".ex_01_flext_result": ("Ex01r",),
            ".ex_01_flext_result_helpers": ("Ex01ResultAdvancedSections",),
            ".ex_02_flext_settings": ("Ex02FlextSettings",),
            ".ex_02_flext_settings_helpers": ("Ex02FlextSettingsFieldChecks",),
            ".ex_03_flext_logger": ("Ex03FlextLogger",),
            ".ex_04_flext_dispatcher": ("Ex04DispatchDsl",),
            ".ex_05_flext_mixins": ("Ex05FlextMixins",),
            ".ex_06_flext_context": ("Ex06FlextContext",),
            ".ex_07_flext_exceptions": ("Ex07FlextExceptions",),
            ".ex_07_flext_exceptions_helpers": ("Ex07FlextExceptionSubclasses",),
            ".ex_08_container_lifecycle": ("Ex08ContainerLifecycle",),
            ".ex_08_container_registration": ("Ex08ContainerRegistration",),
            ".ex_08_container_scoped": ("Ex08ContainerScoped",),
            ".ex_08_flext_container": ("Ex08FlextContainer",),
            ".ex_09_flext_decorators": ("Ex09FlextDecorators",),
            ".ex_10_flext_handlers": ("Ex10FlextHandlers",),
            ".ex_11_flext_service": ("ExampleService",),
            ".ex_12_flext_registry": ("Ex12RegistryDsl",),
            ".ex_12_registry_flow": ("Ex12RegistryFlow",),
            ".ex_12_registry_plugins": ("Ex12RegistryPlugins",),
            ".ex_12_registry_support": ("ProtocolHandler",),
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
            "flext_core._root_typing_parts.facades": (
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


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
