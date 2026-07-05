# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples._exports import EXAMPLES_LAZY_IMPORTS
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from examples import _models as _models, _shared_parts as _shared_parts
    from examples._models.errors import (
        ExamplesFlextModelsErrors as ExamplesFlextModelsErrors,
    )
    from examples._models.ex00 import (
        ExamplesFlextModelsEx00 as ExamplesFlextModelsEx00,
    )
    from examples._models.ex01 import (
        ExamplesFlextModelsEx01 as ExamplesFlextModelsEx01,
    )
    from examples._models.ex02 import (
        ExamplesFlextModelsEx02 as ExamplesFlextModelsEx02,
    )
    from examples._models.ex03 import (
        ExamplesFlextModelsEx03 as ExamplesFlextModelsEx03,
    )
    from examples._models.ex04 import (
        ExamplesFlextModelsEx04 as ExamplesFlextModelsEx04,
    )
    from examples._models.ex05 import (
        ExamplesFlextModelsEx05 as ExamplesFlextModelsEx05,
    )
    from examples._models.ex07 import (
        ExamplesFlextModelsEx07 as ExamplesFlextModelsEx07,
    )
    from examples._models.ex08 import (
        ExamplesFlextModelsEx08 as ExamplesFlextModelsEx08,
    )
    from examples._models.ex10 import (
        ExamplesFlextModelsEx10 as ExamplesFlextModelsEx10,
    )
    from examples._models.ex11 import (
        ExamplesFlextModelsEx11 as ExamplesFlextModelsEx11,
    )
    from examples._models.ex12 import (
        ExamplesFlextModelsEx12 as ExamplesFlextModelsEx12,
    )
    from examples._models.ex14 import (
        ExamplesFlextModelsEx14 as ExamplesFlextModelsEx14,
    )
    from examples._models.output import (
        ExamplesFlextModelsOutput as ExamplesFlextModelsOutput,
    )
    from examples._models.shared import (
        ExamplesFlextSharedHandle as ExamplesFlextSharedHandle,
        ExamplesFlextSharedPerson as ExamplesFlextSharedPerson,
    )
    from examples._shared_parts.shared_part_01 import (
        ExamplesFlextSharedBase as ExamplesFlextSharedBase,
    )
    from examples.constants import c as c
    from examples.ex_01_flext_result import Ex01r as Ex01r
    from examples.ex_01_flext_result_helpers import (
        Ex01ResultAdvancedSections as Ex01ResultAdvancedSections,
    )
    from examples.ex_02_flext_settings import (
        Ex02FlextSettings as Ex02FlextSettings,
    )
    from examples.ex_02_flext_settings_helpers import (
        Ex02FlextSettingsFieldChecks as Ex02FlextSettingsFieldChecks,
    )
    from examples.ex_03_flext_logger import Ex03FlextLogger as Ex03FlextLogger
    from examples.ex_04_flext_dispatcher import Ex04DispatchDsl as Ex04DispatchDsl
    from examples.ex_05_flext_mixins import Ex05FlextMixins as Ex05FlextMixins
    from examples.ex_06_flext_context import Ex06FlextContext as Ex06FlextContext
    from examples.ex_07_flext_exceptions import (
        Ex07FlextExceptions as Ex07FlextExceptions,
    )
    from examples.ex_07_flext_exceptions_helpers import (
        Ex07FlextExceptionSubclasses as Ex07FlextExceptionSubclasses,
    )
    from examples.ex_08_container_lifecycle import (
        Ex08ContainerLifecycle as Ex08ContainerLifecycle,
    )
    from examples.ex_08_container_registration import (
        Ex08ContainerRegistration as Ex08ContainerRegistration,
    )
    from examples.ex_08_container_scoped import (
        Ex08ContainerScoped as Ex08ContainerScoped,
    )
    from examples.ex_08_flext_container import (
        Ex08FlextContainer as Ex08FlextContainer,
    )
    from examples.ex_09_flext_decorators import (
        Ex09FlextDecorators as Ex09FlextDecorators,
    )
    from examples.ex_10_flext_handlers import (
        Ex10FlextHandlers as Ex10FlextHandlers,
    )
    from examples.ex_11_flext_service import ExampleService as ExampleService
    from examples.ex_12_flext_registry import Ex12RegistryDsl as Ex12RegistryDsl
    from examples.ex_12_registry_flow import Ex12RegistryFlow as Ex12RegistryFlow
    from examples.ex_12_registry_plugins import (
        Ex12RegistryPlugins as Ex12RegistryPlugins,
    )
    from examples.ex_12_registry_support import ProtocolHandler as ProtocolHandler
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
    from flext_core import (
        d as d,
        e as e,
        h as h,
        r as r,
        s as s,
        x as x,
    )

_LAZY_IMPORTS = EXAMPLES_LAZY_IMPORTS

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


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
