# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if _t.TYPE_CHECKING:
    from flext_cli import (
        d,
        e,
        h,
        r,
        reset_settings,
        s,
        settings,
        settings_factory,
        td,
        tf,
        tk,
        tm,
        tv,
        x,
    )
    from tests._constants.domain import TestsFlextCoreConstantsDomain
    from tests._constants.errors import TestsFlextCoreConstantsErrors
    from tests._constants.fixtures import TestsFlextCoreConstantsFixtures
    from tests._constants.loggings import TestsFlextCoreConstantsLoggings
    from tests._constants.other import TestsFlextCoreConstantsOther
    from tests._constants.result import TestsFlextCoreConstantsResult
    from tests._constants.services import TestsFlextCoreConstantsServices
    from tests._constants.settings import TestsFlextCoreConstantsSettings
    from tests._constants.strings import TestsFlextCoreConstantsStrings
    from tests._models.mixins import TestsFlextCoreModelsMixins
    from tests.constants import TestsFlextCoreConstants, c
    from tests.models import TestsFlextCoreModels, m
    from tests.protocols import TestsFlextCoreProtocols, p
    from tests.typings import TestsFlextCoreTypes, t
    from tests.unit._models.test_base import TestsFlextCoreModelsBase
    from tests.unit._models.test_cqrs import TestsFlextCoreModelsCqrs
    from tests.unit._models.test_entity import TestFlextModelsEntity
    from tests.unit._models.test_exception_params import TestFlextModelsExceptionParams
    from tests.unit._utilities.test_guards import TestFlextUtilitiesGuards
    from tests.unit._utilities.test_mapper import TestsFlextCoreUtilitiesMapper
    from tests.unit.base import TestsFlextCoreServiceBase
    from tests.utilities import TestsFlextCoreUtilities, u
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._constants",
        "._models",
        ".benchmark",
        ".integration",
        ".unit",
    ),
    build_lazy_import_map(
        {
            ".constants": (
                "TestsFlextCoreConstants",
                "c",
            ),
            ".models": (
                "TestsFlextCoreModels",
                "m",
            ),
            ".protocols": (
                "TestsFlextCoreProtocols",
                "p",
            ),
            ".typings": (
                "TestsFlextCoreTypes",
                "t",
            ),
            ".utilities": (
                "TestsFlextCoreUtilities",
                "u",
            ),
            "flext_cli": (
                "d",
                "e",
                "h",
                "r",
                "reset_settings",
                "s",
                "settings",
                "settings_factory",
                "td",
                "tf",
                "tk",
                "tm",
                "tv",
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
    ),
    module_name=__name__,
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__ = [
    "TestFlextModelsEntity",
    "TestFlextModelsExceptionParams",
    "TestFlextUtilitiesGuards",
    "TestsFlextCoreConstants",
    "TestsFlextCoreConstantsDomain",
    "TestsFlextCoreConstantsErrors",
    "TestsFlextCoreConstantsFixtures",
    "TestsFlextCoreConstantsLoggings",
    "TestsFlextCoreConstantsOther",
    "TestsFlextCoreConstantsResult",
    "TestsFlextCoreConstantsServices",
    "TestsFlextCoreConstantsSettings",
    "TestsFlextCoreConstantsStrings",
    "TestsFlextCoreModels",
    "TestsFlextCoreModelsBase",
    "TestsFlextCoreModelsCqrs",
    "TestsFlextCoreModelsMixins",
    "TestsFlextCoreProtocols",
    "TestsFlextCoreServiceBase",
    "TestsFlextCoreTypes",
    "TestsFlextCoreUtilities",
    "TestsFlextCoreUtilitiesMapper",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "reset_settings",
    "s",
    "settings",
    "settings_factory",
    "t",
    "td",
    "tf",
    "tk",
    "tm",
    "tv",
    "u",
    "x",
]
