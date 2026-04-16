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
    from flext_tests import d, e, h, r, s, td, tf, tk, tm, tv, x
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
    from tests.typings import T, T_co, T_contra, TestsFlextCoreTypes, t
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
            "._constants.domain": ("TestsFlextCoreConstantsDomain",),
            "._constants.errors": ("TestsFlextCoreConstantsErrors",),
            "._constants.fixtures": ("TestsFlextCoreConstantsFixtures",),
            "._constants.loggings": ("TestsFlextCoreConstantsLoggings",),
            "._constants.other": ("TestsFlextCoreConstantsOther",),
            "._constants.result": ("TestsFlextCoreConstantsResult",),
            "._constants.services": ("TestsFlextCoreConstantsServices",),
            "._constants.settings": ("TestsFlextCoreConstantsSettings",),
            "._constants.strings": ("TestsFlextCoreConstantsStrings",),
            "._models.mixins": ("TestsFlextCoreModelsMixins",),
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
                "T",
                "T_co",
                "T_contra",
                "TestsFlextCoreTypes",
                "t",
            ),
            ".unit._models.test_base": ("TestsFlextCoreModelsBase",),
            ".unit._models.test_cqrs": ("TestsFlextCoreModelsCqrs",),
            ".unit._models.test_entity": ("TestFlextModelsEntity",),
            ".unit._models.test_exception_params": ("TestFlextModelsExceptionParams",),
            ".unit._utilities.test_guards": ("TestFlextUtilitiesGuards",),
            ".unit._utilities.test_mapper": ("TestsFlextCoreUtilitiesMapper",),
            ".unit.base": ("TestsFlextCoreServiceBase",),
            ".utilities": (
                "TestsFlextCoreUtilities",
                "u",
            ),
            "flext_tests": (
                "d",
                "e",
                "h",
                "r",
                "s",
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

__all__: list[str] = [
    "T",
    "T_co",
    "T_contra",
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
    "s",
    "t",
    "td",
    "tf",
    "tk",
    "tm",
    "tv",
    "u",
    "x",
]
