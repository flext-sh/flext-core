# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests.unit package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import c, d, e, h, m, p, r, s, t, td, tf, tk, tm, tv, u, x

    from . import _models as _models, _utilities as _utilities
    from .test_constants_new import TestsFlextConstantsNew
    from .test_coverage_loggings import TestsFlextCoverageLoggings
    from .test_dispatcher import TestsFlextCoreDispatcher
    from .test_mixins import TestsFlextMixins
    from .test_registry import TestsFlextCoreRegistry
    from .test_service import TestsFlextService
    from .test_settings_validation_alias import TestsFlextCoreSettingsValidationAlias
    from .test_utilities_collection_coverage_100 import (
        TestsFlextCoreUtilitiesCollection,
    )
    from .test_utilities_coverage import TestsFlextCoreUtilitiesCoverage
__all__: tuple[str, ...] = (
    "TestsFlextConstantsNew",
    "TestsFlextCoreDispatcher",
    "TestsFlextCoreRegistry",
    "TestsFlextCoreSettingsValidationAlias",
    "TestsFlextCoreUtilitiesCollection",
    "TestsFlextCoreUtilitiesCoverage",
    "TestsFlextCoverageLoggings",
    "TestsFlextMixins",
    "TestsFlextService",
    "_models",
    "_utilities",
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
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._models": ("_models",),
            "._utilities": ("_utilities",),
            ".test_constants_new": ("TestsFlextConstantsNew",),
            ".test_coverage_loggings": ("TestsFlextCoverageLoggings",),
            ".test_dispatcher": ("TestsFlextCoreDispatcher",),
            ".test_mixins": ("TestsFlextMixins",),
            ".test_registry": ("TestsFlextCoreRegistry",),
            ".test_service": ("TestsFlextService",),
            ".test_settings_validation_alias": (
                "TestsFlextCoreSettingsValidationAlias",
            ),
            ".test_utilities_collection_coverage_100": (
                "TestsFlextCoreUtilitiesCollection",
            ),
            ".test_utilities_coverage": ("TestsFlextCoreUtilitiesCoverage",),
            "flext_tests": (
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
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
