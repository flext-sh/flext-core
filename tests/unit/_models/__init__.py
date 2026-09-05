# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests.unit. Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import c, d, e, h, m, p, r, s, t, td, tf, tk, tm, tv, u, x

    from .test_base import Sample, SampleValue, TestsFlextCoreBase
    from .test_cqrs import TestsFlextCoreCqrs
    from .test_enforcement_sources import TestsFlextCoreEnforcementSources
    from .test_entity import TestsFlextCoreEntity
    from .test_exception_params_core import TestsFlextModelsExceptionParamsCore
    from .test_exception_params_operations import (
        TestsFlextCoreExceptionParamsOperations,
    )
    from .test_exception_params_resources import (
        TestsFlextModelsExceptionParamsResources,
    )
__all__: tuple[str, ...] = (
    "Sample",
    "SampleValue",
    "TestsFlextCoreBase",
    "TestsFlextCoreCqrs",
    "TestsFlextCoreEnforcementSources",
    "TestsFlextCoreEntity",
    "TestsFlextCoreExceptionParamsOperations",
    "TestsFlextModelsExceptionParamsCore",
    "TestsFlextModelsExceptionParamsResources",
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
            ".test_base": ("Sample", "SampleValue", "TestsFlextCoreBase"),
            ".test_cqrs": ("TestsFlextCoreCqrs",),
            ".test_enforcement_sources": ("TestsFlextCoreEnforcementSources",),
            ".test_entity": ("TestsFlextCoreEntity",),
            ".test_exception_params_core": ("TestsFlextModelsExceptionParamsCore",),
            ".test_exception_params_operations": (
                "TestsFlextCoreExceptionParamsOperations",
            ),
            ".test_exception_params_resources": (
                "TestsFlextModelsExceptionParamsResources",
            ),
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
