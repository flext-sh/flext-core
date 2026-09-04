# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Constants package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import c, d, e, h, m, p, r, s, t, td, tf, tk, tm, tv, u, x

    from .domain import TestsFlextConstantsDomain
    from .errors import TestsFlextConstantsErrors
    from .fixtures import TestsFlextConstantsFixtures
    from .loggings import TestsFlextConstantsLoggings
    from .other import TestsFlextConstantsOther
    from .result import TestsFlextConstantsResult
    from .services import TestsFlextConstantsServices
    from .settings import TestsFlextConstantsSettings
__all__: tuple[str, ...] = (
    "TestsFlextConstantsDomain",
    "TestsFlextConstantsErrors",
    "TestsFlextConstantsFixtures",
    "TestsFlextConstantsLoggings",
    "TestsFlextConstantsOther",
    "TestsFlextConstantsResult",
    "TestsFlextConstantsServices",
    "TestsFlextConstantsSettings",
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
            ".domain": ("TestsFlextConstantsDomain",),
            ".errors": ("TestsFlextConstantsErrors",),
            ".fixtures": ("TestsFlextConstantsFixtures",),
            ".loggings": ("TestsFlextConstantsLoggings",),
            ".other": ("TestsFlextConstantsOther",),
            ".result": ("TestsFlextConstantsResult",),
            ".services": ("TestsFlextConstantsServices",),
            ".settings": ("TestsFlextConstantsSettings",),
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
