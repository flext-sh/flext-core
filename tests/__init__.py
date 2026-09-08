# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import benchmark as benchmark
    from . import fixtures as fixtures
    from . import integration as integration
    from . import unit as unit
    from flext_core import FlextConstants
    from flext_tests import FlextTestsConstants

    from .base import TestsFlextServiceBase, TestsFlextServiceBase as s
    from .models import TestsFlextModels, TestsFlextModels as m
    from .protocols import TestsFlextProtocols, TestsFlextProtocols as p
    from .typings import TestsFlextTypes, TestsFlextTypes as t
    from .utilities import TestsFlextUtilities, TestsFlextUtilities as u
__all__: tuple[str, ...] = (
    "FlextConstants",
    "FlextTestsConstants",
    "TestsFlextModels",
    "TestsFlextProtocols",
    "TestsFlextServiceBase",
    "TestsFlextTypes",
    "TestsFlextUtilities",
    "benchmark",
    "fixtures",
    "integration",
    "m",
    "p",
    "s",
    "t",
    "u",
    "unit",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".base": ("TestsFlextServiceBase", "s"),
            ".benchmark": ("benchmark",),
            ".fixtures": ("fixtures",),
            ".integration": ("integration",),
            ".models": ("TestsFlextModels", "m"),
            ".protocols": ("TestsFlextProtocols", "p"),
            ".typings": ("TestsFlextTypes", "t"),
            ".unit": ("unit",),
            ".utilities": ("TestsFlextUtilities", "u"),
            "flext_core": ("FlextConstants",),
            "flext_tests": ("FlextTestsConstants",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
