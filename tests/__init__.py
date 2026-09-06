# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core import FlextConstants
    from flext_tests import FlextTestsConstants

    from . import (
        benchmark as benchmark,
        fixtures as fixtures,
        integration as integration,
        unit as unit,
    )
    from .base import s
    from .models import m
    from .protocols import p
    from .typings import t
    from .utilities import u
__all__: tuple[str, ...] = (
    "FlextConstants",
    "FlextTestsConstants",
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
            ".base": ("s",),
            ".benchmark": ("benchmark",),
            ".fixtures": ("fixtures",),
            ".integration": ("integration",),
            ".models": ("m",),
            ".protocols": ("p",),
            ".typings": ("t",),
            ".unit": ("unit",),
            ".utilities": ("u",),
            "flext_core": ("FlextConstants",),
            "flext_tests": ("FlextTestsConstants",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
