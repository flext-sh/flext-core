# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import api

    from . import benchmark, fixtures, integration, unit


__all__: tuple[str, ...] = ("api", "benchmark", "fixtures", "integration", "unit")

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".benchmark": ("benchmark",),
            ".fixtures": ("fixtures",),
            ".integration": ("integration",),
            ".unit": ("unit",),
            "flext_tests": ("api",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
