# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_cli import cli, main
    from flext_tests import (
        active_rules,
        api,
        config,
        discover_repository_root,
        install_local_packages,
        load_infra_report,
        settings,
        split_csv,
    )

    from flext_core import core, lazy_attribute

    from . import benchmark, fixtures, integration, unit
__all__: tuple[str, ...] = (
    "active_rules",
    "api",
    "benchmark",
    "cli",
    "config",
    "core",
    "discover_repository_root",
    "fixtures",
    "install_local_packages",
    "integration",
    "lazy_attribute",
    "load_infra_report",
    "main",
    "settings",
    "split_csv",
    "unit",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".benchmark": ("benchmark",),
            ".fixtures": ("fixtures",),
            ".integration": ("integration",),
            ".unit": ("unit",),
            "flext_cli": ("cli", "main"),
            "flext_core": ("core", "lazy_attribute"),
            "flext_tests": (
                "active_rules",
                "api",
                "config",
                "discover_repository_root",
                "install_local_packages",
                "load_infra_report",
                "settings",
                "split_csv",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
