# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_cli import cli
    from flext_infra import docs_main, infra, main
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
    from pydantic_core import from_json, to_json, to_jsonable_python

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
    "docs_main",
    "fixtures",
    "from_json",
    "infra",
    "install_local_packages",
    "integration",
    "lazy_attribute",
    "load_infra_report",
    "main",
    "settings",
    "split_csv",
    "to_json",
    "to_jsonable_python",
    "unit",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".benchmark": ("benchmark",),
            ".fixtures": ("fixtures",),
            ".integration": ("integration",),
            ".unit": ("unit",),
            "flext_cli": ("cli",),
            "flext_core": ("core", "lazy_attribute"),
            "flext_infra": ("docs_main", "infra", "main"),
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
            "pydantic_core": ("from_json", "to_json", "to_jsonable_python"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
