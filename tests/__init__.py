# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_cli import cli, main
    from flext_infra import docs_main, infra
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
    from .base import TestsFlextServiceBase, TestsFlextServiceBase as s
    from .constants import TestsFlextConstants, TestsFlextConstants as c
    from .models import TestsFlextModels, m
    from .protocols import TestsFlextProtocols, p
    from .typings import TestsFlextTypes, t
    from .utilities import TestsFlextUtilities, TestsFlextUtilities as u


__all__: tuple[str, ...] = (
    "TestsFlextConstants",
    "TestsFlextModels",
    "TestsFlextProtocols",
    "TestsFlextServiceBase",
    "TestsFlextTypes",
    "TestsFlextUtilities",
    "active_rules",
    "api",
    "benchmark",
    "c",
    "cli",
    "config",
    "core",
    "discover_repository_root",
    "docs_main",
    "fixtures",
    "infra",
    "install_local_packages",
    "integration",
    "lazy_attribute",
    "load_infra_report",
    "m",
    "main",
    "p",
    "s",
    "settings",
    "split_csv",
    "t",
    "u",
    "unit",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".base": ("TestsFlextServiceBase", "s"),
            ".benchmark": ("benchmark",),
            ".constants": ("TestsFlextConstants", "c"),
            ".fixtures": ("fixtures",),
            ".integration": ("integration",),
            ".models": ("TestsFlextModels", "m"),
            ".protocols": ("TestsFlextProtocols", "p"),
            ".typings": ("TestsFlextTypes", "t"),
            ".unit": ("unit",),
            ".utilities": ("TestsFlextUtilities", "u"),
            "flext_cli": ("cli", "main"),
            "flext_core": ("core", "lazy_attribute"),
            "flext_infra": ("docs_main", "infra"),
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
