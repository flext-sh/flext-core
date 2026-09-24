# AUTO-GENERATED FILE — Regenerate with: make gen
"""Scripts package."""

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
        discover_repository_root,
        install_local_packages,
        load_infra_report,
        split_csv,
        td,
        tf,
        tk,
        tm,
        tv,
    )

    from flext_core import config, core, d, e, h, lazy_attribute, r, s, settings, x

    from .constants import ScriptsFlextConstants, c
    from .models import ScriptsFlextModels, m
    from .protocols import ScriptsFlextProtocols, p
    from .typings import ScriptsFlextTypes, t
    from .utilities import ScriptsFlextUtilities, u


__all__: tuple[str, ...] = (
    "ScriptsFlextConstants",
    "ScriptsFlextModels",
    "ScriptsFlextProtocols",
    "ScriptsFlextTypes",
    "ScriptsFlextUtilities",
    "active_rules",
    "api",
    "c",
    "cli",
    "config",
    "core",
    "d",
    "discover_repository_root",
    "docs_main",
    "e",
    "h",
    "infra",
    "install_local_packages",
    "lazy_attribute",
    "load_infra_report",
    "m",
    "main",
    "p",
    "r",
    "s",
    "settings",
    "split_csv",
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
            ".constants": ("ScriptsFlextConstants", "c"),
            ".models": ("ScriptsFlextModels", "m"),
            ".protocols": ("ScriptsFlextProtocols", "p"),
            ".typings": ("ScriptsFlextTypes", "t"),
            ".utilities": ("ScriptsFlextUtilities", "u"),
            "flext_cli": ("cli", "main"),
            "flext_core": (
                "config",
                "core",
                "d",
                "e",
                "h",
                "lazy_attribute",
                "r",
                "s",
                "settings",
                "x",
            ),
            "flext_infra": ("docs_main", "infra"),
            "flext_tests": (
                "active_rules",
                "api",
                "discover_repository_root",
                "install_local_packages",
                "load_infra_report",
                "split_csv",
                "td",
                "tf",
                "tk",
                "tm",
                "tv",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
