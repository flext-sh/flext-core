# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples package."""

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

    from . import _models, _shared_parts
    from .constants import c
    from .models import ExamplesFlextModels, m
    from .protocols import p
    from .shared import ExamplesFlextShared
    from .typings import t
    from .utilities import u


__all__: tuple[str, ...] = (
    "ExamplesFlextModels",
    "ExamplesFlextShared",
    "_models",
    "_shared_parts",
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
            "._models": ("_models",),
            "._shared_parts": ("_shared_parts",),
            ".constants": ("c",),
            ".models": ("ExamplesFlextModels", "m"),
            ".protocols": ("p",),
            ".shared": ("ExamplesFlextShared",),
            ".typings": ("t",),
            ".utilities": ("u",),
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
