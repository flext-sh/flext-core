# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import (
        api,
        cli,
        config,
        from_json,
        install_local_packages,
        load_infra_report,
        services,
        settings,
        to_json,
        to_jsonable_python,
    )

    from flext_core import core, lazy_attribute

    from . import benchmark, fixtures, integration, unit
    from .base import TestsFlextServiceBase, TestsFlextServiceBase as s
    from .constants import TestsFlextConstants, TestsFlextConstants as c
    from .models import TestsFlextModels
    from .protocols import TestsFlextProtocols
    from .typings import TestsFlextTypes
    from .utilities import TestsFlextUtilities, TestsFlextUtilities as u
__all__: tuple[str, ...] = (
    "TestsFlextConstants",
    "TestsFlextModels",
    "TestsFlextProtocols",
    "TestsFlextServiceBase",
    "TestsFlextTypes",
    "TestsFlextUtilities",
    "api",
    "benchmark",
    "c",
    "cli",
    "config",
    "core",
    "fixtures",
    "from_json",
    "install_local_packages",
    "integration",
    "lazy_attribute",
    "load_infra_report",
    "s",
    "services",
    "settings",
    "to_json",
    "to_jsonable_python",
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
            ".models": ("TestsFlextModels",),
            ".protocols": ("TestsFlextProtocols",),
            ".typings": ("TestsFlextTypes",),
            ".unit": ("unit",),
            ".utilities": ("TestsFlextUtilities", "u"),
            "flext_core": ("core", "lazy_attribute"),
            "flext_tests": (
                "api",
                "cli",
                "config",
                "from_json",
                "install_local_packages",
                "load_infra_report",
                "services",
                "settings",
                "to_json",
                "to_jsonable_python",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
