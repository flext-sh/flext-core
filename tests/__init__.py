# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import api, td, tf, tk, tm, u

    from flext_core import d, e, h, r, x
    from tests import benchmark, fixtures, integration, unit
    from tests.base import TestsFlextServiceBase, s
    from tests.constants import TestsFlextConstants, c
    from tests.models import TestsFlextModels, m
    from tests.protocols import TestsFlextProtocols, p
    from tests.typings import TestsFlextTypes, t
    from tests.utilities import TestsFlextUtilities


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
    "d",
    "e",
    "fixtures",
    "h",
    "integration",
    "m",
    "p",
    "r",
    "s",
    "t",
    "td",
    "tf",
    "tk",
    "tm",
    "u",
    "unit",
    "x",
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
            ".utilities": ("TestsFlextUtilities",),
            "flext_core": ("d", "e", "h", "r", "x"),
            "flext_tests": ("api", "td", "tf", "tk", "tm", "u"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
