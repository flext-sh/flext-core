# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

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

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "TestsFlextConstants": ".constants",
        "TestsFlextModels": ".models",
        "TestsFlextProtocols": ".protocols",
        "TestsFlextServiceBase": ".base",
        "TestsFlextTypes": ".typings",
        "TestsFlextUtilities": ".utilities",
        "api": "flext_tests",
        "benchmark": ".benchmark",
        "c": ".constants",
        "d": "flext_core",
        "e": "flext_core",
        "fixtures": ".fixtures",
        "h": "flext_core",
        "integration": ".integration",
        "m": ".models",
        "p": ".protocols",
        "r": "flext_core",
        "s": ".base",
        "t": ".typings",
        "td": "flext_tests",
        "tf": "flext_tests",
        "tk": "flext_tests",
        "tm": "flext_tests",
        "u": "flext_tests",
        "unit": ".unit",
        "x": "flext_core",
    }),
    public_exports=__all__,
)
