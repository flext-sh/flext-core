# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import api, core, td, tf, tk, tm

    from flext_core import config, d, e, h, r, settings, x
    from tests.base import TestsFlextServiceBase, s
    from tests.constants import TestsFlextConstants, c
    from tests.models import TestsFlextModels, m
    from tests.protocols import TestsFlextProtocols, p
    from tests.typings import TestsFlextTypes, t
    from tests.utilities import TestsFlextUtilities, u


__all__: tuple[str, ...] = (
    "TestsFlextConstants",
    "TestsFlextModels",
    "TestsFlextProtocols",
    "TestsFlextServiceBase",
    "TestsFlextTypes",
    "TestsFlextUtilities",
    "api",
    "c",
    "config",
    "core",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "settings",
    "t",
    "td",
    "tf",
    "tk",
    "tm",
    "u",
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
        "c": ".constants",
        "config": "flext_core",
        "core": "flext_tests",
        "d": "flext_core",
        "e": "flext_core",
        "h": "flext_core",
        "m": ".models",
        "p": ".protocols",
        "r": "flext_core",
        "s": ".base",
        "settings": "flext_core",
        "t": ".typings",
        "td": "flext_tests",
        "tf": "flext_tests",
        "tk": "flext_tests",
        "tm": "flext_tests",
        "u": ".utilities",
        "x": "flext_core",
    }),
    public_exports=__all__,
)
