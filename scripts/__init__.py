# AUTO-GENERATED FILE — Regenerate with: make gen
"""Scripts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core import d, e, h, r, s, x
    from scripts.constants import ScriptsFlextConstants, c
    from scripts.models import ScriptsFlextModels, m
    from scripts.protocols import ScriptsFlextProtocols, p
    from scripts.typings import ScriptsFlextTypes, t
    from scripts.utilities import ScriptsFlextUtilities, u


__all__: tuple[str, ...] = (
    "ScriptsFlextConstants",
    "ScriptsFlextModels",
    "ScriptsFlextProtocols",
    "ScriptsFlextTypes",
    "ScriptsFlextUtilities",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "u",
    "x",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "ScriptsFlextConstants": ".constants",
        "ScriptsFlextModels": ".models",
        "ScriptsFlextProtocols": ".protocols",
        "ScriptsFlextTypes": ".typings",
        "ScriptsFlextUtilities": ".utilities",
        "c": ".constants",
        "d": "flext_core",
        "e": "flext_core",
        "h": "flext_core",
        "m": ".models",
        "p": ".protocols",
        "r": "flext_core",
        "s": "flext_core",
        "t": ".typings",
        "u": ".utilities",
        "x": "flext_core",
    }),
    public_exports=__all__,
)
