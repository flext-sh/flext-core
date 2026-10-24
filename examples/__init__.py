# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from examples.constants import c
    from examples.ex_05_flext_mixins import ExamplesFlextMixins
    from examples.ex_06_flext_context import ExamplesFlextContext
    from examples.models import ExamplesFlextModels, m
    from examples.protocols import p
    from examples.shared import ExamplesFlextShared
    from examples.typings import t
    from examples.utilities import u
    from flext_core import d, e, h, r, s, x


__all__: tuple[str, ...] = (
    "ExamplesFlextContext",
    "ExamplesFlextMixins",
    "ExamplesFlextModels",
    "ExamplesFlextShared",
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
        "ExamplesFlextContext": ".ex_06_flext_context",
        "ExamplesFlextMixins": ".ex_05_flext_mixins",
        "ExamplesFlextModels": ".models",
        "ExamplesFlextShared": ".shared",
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
