# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples. Protocols package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from examples._protocols.ex00 import ExamplesFlextProtocolsEx00
    from examples._protocols.ex01 import ExamplesFlextProtocolsEx01
    from examples._protocols.ex02 import ExamplesFlextProtocolsEx02
    from examples._protocols.ex03 import ExamplesFlextProtocolsEx03
    from examples._protocols.ex04 import ExamplesFlextProtocolsEx04
    from examples._protocols.ex05 import ExamplesFlextProtocolsEx05
    from examples._protocols.ex07 import ExamplesFlextProtocolsEx07
    from examples._protocols.ex08 import ExamplesFlextProtocolsEx08
    from examples._protocols.ex10 import ExamplesFlextProtocolsEx10
    from examples._protocols.ex11 import ExamplesFlextProtocolsEx11
    from examples._protocols.ex12 import ExamplesFlextProtocolsEx12
    from examples._protocols.ex14 import ExamplesFlextProtocolsEx14
    from examples._protocols.handle import ExamplesFlextProtocolsHandle
    from examples._protocols.person import ExamplesFlextProtocolsPerson


__all__: tuple[str, ...] = (
    "ExamplesFlextProtocolsEx00",
    "ExamplesFlextProtocolsEx01",
    "ExamplesFlextProtocolsEx02",
    "ExamplesFlextProtocolsEx03",
    "ExamplesFlextProtocolsEx04",
    "ExamplesFlextProtocolsEx05",
    "ExamplesFlextProtocolsEx07",
    "ExamplesFlextProtocolsEx08",
    "ExamplesFlextProtocolsEx10",
    "ExamplesFlextProtocolsEx11",
    "ExamplesFlextProtocolsEx12",
    "ExamplesFlextProtocolsEx14",
    "ExamplesFlextProtocolsHandle",
    "ExamplesFlextProtocolsPerson",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "ExamplesFlextProtocolsEx00": ".ex00",
        "ExamplesFlextProtocolsEx01": ".ex01",
        "ExamplesFlextProtocolsEx02": ".ex02",
        "ExamplesFlextProtocolsEx03": ".ex03",
        "ExamplesFlextProtocolsEx04": ".ex04",
        "ExamplesFlextProtocolsEx05": ".ex05",
        "ExamplesFlextProtocolsEx07": ".ex07",
        "ExamplesFlextProtocolsEx08": ".ex08",
        "ExamplesFlextProtocolsEx10": ".ex10",
        "ExamplesFlextProtocolsEx11": ".ex11",
        "ExamplesFlextProtocolsEx12": ".ex12",
        "ExamplesFlextProtocolsEx14": ".ex14",
        "ExamplesFlextProtocolsHandle": ".handle",
        "ExamplesFlextProtocolsPerson": ".person",
    },
    public_exports=__all__,
)
