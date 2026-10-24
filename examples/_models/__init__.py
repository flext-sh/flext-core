# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples. Models package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from examples._models.errors import ExamplesFlextModelsErrors
    from examples._models.ex00 import ExamplesFlextModelsEx00
    from examples._models.ex01 import ExamplesFlextModelsEx01
    from examples._models.ex02 import ExamplesFlextModelsEx02
    from examples._models.ex03 import ExamplesFlextModelsEx03
    from examples._models.ex04 import ExamplesFlextModelsEx04
    from examples._models.ex05 import ExamplesFlextModelsEx05
    from examples._models.ex07 import ExamplesFlextModelsEx07
    from examples._models.ex08 import ExamplesFlextModelsEx08
    from examples._models.ex10 import ExamplesFlextModelsEx10
    from examples._models.ex11 import ExamplesFlextModelsEx11
    from examples._models.ex12 import ExamplesFlextModelsEx12
    from examples._models.ex14 import ExamplesFlextModelsEx14
    from examples._models.handle import ExamplesFlextSharedHandle
    from examples._models.output import ExamplesFlextModelsOutput
    from examples._models.person import ExamplesFlextSharedPerson


__all__: tuple[str, ...] = (
    "ExamplesFlextModelsErrors",
    "ExamplesFlextModelsEx00",
    "ExamplesFlextModelsEx01",
    "ExamplesFlextModelsEx02",
    "ExamplesFlextModelsEx03",
    "ExamplesFlextModelsEx04",
    "ExamplesFlextModelsEx05",
    "ExamplesFlextModelsEx07",
    "ExamplesFlextModelsEx08",
    "ExamplesFlextModelsEx10",
    "ExamplesFlextModelsEx11",
    "ExamplesFlextModelsEx12",
    "ExamplesFlextModelsEx14",
    "ExamplesFlextModelsOutput",
    "ExamplesFlextSharedHandle",
    "ExamplesFlextSharedPerson",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "ExamplesFlextModelsErrors": ".errors",
        "ExamplesFlextModelsEx00": ".ex00",
        "ExamplesFlextModelsEx01": ".ex01",
        "ExamplesFlextModelsEx02": ".ex02",
        "ExamplesFlextModelsEx03": ".ex03",
        "ExamplesFlextModelsEx04": ".ex04",
        "ExamplesFlextModelsEx05": ".ex05",
        "ExamplesFlextModelsEx07": ".ex07",
        "ExamplesFlextModelsEx08": ".ex08",
        "ExamplesFlextModelsEx10": ".ex10",
        "ExamplesFlextModelsEx11": ".ex11",
        "ExamplesFlextModelsEx12": ".ex12",
        "ExamplesFlextModelsEx14": ".ex14",
        "ExamplesFlextModelsOutput": ".output",
        "ExamplesFlextSharedHandle": ".handle",
        "ExamplesFlextSharedPerson": ".person",
    }),
    public_exports=__all__,
)
