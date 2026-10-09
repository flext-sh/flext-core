# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Constants package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from tests._constants.domain import TestsFlextConstantsDomain
    from tests._constants.errors import TestsFlextConstantsErrors
    from tests._constants.fixtures import TestsFlextConstantsFixtures
    from tests._constants.other import TestsFlextConstantsOther
    from tests._constants.result import TestsFlextConstantsResult
    from tests._constants.services import TestsFlextConstantsServices


__all__: tuple[str, ...] = (
    "TestsFlextConstantsDomain",
    "TestsFlextConstantsErrors",
    "TestsFlextConstantsFixtures",
    "TestsFlextConstantsOther",
    "TestsFlextConstantsResult",
    "TestsFlextConstantsServices",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "TestsFlextConstantsDomain": ".domain",
        "TestsFlextConstantsErrors": ".errors",
        "TestsFlextConstantsFixtures": ".fixtures",
        "TestsFlextConstantsOther": ".other",
        "TestsFlextConstantsResult": ".result",
        "TestsFlextConstantsServices": ".services",
    }),
    public_exports=__all__,
)
