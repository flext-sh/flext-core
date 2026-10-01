# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Constants package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

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

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".domain": ("TestsFlextConstantsDomain",),
            ".errors": ("TestsFlextConstantsErrors",),
            ".fixtures": ("TestsFlextConstantsFixtures",),
            ".other": ("TestsFlextConstantsOther",),
            ".result": ("TestsFlextConstantsResult",),
            ".services": ("TestsFlextConstantsServices",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
