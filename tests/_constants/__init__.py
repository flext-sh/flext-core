"""Auto-generated lazy exports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing

from flext_core.lazy import install_lazy_exports

if typing.TYPE_CHECKING:
    from .domain import TestsFlextCoreConstantsDomain
    from .errors import TestsFlextCoreConstantsErrors
    from .fixtures import TestsFlextCoreConstantsFixtures
    from .loggings import TestsFlextCoreConstantsLoggings
    from .other import TestsFlextCoreConstantsOther
    from .result import TestsFlextCoreConstantsResult
    from .services import TestsFlextCoreConstantsServices
    from .settings import TestsFlextCoreConstantsSettings
    from .strings import TestsFlextCoreConstantsStrings

__all__ = [
    "TestsFlextCoreConstantsDomain",
    "TestsFlextCoreConstantsErrors",
    "TestsFlextCoreConstantsFixtures",
    "TestsFlextCoreConstantsLoggings",
    "TestsFlextCoreConstantsOther",
    "TestsFlextCoreConstantsResult",
    "TestsFlextCoreConstantsServices",
    "TestsFlextCoreConstantsSettings",
    "TestsFlextCoreConstantsStrings",
]

_LAZY_IMPORTS = {
    "TestsFlextCoreConstantsDomain": (".domain", "TestsFlextCoreConstantsDomain"),
    "TestsFlextCoreConstantsErrors": (".errors", "TestsFlextCoreConstantsErrors"),
    "TestsFlextCoreConstantsFixtures": (".fixtures", "TestsFlextCoreConstantsFixtures"),
    "TestsFlextCoreConstantsLoggings": (".loggings", "TestsFlextCoreConstantsLoggings"),
    "TestsFlextCoreConstantsOther": (".other", "TestsFlextCoreConstantsOther"),
    "TestsFlextCoreConstantsResult": (".result", "TestsFlextCoreConstantsResult"),
    "TestsFlextCoreConstantsServices": (".services", "TestsFlextCoreConstantsServices"),
    "TestsFlextCoreConstantsSettings": (".settings", "TestsFlextCoreConstantsSettings"),
    "TestsFlextCoreConstantsStrings": (".strings", "TestsFlextCoreConstantsStrings"),
}

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
