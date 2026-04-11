# AUTO-GENERATED FILE — Regenerate with: make gen
"""Constants package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from _constants.domain import TestsFlextCoreConstantsDomain
    from _constants.errors import TestsFlextCoreConstantsErrors
    from _constants.fixtures import TestsFlextCoreConstantsFixtures
    from _constants.loggings import TestsFlextCoreConstantsLoggings
    from _constants.other import TestsFlextCoreConstantsOther
    from _constants.result import TestsFlextCoreConstantsResult
    from _constants.services import TestsFlextCoreConstantsServices
    from _constants.settings import TestsFlextCoreConstantsSettings
    from _constants.strings import TestsFlextCoreConstantsStrings
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".domain": ("TestsFlextCoreConstantsDomain",),
        ".errors": ("TestsFlextCoreConstantsErrors",),
        ".fixtures": ("TestsFlextCoreConstantsFixtures",),
        ".loggings": ("TestsFlextCoreConstantsLoggings",),
        ".other": ("TestsFlextCoreConstantsOther",),
        ".result": ("TestsFlextCoreConstantsResult",),
        ".services": ("TestsFlextCoreConstantsServices",),
        ".settings": ("TestsFlextCoreConstantsSettings",),
        ".strings": ("TestsFlextCoreConstantsStrings",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

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
