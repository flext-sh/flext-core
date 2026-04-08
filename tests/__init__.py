# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Tests package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _t.TYPE_CHECKING:
    from flext_core.decorators import FlextDecorators as d
    from flext_core.exceptions import FlextExceptions as e
    from flext_core.handlers import FlextHandlers as h
    from flext_core.mixins import FlextMixins as x
    from flext_core.result import FlextResult as r
    from tests.base import TestsFlextCoreServiceBase, TestsFlextCoreServiceBase as s
    from tests.constants import TestsFlextCoreConstants, TestsFlextCoreConstants as c
    from tests.models import TestsFlextCoreModels, TestsFlextCoreModels as m
    from tests.protocols import TestsFlextCoreProtocols, TestsFlextCoreProtocols as p
    from tests.typings import (
        T,
        T_co,
        T_contra,
        TestsFlextCoreTypes,
        TestsFlextCoreTypes as t,
    )
    from tests.unit._utilities.test_guards import TestFlextUtilitiesGuards
    from tests.unit.protocols import TestsFlextUnitProtocols
    from tests.utilities import TestsFlextCoreUtilities, TestsFlextCoreUtilities as u
_LAZY_IMPORTS = merge_lazy_imports(
    ("tests.unit",),
    {
        "T": ("tests.typings", "T"),
        "T_co": ("tests.typings", "T_co"),
        "T_contra": ("tests.typings", "T_contra"),
        "TestsFlextCoreConstants": ("tests.constants", "TestsFlextCoreConstants"),
        "TestsFlextCoreModels": ("tests.models", "TestsFlextCoreModels"),
        "TestsFlextCoreProtocols": ("tests.protocols", "TestsFlextCoreProtocols"),
        "TestsFlextCoreServiceBase": ("tests.base", "TestsFlextCoreServiceBase"),
        "TestsFlextCoreTypes": ("tests.typings", "TestsFlextCoreTypes"),
        "TestsFlextCoreUtilities": ("tests.utilities", "TestsFlextCoreUtilities"),
        "c": ("tests.constants", "TestsFlextCoreConstants"),
        "d": ("flext_core.decorators", "FlextDecorators"),
        "e": ("flext_core.exceptions", "FlextExceptions"),
        "h": ("flext_core.handlers", "FlextHandlers"),
        "m": ("tests.models", "TestsFlextCoreModels"),
        "p": ("tests.protocols", "TestsFlextCoreProtocols"),
        "r": ("flext_core.result", "FlextResult"),
        "s": ("tests.base", "TestsFlextCoreServiceBase"),
        "t": ("tests.typings", "TestsFlextCoreTypes"),
        "u": ("tests.utilities", "TestsFlextCoreUtilities"),
        "x": ("flext_core.mixins", "FlextMixins"),
    },
)
_ = _LAZY_IMPORTS.pop("cleanup_submodule_namespace", None)
_ = _LAZY_IMPORTS.pop("install_lazy_exports", None)
_ = _LAZY_IMPORTS.pop("lazy_getattr", None)
_ = _LAZY_IMPORTS.pop("logger", None)
_ = _LAZY_IMPORTS.pop("merge_lazy_imports", None)
_ = _LAZY_IMPORTS.pop("output", None)
_ = _LAZY_IMPORTS.pop("output_reporting", None)

__all__ = [
    "T",
    "T_co",
    "T_contra",
    "TestFlextUtilitiesGuards",
    "TestsFlextCoreConstants",
    "TestsFlextCoreModels",
    "TestsFlextCoreProtocols",
    "TestsFlextCoreServiceBase",
    "TestsFlextCoreTypes",
    "TestsFlextCoreUtilities",
    "TestsFlextUnitProtocols",
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
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
