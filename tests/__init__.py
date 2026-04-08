# AUTO-GENERATED FILE — Regenerate with: make gen
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
    (".unit",),
    {
        "T": ".typings",
        "T_co": ".typings",
        "T_contra": ".typings",
        "TestsFlextCoreConstants": ".constants",
        "TestsFlextCoreModels": ".models",
        "TestsFlextCoreProtocols": ".protocols",
        "TestsFlextCoreServiceBase": ".base",
        "TestsFlextCoreTypes": ".typings",
        "TestsFlextCoreUtilities": ".utilities",
        "c": (".constants", "TestsFlextCoreConstants"),
        "d": ("flext_core.decorators", "FlextDecorators"),
        "e": ("flext_core.exceptions", "FlextExceptions"),
        "h": ("flext_core.handlers", "FlextHandlers"),
        "m": (".models", "TestsFlextCoreModels"),
        "p": (".protocols", "TestsFlextCoreProtocols"),
        "r": ("flext_core.result", "FlextResult"),
        "s": (".base", "TestsFlextCoreServiceBase"),
        "t": (".typings", "TestsFlextCoreTypes"),
        "u": (".utilities", "TestsFlextCoreUtilities"),
        "x": ("flext_core.mixins", "FlextMixins"),
    },
    exclude_names=(
        "cleanup_submodule_namespace",
        "install_lazy_exports",
        "lazy_getattr",
        "logger",
        "merge_lazy_imports",
        "output",
        "output_reporting",
    ),
    module_name=__name__,
)

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
