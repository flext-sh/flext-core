# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if _t.TYPE_CHECKING:
    from flext_core.decorators import d
    from flext_core.exceptions import e
    from flext_core.handlers import h
    from flext_core.mixins import x
    from flext_core.result import r
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
    build_lazy_import_map(
        {
            ".base": ("TestsFlextCoreServiceBase",),
            ".constants": ("TestsFlextCoreConstants",),
            ".models": ("TestsFlextCoreModels",),
            ".protocols": ("TestsFlextCoreProtocols",),
            ".typings": (
                "T",
                "T_co",
                "T_contra",
                "TestsFlextCoreTypes",
            ),
            ".utilities": ("TestsFlextCoreUtilities",),
            "flext_core.decorators": ("d",),
            "flext_core.exceptions": ("e",),
            "flext_core.handlers": ("h",),
            "flext_core.mixins": ("x",),
            "flext_core.result": ("r",),
        },
        alias_groups={
            ".base": (("s", "TestsFlextCoreServiceBase"),),
            ".constants": (("c", "TestsFlextCoreConstants"),),
            ".models": (("m", "TestsFlextCoreModels"),),
            ".protocols": (("p", "TestsFlextCoreProtocols"),),
            ".typings": (("t", "TestsFlextCoreTypes"),),
            ".utilities": (("u", "TestsFlextCoreUtilities"),),
        },
    ),
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
