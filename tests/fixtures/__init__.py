# AUTO-GENERATED FILE — Regenerate with: make gen
"""Fixtures package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import (
        c as c,
        d as d,
        e as e,
        h as h,
        m as m,
        p as p,
        r as r,
        s as s,
        t as t,
        td as td,
        tf as tf,
        tk as tk,
        tm as tm,
        tv as tv,
        u as u,
        x as x,
    )

    from tests.fixtures.bad_module import (
        TestsFlextBadAccessors as TestsFlextBadAccessors,
        TestsFlextBadAnyField as TestsFlextBadAnyField,
        TestsFlextBadBareCollection as TestsFlextBadBareCollection,
        TestsFlextBadConstants as TestsFlextBadConstants,
        TestsFlextBadFrozen as TestsFlextBadFrozen,
        TestsFlextBadInlineUnion as TestsFlextBadInlineUnion,
        TestsFlextBadMissingDesc as TestsFlextBadMissingDesc,
        TestsFlextBadMutableDefault as TestsFlextBadMutableDefault,
        TestsFlextBadWorkerSettings as TestsFlextBadWorkerSettings,
    )
    from tests.fixtures.clean_module import (
        TestsFlextCleanConstants as TestsFlextCleanConstants,
        TestsFlextCleanModels as TestsFlextCleanModels,
        TestsFlextCleanProtocols as TestsFlextCleanProtocols,
        TestsFlextCleanServiceBase as TestsFlextCleanServiceBase,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".bad_module": (
            "TestsFlextBadAccessors",
            "TestsFlextBadAnyField",
            "TestsFlextBadBareCollection",
            "TestsFlextBadConstants",
            "TestsFlextBadFrozen",
            "TestsFlextBadInlineUnion",
            "TestsFlextBadMissingDesc",
            "TestsFlextBadMutableDefault",
            "TestsFlextBadWorkerSettings",
        ),
        ".clean_module": (
            "TestsFlextCleanConstants",
            "TestsFlextCleanModels",
            "TestsFlextCleanProtocols",
            "TestsFlextCleanServiceBase",
        ),
        "flext_tests": (
            "c",
            "d",
            "e",
            "h",
            "m",
            "p",
            "r",
            "s",
            "t",
            "td",
            "tf",
            "tk",
            "tm",
            "tv",
            "u",
            "x",
        ),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
