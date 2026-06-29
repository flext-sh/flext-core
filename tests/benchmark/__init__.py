# AUTO-GENERATED FILE — Regenerate with: make gen
"""Benchmark package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
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

    from tests.benchmark.test_container_memory import (
        TestsFlextContainerMemory as TestsFlextContainerMemory,
    )
    from tests.benchmark.test_container_performance import (
        TestsFlextContainerPerformance as TestsFlextContainerPerformance,
    )
    from tests.benchmark.test_lazy_performance import (
        TestsFlextLazyPerformance as TestsFlextLazyPerformance,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_container_memory": ("TestsFlextContainerMemory",),
        ".test_container_performance": ("TestsFlextContainerPerformance",),
        ".test_lazy_performance": ("TestsFlextLazyPerformance",),
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
