# AUTO-GENERATED FILE — Regenerate with: make gen
"""Constants package."""

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

    from tests._constants.domain import (
        TestsFlextConstantsDomain as TestsFlextConstantsDomain,
    )
    from tests._constants.errors import (
        TestsFlextConstantsErrors as TestsFlextConstantsErrors,
    )
    from tests._constants.fixtures import (
        TestsFlextConstantsFixtures as TestsFlextConstantsFixtures,
    )
    from tests._constants.loggings import (
        TestsFlextConstantsLoggings as TestsFlextConstantsLoggings,
    )
    from tests._constants.other import (
        TestsFlextConstantsOther as TestsFlextConstantsOther,
    )
    from tests._constants.result import (
        TestsFlextConstantsResult as TestsFlextConstantsResult,
    )
    from tests._constants.services import (
        TestsFlextConstantsServices as TestsFlextConstantsServices,
    )
    from tests._constants.settings import (
        TestsFlextConstantsSettings as TestsFlextConstantsSettings,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".domain": ("TestsFlextConstantsDomain",),
        ".errors": ("TestsFlextConstantsErrors",),
        ".fixtures": ("TestsFlextConstantsFixtures",),
        ".loggings": ("TestsFlextConstantsLoggings",),
        ".other": ("TestsFlextConstantsOther",),
        ".result": ("TestsFlextConstantsResult",),
        ".services": ("TestsFlextConstantsServices",),
        ".settings": ("TestsFlextConstantsSettings",),
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
