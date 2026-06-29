# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

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

    from tests.unit._models.test_base import (
        TestsFlextModelsBase as TestsFlextModelsBase,
    )
    from tests.unit._models.test_cqrs import (
        TestsFlextModelsCQRS as TestsFlextModelsCQRS,
    )
    from tests.unit._models.test_enforcement_sources import (
        TestsFlextModelsEnforcementSources as TestsFlextModelsEnforcementSources,
    )
    from tests.unit._models.test_entity import (
        TestsFlextModelsEntity as TestsFlextModelsEntity,
    )
    from tests.unit._models.test_exception_params_core import (
        TestsFlextModelsExceptionParamsCore as TestsFlextModelsExceptionParamsCore,
    )
    from tests.unit._models.test_exception_params_operations import (
        TestsFlextModelsExceptionParamsOperations as TestsFlextModelsExceptionParamsOperations,
    )
    from tests.unit._models.test_exception_params_resources import (
        TestsFlextModelsExceptionParamsResources as TestsFlextModelsExceptionParamsResources,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_base": ("TestsFlextModelsBase",),
        ".test_cqrs": ("TestsFlextModelsCQRS",),
        ".test_enforcement_sources": ("TestsFlextModelsEnforcementSources",),
        ".test_entity": ("TestsFlextModelsEntity",),
        ".test_exception_params": ("test_exception_params",),
        ".test_exception_params_core": ("TestsFlextModelsExceptionParamsCore",),
        ".test_exception_params_operations": (
            "TestsFlextModelsExceptionParamsOperations",
        ),
        ".test_exception_params_resources": (
            "TestsFlextModelsExceptionParamsResources",
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
