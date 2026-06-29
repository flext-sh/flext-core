# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from examples._models.errors import (
        ExamplesFlextModelsErrors as ExamplesFlextModelsErrors,
    )
    from examples._models.ex00 import ExamplesFlextModelsEx00 as ExamplesFlextModelsEx00
    from examples._models.ex01 import ExamplesFlextModelsEx01 as ExamplesFlextModelsEx01
    from examples._models.ex02 import ExamplesFlextModelsEx02 as ExamplesFlextModelsEx02
    from examples._models.ex03 import ExamplesFlextModelsEx03 as ExamplesFlextModelsEx03
    from examples._models.ex04 import ExamplesFlextModelsEx04 as ExamplesFlextModelsEx04
    from examples._models.ex05 import ExamplesFlextModelsEx05 as ExamplesFlextModelsEx05
    from examples._models.ex07 import ExamplesFlextModelsEx07 as ExamplesFlextModelsEx07
    from examples._models.ex08 import ExamplesFlextModelsEx08 as ExamplesFlextModelsEx08
    from examples._models.ex10 import ExamplesFlextModelsEx10 as ExamplesFlextModelsEx10
    from examples._models.ex11 import ExamplesFlextModelsEx11 as ExamplesFlextModelsEx11
    from examples._models.ex12 import ExamplesFlextModelsEx12 as ExamplesFlextModelsEx12
    from examples._models.ex14 import ExamplesFlextModelsEx14 as ExamplesFlextModelsEx14
    from examples._models.output import (
        ExamplesFlextModelsOutput as ExamplesFlextModelsOutput,
    )
    from examples._models.shared import (
        ExamplesFlextSharedHandle as ExamplesFlextSharedHandle,
        ExamplesFlextSharedPerson as ExamplesFlextSharedPerson,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".errors": ("ExamplesFlextModelsErrors",),
        ".ex00": ("ExamplesFlextModelsEx00",),
        ".ex01": ("ExamplesFlextModelsEx01",),
        ".ex02": ("ExamplesFlextModelsEx02",),
        ".ex03": ("ExamplesFlextModelsEx03",),
        ".ex04": ("ExamplesFlextModelsEx04",),
        ".ex05": ("ExamplesFlextModelsEx05",),
        ".ex07": ("ExamplesFlextModelsEx07",),
        ".ex08": ("ExamplesFlextModelsEx08",),
        ".ex10": ("ExamplesFlextModelsEx10",),
        ".ex11": ("ExamplesFlextModelsEx11",),
        ".ex12": ("ExamplesFlextModelsEx12",),
        ".ex14": ("ExamplesFlextModelsEx14",),
        ".output": ("ExamplesFlextModelsOutput",),
        ".shared": (
            "ExamplesFlextSharedHandle",
            "ExamplesFlextSharedPerson",
        ),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
