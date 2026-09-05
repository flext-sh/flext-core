# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples. Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .errors import ExamplesFlextModelsErrors
    from .ex00 import ExamplesFlextModelsEx00
    from .ex01 import ExamplesFlextModelsEx01
    from .ex02 import ExamplesFlextModelsEx02
    from .ex03 import ExamplesFlextModelsEx03
    from .ex04 import ExamplesFlextModelsEx04
    from .ex05 import ExamplesFlextModelsEx05
    from .ex07 import ExamplesFlextModelsEx07
    from .ex08 import ExamplesFlextModelsEx08
    from .ex10 import ExamplesFlextModelsEx10
    from .ex11 import ExamplesFlextModelsEx11
    from .ex12 import ExamplesFlextModelsEx12
    from .ex14 import ExamplesFlextModelsEx14
    from .output import ExamplesFlextModelsOutput
    from .shared import ExamplesFlextSharedHandle, ExamplesFlextSharedPerson
__all__: tuple[str, ...] = (
    "ExamplesFlextModelsErrors",
    "ExamplesFlextModelsEx00",
    "ExamplesFlextModelsEx01",
    "ExamplesFlextModelsEx02",
    "ExamplesFlextModelsEx03",
    "ExamplesFlextModelsEx04",
    "ExamplesFlextModelsEx05",
    "ExamplesFlextModelsEx07",
    "ExamplesFlextModelsEx08",
    "ExamplesFlextModelsEx10",
    "ExamplesFlextModelsEx11",
    "ExamplesFlextModelsEx12",
    "ExamplesFlextModelsEx14",
    "ExamplesFlextModelsOutput",
    "ExamplesFlextSharedHandle",
    "ExamplesFlextSharedPerson",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
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
            ".shared": ("ExamplesFlextSharedHandle", "ExamplesFlextSharedPerson"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
