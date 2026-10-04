# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Result package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._result.base import FlextResultBase
    from flext_core._result.behavior import FlextResultBehavior
    from flext_core._result.composition import FlextResultComposition
    from flext_core._result.construction import (
        FlextResultConstruction,
        copy_result,
        ok_result,
    )
    from flext_core._result.transforms import FlextResultTransforms
    from flext_core._result.unwrap import FlextResultUnwrap


__all__: tuple[str, ...] = (
    "FlextResultBase",
    "FlextResultBehavior",
    "FlextResultComposition",
    "FlextResultConstruction",
    "FlextResultTransforms",
    "FlextResultUnwrap",
    "copy_result",
    "ok_result",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".base": ("FlextResultBase",),
            ".behavior": ("FlextResultBehavior",),
            ".composition": ("FlextResultComposition",),
            ".construction": ("FlextResultConstruction", "copy_result", "ok_result"),
            ".transforms": ("FlextResultTransforms",),
            ".unwrap": ("FlextResultUnwrap",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
