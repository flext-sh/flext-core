# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Result package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

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

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextResultBase": ".base",
        "FlextResultBehavior": ".behavior",
        "FlextResultComposition": ".composition",
        "FlextResultConstruction": ".construction",
        "FlextResultTransforms": ".transforms",
        "FlextResultUnwrap": ".unwrap",
        "copy_result": ".construction",
        "ok_result": ".construction",
    }),
    public_exports=__all__,
)
