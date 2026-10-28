# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Context. Scope package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._context._scope.flextmodelscontextscope_part_01 import (
        FlextModelsContextScopePart01,
    )
    from flext_core._models._context._scope.flextmodelscontextscope_part_02 import (
        FlextModelsContextScopePart02,
    )
    from flext_core._models._context._scope.flextmodelscontextscope_part_03 import (
        FlextModelsContextScopePart03,
    )


__all__: tuple[str, ...] = (
    "FlextModelsContextScopePart01",
    "FlextModelsContextScopePart02",
    "FlextModelsContextScopePart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsContextScopePart01": ".flextmodelscontextscope_part_01",
        "FlextModelsContextScopePart02": ".flextmodelscontextscope_part_02",
        "FlextModelsContextScopePart03": ".flextmodelscontextscope_part_03",
    }),
    public_exports=__all__,
)
