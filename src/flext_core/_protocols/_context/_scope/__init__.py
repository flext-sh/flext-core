# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Context. Scope package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._context._scope.flextmodelscontextscope_part_01 import (
        FlextProtocolsContextScopeFlextmodelscontextscopePart01,
    )
    from flext_core._protocols._context._scope.flextmodelscontextscope_part_02 import (
        FlextProtocolsContextScopeFlextmodelscontextscopePart02,
    )
    from flext_core._protocols._context._scope.flextmodelscontextscope_part_03 import (
        FlextProtocolsContextScopeFlextmodelscontextscopePart03,
    )


__all__: tuple[str, ...] = (
    "FlextProtocolsContextScopeFlextmodelscontextscopePart01",
    "FlextProtocolsContextScopeFlextmodelscontextscopePart02",
    "FlextProtocolsContextScopeFlextmodelscontextscopePart03",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextProtocolsContextScopeFlextmodelscontextscopePart01": (
            ".flextmodelscontextscope_part_01"
        ),
        "FlextProtocolsContextScopeFlextmodelscontextscopePart02": (
            ".flextmodelscontextscope_part_02"
        ),
        "FlextProtocolsContextScopeFlextmodelscontextscopePart03": (
            ".flextmodelscontextscope_part_03"
        ),
    },
    public_exports=__all__,
)
