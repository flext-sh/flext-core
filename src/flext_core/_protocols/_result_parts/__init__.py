# AUTO-GENERATED FILE — Regenerate with: make gen
"""Result Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._protocols._result_parts.flextprotocolsresult_part_04 import (
        FlextProtocolsResult as FlextProtocolsResult,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextprotocolsresult_part_04": ("FlextProtocolsResult",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
