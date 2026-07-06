# AUTO-GENERATED FILE — Regenerate with: make gen
"""Result Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._result_parts.flextprotocolsresult_part_04 import (
        FlextProtocolsResult,
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
