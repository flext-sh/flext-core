# AUTO-GENERATED FILE — Regenerate with: make gen
"""Loggings Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._loggings_parts.flextlogger_part_05 import (
        FlextUtilitiesLogging as FlextUtilitiesLogging,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextlogger_part_05": ("FlextUtilitiesLogging",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
