# AUTO-GENERATED FILE — Regenerate with: make gen
"""Checker Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._checker_parts.checker_part_03 import (
        FlextUtilitiesChecker,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".checker_part_03": ("FlextUtilitiesChecker",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
