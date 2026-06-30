# AUTO-GENERATED FILE — Regenerate with: make gen
"""Handlers Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._handlers_parts.flexthandlers_part_07 import (
        FlextHandlers as FlextHandlers,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flexthandlers_part_07": ("FlextHandlers",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
