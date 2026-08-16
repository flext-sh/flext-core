# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Checker Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .checker_part_03 import FlextUtilitiesChecker

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    ".checker_part_03": ("FlextUtilitiesChecker",)
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
)

__all__: tuple[str, ...] = ("FlextUtilitiesChecker",)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
