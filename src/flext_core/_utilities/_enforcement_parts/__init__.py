# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Enforcement Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .enforcement_part_01 import PREDICATE_BINDINGS
    from .enforcement_part_05 import FlextUtilitiesEnforcement
    from .enforcement_part_06 import EXTENDED_PREDICATE_BINDINGS

_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    ".enforcement_part_01": ("PREDICATE_BINDINGS",),
    ".enforcement_part_05": ("FlextUtilitiesEnforcement",),
    ".enforcement_part_06": ("EXTENDED_PREDICATE_BINDINGS",),
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
)

__all__: tuple[str, ...] = (
    "EXTENDED_PREDICATE_BINDINGS",
    "PREDICATE_BINDINGS",
    "FlextUtilitiesEnforcement",
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
