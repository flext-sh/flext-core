# AUTO-GENERATED FILE — Regenerate with: make gen
"""Enforcement Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._enforcement_parts.enforcement_part_01 import (
        PREDICATE_BINDINGS,
    )
    from flext_core._utilities._enforcement_parts.enforcement_part_05 import (
        FlextUtilitiesEnforcement,
    )
    from flext_core._utilities._enforcement_parts.enforcement_part_06 import (
        EXTENDED_PREDICATE_BINDINGS,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".enforcement_part_01": ("PREDICATE_BINDINGS",),
        ".enforcement_part_05": ("FlextUtilitiesEnforcement",),
        ".enforcement_part_06": ("EXTENDED_PREDICATE_BINDINGS",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
