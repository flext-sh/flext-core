# AUTO-GENERATED FILE — Regenerate with: make gen
"""Base Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_01 import (
        FlextBaseErrorMetadataMixin,
    )
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_02 import (
        FlextBaseErrorStateMixin,
    )
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_03 import (
        FlextExceptionsBase,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".flextexceptionsbase_part_01": ("FlextBaseErrorMetadataMixin",),
        ".flextexceptionsbase_part_02": ("FlextBaseErrorStateMixin",),
        ".flextexceptionsbase_part_03": ("FlextExceptionsBase",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
