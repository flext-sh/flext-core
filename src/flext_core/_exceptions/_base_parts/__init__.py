# AUTO-GENERATED FILE — Regenerate with: make gen
"""Base Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_01 import (
        FlextBaseErrorMetadataMixin as FlextBaseErrorMetadataMixin,
    )
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_02 import (
        FlextBaseErrorStateMixin as FlextBaseErrorStateMixin,
    )
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_03 import (
        FlextExceptionsBase as FlextExceptionsBase,
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
