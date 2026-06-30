# AUTO-GENERATED FILE — Regenerate with: make gen
"""Exceptions package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_01 import (
        FlextBaseErrorMetadataMixin as FlextBaseErrorMetadataMixin,
    )
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_02 import (
        FlextBaseErrorStateMixin as FlextBaseErrorStateMixin,
    )
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_03 import (
        FlextExceptionsBase as FlextExceptionsBase,
    )
    from flext_core._exceptions._factories_parts.flextexceptionsfactories_part_04 import (
        FlextExceptionsFactories as FlextExceptionsFactories,
    )
    from flext_core._exceptions.helpers import (
        FlextExceptionsHelpers as FlextExceptionsHelpers,
    )
    from flext_core._exceptions.metrics import (
        FlextExceptionsMetrics as FlextExceptionsMetrics,
    )
    from flext_core._exceptions.template import (
        FlextExceptionsTemplate as FlextExceptionsTemplate,
    )
    from flext_core._exceptions.types import (
        FlextExceptionsTypes as FlextExceptionsTypes,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._base_parts": ("_base_parts",),
        "._base_parts.flextexceptionsbase_part_01": ("FlextBaseErrorMetadataMixin",),
        "._base_parts.flextexceptionsbase_part_02": ("FlextBaseErrorStateMixin",),
        "._base_parts.flextexceptionsbase_part_03": ("FlextExceptionsBase",),
        "._factories_parts": ("_factories_parts",),
        "._factories_parts.flextexceptionsfactories_part_04": (
            "FlextExceptionsFactories",
        ),
        ".helpers": ("FlextExceptionsHelpers",),
        ".metrics": ("FlextExceptionsMetrics",),
        ".template": ("FlextExceptionsTemplate",),
        ".types": ("FlextExceptionsTypes",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
