# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions import _base_parts, _factories_parts
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_01 import (
        FlextBaseErrorMetadataMixin,
    )
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_02 import (
        FlextBaseErrorStateMixin,
    )
    from flext_core._exceptions.base import FlextExceptionsBase
    from flext_core._exceptions.exception_types import FlextExceptionsTypes
    from flext_core._exceptions.factories import FlextExceptionsFactories
    from flext_core._exceptions.helpers import FlextExceptionsHelpers
    from flext_core._exceptions.metrics import FlextExceptionsMetrics
    from flext_core._exceptions.template import FlextExceptionsTemplate


__all__: tuple[str, ...] = (
    "FlextBaseErrorMetadataMixin",
    "FlextBaseErrorStateMixin",
    "FlextExceptionsBase",
    "FlextExceptionsFactories",
    "FlextExceptionsHelpers",
    "FlextExceptionsMetrics",
    "FlextExceptionsTemplate",
    "FlextExceptionsTypes",
    "_base_parts",
    "_factories_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._base_parts": ("_base_parts",),
            "._base_parts.flextexceptionsbase_part_01": (
                "FlextBaseErrorMetadataMixin",
            ),
            "._base_parts.flextexceptionsbase_part_02": ("FlextBaseErrorStateMixin",),
            "._factories_parts": ("_factories_parts",),
            ".base": ("FlextExceptionsBase",),
            ".exception_types": ("FlextExceptionsTypes",),
            ".factories": ("FlextExceptionsFactories",),
            ".helpers": ("FlextExceptionsHelpers",),
            ".metrics": ("FlextExceptionsMetrics",),
            ".template": ("FlextExceptionsTemplate",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
