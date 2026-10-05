# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

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

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextBaseErrorMetadataMixin": "._base_parts.flextexceptionsbase_part_01",
        "FlextBaseErrorStateMixin": "._base_parts.flextexceptionsbase_part_02",
        "FlextExceptionsBase": ".base",
        "FlextExceptionsFactories": ".factories",
        "FlextExceptionsHelpers": ".helpers",
        "FlextExceptionsMetrics": ".metrics",
        "FlextExceptionsTemplate": ".template",
        "FlextExceptionsTypes": ".exception_types",
        "_base_parts": "._base_parts",
        "_factories_parts": "._factories_parts",
    }),
    public_exports=__all__,
)
