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
    from flext_core._exceptions import _base as _base, _factories as _factories
    from flext_core._exceptions._base import (
        FlextBaseErrorMetadataMixin,
        FlextBaseErrorStateMixin,
        FlextExceptionsBasePart03,
    )
    from flext_core._exceptions._factories import (
        FlextExceptionsFactoriesPart01,
        FlextExceptionsFactoriesPart02,
        FlextExceptionsFactoriesPart03,
        FlextExceptionsFactoriesPart04,
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
    "FlextExceptionsBasePart03",
    "FlextExceptionsFactories",
    "FlextExceptionsFactoriesPart01",
    "FlextExceptionsFactoriesPart02",
    "FlextExceptionsFactoriesPart03",
    "FlextExceptionsFactoriesPart04",
    "FlextExceptionsHelpers",
    "FlextExceptionsMetrics",
    "FlextExceptionsTemplate",
    "FlextExceptionsTypes",
    "_base",
    "_factories",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextBaseErrorMetadataMixin": "._base.flextexceptionsbase_part_01",
        "FlextBaseErrorStateMixin": "._base.flextexceptionsbase_part_02",
        "FlextExceptionsBase": ".base",
        "FlextExceptionsBasePart03": "._base.flextexceptionsbase_part_03",
        "FlextExceptionsFactories": ".factories",
        "FlextExceptionsFactoriesPart01": (
            "._factories.flextexceptionsfactories_part_01"
        ),
        "FlextExceptionsFactoriesPart02": (
            "._factories.flextexceptionsfactories_part_02"
        ),
        "FlextExceptionsFactoriesPart03": (
            "._factories.flextexceptionsfactories_part_03"
        ),
        "FlextExceptionsFactoriesPart04": (
            "._factories.flextexceptionsfactories_part_04"
        ),
        "FlextExceptionsHelpers": ".helpers",
        "FlextExceptionsMetrics": ".metrics",
        "FlextExceptionsTemplate": ".template",
        "FlextExceptionsTypes": ".exception_types",
        "_base": "._base",
        "_factories": "._factories",
    }),
    public_exports=__all__,
)
