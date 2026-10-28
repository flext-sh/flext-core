# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions._base.flextexceptionsbase_part_01 import (
        FlextBaseErrorMetadata,
    )
    from flext_core._exceptions._base.flextexceptionsbase_part_02 import (
        FlextBaseErrorState,
    )
    from flext_core._exceptions._base.flextexceptionsbase_part_03 import (
        FlextExceptionsBasePart03,
    )
    from flext_core._exceptions._factories.fail import FlextExceptionsFactoriesFail
    from flext_core._exceptions.base import FlextExceptionsBase
    from flext_core._exceptions.exception_types import FlextExceptionsTypes
    from flext_core._exceptions.factories import FlextExceptionsFactories
    from flext_core._exceptions.helpers import FlextExceptionsHelpers
    from flext_core._exceptions.metrics import FlextExceptionsMetrics
    from flext_core._exceptions.template import FlextExceptionsTemplate


__all__: tuple[str, ...] = (
    "FlextBaseErrorMetadata",
    "FlextBaseErrorState",
    "FlextExceptionsBase",
    "FlextExceptionsBasePart03",
    "FlextExceptionsFactories",
    "FlextExceptionsFactoriesFail",
    "FlextExceptionsHelpers",
    "FlextExceptionsMetrics",
    "FlextExceptionsTemplate",
    "FlextExceptionsTypes",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextBaseErrorMetadata": "._base.flextexceptionsbase_part_01",
        "FlextBaseErrorState": "._base.flextexceptionsbase_part_02",
        "FlextExceptionsBase": ".base",
        "FlextExceptionsBasePart03": "._base.flextexceptionsbase_part_03",
        "FlextExceptionsFactories": ".factories",
        "FlextExceptionsFactoriesFail": "._factories.fail",
        "FlextExceptionsHelpers": ".helpers",
        "FlextExceptionsMetrics": ".metrics",
        "FlextExceptionsTemplate": ".template",
        "FlextExceptionsTypes": ".exception_types",
    }),
    public_exports=__all__,
)
