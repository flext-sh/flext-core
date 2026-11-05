# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flext_core
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions.base import FlextExceptionsBase
    from flext_core._exceptions.exception_types import FlextExceptionsTypes
    from flext_core._exceptions.factories import FlextExceptionsFactories
    from flext_core._exceptions.helpers import FlextExceptionsHelpers
    from flext_core._exceptions.metrics import FlextExceptionsMetrics
    from flext_core._exceptions.template import FlextExceptionsTemplate
    from flext_core._exceptions.violations import FlextExceptionsViolations


__all__: tuple[str, ...] = (
    "FlextExceptionsBase",
    "FlextExceptionsFactories",
    "FlextExceptionsHelpers",
    "FlextExceptionsMetrics",
    "FlextExceptionsTemplate",
    "FlextExceptionsTypes",
    "FlextExceptionsViolations",
    "flext_core",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextExceptionsBase": ".base",
        "FlextExceptionsFactories": ".factories",
        "FlextExceptionsHelpers": ".helpers",
        "FlextExceptionsMetrics": ".metrics",
        "FlextExceptionsTemplate": ".template",
        "FlextExceptionsTypes": ".exception_types",
        "FlextExceptionsViolations": ".violations",
    },
    public_exports=__all__,
)
