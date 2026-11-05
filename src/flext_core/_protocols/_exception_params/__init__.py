# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Exception Params package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flext_core
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._exception_params.base import (
        FlextProtocolsExceptionParamsBase,
    )
    from flext_core._protocols._exception_params.operation_errors import (
        FlextProtocolsExceptionParamsOperationErrors,
    )
    from flext_core._protocols._exception_params.service_errors import (
        FlextProtocolsExceptionParamsServiceErrors,
    )


__all__: tuple[str, ...] = (
    "FlextProtocolsExceptionParamsBase",
    "FlextProtocolsExceptionParamsOperationErrors",
    "FlextProtocolsExceptionParamsServiceErrors",
    "flext_core",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextProtocolsExceptionParamsBase": ".base",
        "FlextProtocolsExceptionParamsOperationErrors": ".operation_errors",
        "FlextProtocolsExceptionParamsServiceErrors": ".service_errors",
    },
    public_exports=__all__,
)
