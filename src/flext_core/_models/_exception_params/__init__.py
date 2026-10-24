# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Exception Params package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._exception_params.base import (
        FlextModelsExceptionParamsPart01,
    )
    from flext_core._models._exception_params.operation_errors import (
        FlextModelsExceptionParamsPart03,
    )
    from flext_core._models._exception_params.service_errors import (
        FlextModelsExceptionParamsPart02,
    )


__all__: tuple[str, ...] = (
    "FlextModelsExceptionParamsPart01",
    "FlextModelsExceptionParamsPart02",
    "FlextModelsExceptionParamsPart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsExceptionParamsPart01": ".base",
        "FlextModelsExceptionParamsPart02": ".service_errors",
        "FlextModelsExceptionParamsPart03": ".operation_errors",
    }),
    public_exports=__all__,
)
