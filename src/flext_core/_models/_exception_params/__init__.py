# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Exception Params package.

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
    from flext_core._models._exception_params.base import FlextModelsExceptionParamsBase
    from flext_core._models._exception_params.operation_errors import (
        FlextModelsExceptionParamsOperationErrors,
    )
    from flext_core._models._exception_params.service_errors import (
        FlextModelsExceptionParamsServiceErrors,
    )


__all__: tuple[str, ...] = (
    "FlextModelsExceptionParamsBase",
    "FlextModelsExceptionParamsOperationErrors",
    "FlextModelsExceptionParamsServiceErrors",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsExceptionParamsBase": ".base",
        "FlextModelsExceptionParamsOperationErrors": ".operation_errors",
        "FlextModelsExceptionParamsServiceErrors": ".service_errors",
    }),
    public_exports=__all__,
)
