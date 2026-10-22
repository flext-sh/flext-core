# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Exception Params Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._exception_params_parts import (
        flextmodelsexceptionparams_part_01 as _owner_0,
    )

    FlextModelsExceptionParamsPart01 = _owner_0.FlextModelsExceptionParamsPart01
    from flext_core._models._exception_params_parts import (
        flextmodelsexceptionparams_part_02 as _owner_1,
    )

    FlextModelsExceptionParamsPart02 = _owner_1.FlextModelsExceptionParamsPart02
    from flext_core._models._exception_params_parts import (
        flextmodelsexceptionparams_part_03 as _owner_2,
    )

    FlextModelsExceptionParamsPart03 = _owner_2.FlextModelsExceptionParamsPart03


__all__: tuple[str, ...] = (
    "FlextModelsExceptionParamsPart01",
    "FlextModelsExceptionParamsPart02",
    "FlextModelsExceptionParamsPart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextModelsExceptionParamsPart01": ".flextmodelsexceptionparams_part_01",
        "FlextModelsExceptionParamsPart02": ".flextmodelsexceptionparams_part_02",
        "FlextModelsExceptionParamsPart03": ".flextmodelsexceptionparams_part_03",
    }),
    public_exports=__all__,
)
