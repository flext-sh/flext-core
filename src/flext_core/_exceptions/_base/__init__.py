# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions. Base package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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


__all__: tuple[str, ...] = (
    "FlextBaseErrorMetadata",
    "FlextBaseErrorState",
    "FlextExceptionsBasePart03",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextBaseErrorMetadata": ".flextexceptionsbase_part_01",
        "FlextBaseErrorState": ".flextexceptionsbase_part_02",
        "FlextExceptionsBasePart03": ".flextexceptionsbase_part_03",
    }),
    public_exports=__all__,
)
