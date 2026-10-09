# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Exceptions. Base Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_01 import (
        FlextBaseErrorMetadataMixin,
    )
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_02 import (
        FlextBaseErrorStateMixin,
    )
    from flext_core._exceptions._base_parts.flextexceptionsbase_part_03 import (
        FlextExceptionsBase,
    )


__all__: tuple[str, ...] = (
    "FlextBaseErrorMetadataMixin",
    "FlextBaseErrorStateMixin",
    "FlextExceptionsBase",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextBaseErrorMetadataMixin": ".flextexceptionsbase_part_01",
        "FlextBaseErrorStateMixin": ".flextexceptionsbase_part_02",
        "FlextExceptionsBase": ".flextexceptionsbase_part_03",
    }),
    public_exports=__all__,
)
