"""Flext Core. Typings. Type adapters parts package.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._typings._typeadapters_parts.flexttypestypeadapter_json_part_01 import (
        FlextTypesTypeAdapterJson,
    )
    from flext_core._typings._typeadapters_parts.flexttypestypeadapter_scalars_part_01 import (
        FlextTypesTypeAdapterScalars,
    )


__all__: tuple[str, ...] = (
    "FlextTypesTypeAdapterJson",
    "FlextTypesTypeAdapterScalars",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextTypesTypeAdapterJson": ".flexttypestypeadapter_json_part_01",
        "FlextTypesTypeAdapterScalars": ".flexttypestypeadapter_scalars_part_01",
    }),
    public_exports=__all__,
)
