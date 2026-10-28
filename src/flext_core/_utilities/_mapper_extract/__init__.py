# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Mapper Extract package.

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
    from flext_core._utilities._mapper_extract.mapper_extract_part_01 import (
        FlextUtilitiesMapperExtractPart01,
    )
    from flext_core._utilities._mapper_extract.mapper_extract_part_02 import (
        FlextUtilitiesMapperExtractPart02,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesMapperExtractPart01",
    "FlextUtilitiesMapperExtractPart02",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesMapperExtractPart01": ".mapper_extract_part_01",
        "FlextUtilitiesMapperExtractPart02": ".mapper_extract_part_02",
    }),
    public_exports=__all__,
)
