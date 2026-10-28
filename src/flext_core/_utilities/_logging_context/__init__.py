# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Logging Context package.

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
    from flext_core._utilities._logging_context.logging_context_part_01 import (
        FlextUtilitiesLoggingContextPart01,
    )
    from flext_core._utilities._logging_context.logging_context_part_02 import (
        FlextUtilitiesLoggingContextPart02,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesLoggingContextPart01",
    "FlextUtilitiesLoggingContextPart02",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesLoggingContextPart01": ".logging_context_part_01",
        "FlextUtilitiesLoggingContextPart02": ".logging_context_part_02",
    }),
    public_exports=__all__,
)
