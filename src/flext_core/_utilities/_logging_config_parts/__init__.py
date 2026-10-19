# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Logging Config Parts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._logging_config_parts.logging_config_part_03 import (
        FlextUtilitiesLoggingConfig,
    )
    from flext_core._utilities._logging_config_parts.logging_stream import (
        FlextUtilitiesLoggingStream,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesLoggingConfig",
    "FlextUtilitiesLoggingStream",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesLoggingConfig": ".logging_config_part_03",
        "FlextUtilitiesLoggingStream": ".logging_stream",
    }),
    public_exports=__all__,
)[0]
