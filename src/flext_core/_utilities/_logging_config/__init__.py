# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Logging Config package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._logging_config.logging_config_part_01 import (
        FlextUtilitiesLoggingConfigPart01,
    )
    from flext_core._utilities._logging_config.logging_config_part_02 import (
        FlextUtilitiesLoggingConfigPart02,
    )
    from flext_core._utilities._logging_config.logging_config_part_03 import (
        FlextUtilitiesLoggingConfigPart03,
    )
    from flext_core._utilities._logging_config.logging_stream import (
        FlextUtilitiesLoggingStream,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesLoggingConfigPart01",
    "FlextUtilitiesLoggingConfigPart02",
    "FlextUtilitiesLoggingConfigPart03",
    "FlextUtilitiesLoggingStream",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesLoggingConfigPart01": ".logging_config_part_01",
        "FlextUtilitiesLoggingConfigPart02": ".logging_config_part_02",
        "FlextUtilitiesLoggingConfigPart03": ".logging_config_part_03",
        "FlextUtilitiesLoggingStream": ".logging_stream",
    }),
    public_exports=__all__,
)
