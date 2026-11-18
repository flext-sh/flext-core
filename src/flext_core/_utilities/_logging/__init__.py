# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities. Logging package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._logging.async_writer import (
        FlextUtilitiesLoggingAsyncWriter,
    )
    from flext_core._utilities._logging.bound_logger import (
        FlextUtilitiesLoggingBoundLogger,
    )
    from flext_core._utilities._logging.composition import (
        FlextUtilitiesLoggingComposition,
    )
    from flext_core._utilities._logging.context import (
        FlextUtilitiesLoggingContextBinding,
    )
    from flext_core._utilities._logging.exception_info import (
        FlextUtilitiesLoggingExceptionInfo,
    )
    from flext_core._utilities._logging.handlers import FlextUtilitiesLoggingHandlers
    from flext_core._utilities._logging.levels import FlextUtilitiesLoggingLevels
    from flext_core._utilities._logging.processors import (
        FlextUtilitiesLoggingProcessors,
    )
    from flext_core._utilities._logging.state import FlextUtilitiesLoggingState
    from flext_core._utilities._logging.stream import FlextUtilitiesLoggingStream


__all__: tuple[str, ...] = (
    "FlextUtilitiesLoggingAsyncWriter",
    "FlextUtilitiesLoggingBoundLogger",
    "FlextUtilitiesLoggingComposition",
    "FlextUtilitiesLoggingContextBinding",
    "FlextUtilitiesLoggingExceptionInfo",
    "FlextUtilitiesLoggingHandlers",
    "FlextUtilitiesLoggingLevels",
    "FlextUtilitiesLoggingProcessors",
    "FlextUtilitiesLoggingState",
    "FlextUtilitiesLoggingStream",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextUtilitiesLoggingAsyncWriter": ".async_writer",
        "FlextUtilitiesLoggingBoundLogger": ".bound_logger",
        "FlextUtilitiesLoggingComposition": ".composition",
        "FlextUtilitiesLoggingContextBinding": ".context",
        "FlextUtilitiesLoggingExceptionInfo": ".exception_info",
        "FlextUtilitiesLoggingHandlers": ".handlers",
        "FlextUtilitiesLoggingLevels": ".levels",
        "FlextUtilitiesLoggingProcessors": ".processors",
        "FlextUtilitiesLoggingState": ".state",
        "FlextUtilitiesLoggingStream": ".stream",
    },
    public_exports=__all__,
    deferred_members=(
        "FlextUtilitiesLoggingComposition",
        "FlextUtilitiesLoggingContextBinding",
        "FlextUtilitiesLoggingExceptionInfo",
        "FlextUtilitiesLoggingHandlers",
        "FlextUtilitiesLoggingLevels",
        "FlextUtilitiesLoggingProcessors",
        "FlextUtilitiesLoggingState",
    ),
)
